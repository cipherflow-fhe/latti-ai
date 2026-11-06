package core

import (
	"bytes"
	"errors"
	"io"
	"os"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cipherflow-fhe/lattigo/clientcontext"
	"github.com/cipherflow-fhe/lattigo/ring"
	"github.com/cipherflow-fhe/lattigo/schemes/ckks"
)

func fullBytes(t *testing.T, c *Client) ([]byte, error) {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "full-")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if _, err = c.WriteFull(clientcontext.FileSink{Writer: f, WriterAt: f}); err != nil {
		return nil, err
	}
	return os.ReadFile(f.Name())
}

func streamEvent(t *testing.T, s *OutputStream, ack error) OutputEvent {
	t.Helper()
	type result struct {
		event OutputEvent
		err   error
	}
	ch := make(chan result, 1)
	go func() { event, err := s.Next(ack); ch <- result{event, err} }()
	select {
	case r := <-ch:
		if r.err != nil {
			t.Fatal(r.err)
		}
		return r.event
	case <-time.After(5 * time.Second):
		s.Cancel()
		t.Fatal("stream did not settle")
		return OutputEvent{}
	}
}

func TestOutputBackpressureAndPatch(t *testing.T) {
	var advanced atomic.Bool
	s := startOutput(func(s *OutputStream) error {
		return s.output("test", func(sink clientcontext.Sink) (uint64, error) {
			if _, err := sink.Write([]byte("0000payload")); err != nil {
				return 0, err
			}
			advanced.Store(true)
			return 11, sink.Patch(0, []byte("0007"))
		})
	})
	defer s.Cancel()
	if event := streamEvent(t, s, nil); event.Type != "begin-output" {
		t.Fatal(event)
	}
	event := streamEvent(t, s, nil)
	if event.Type != "write-chunk" || event.Offset != "0" || string(event.Data) != "0000payload" {
		t.Fatal(event)
	}
	if advanced.Load() {
		t.Fatal("producer advanced before disk acknowledgement")
	}
	event = streamEvent(t, s, nil)
	if event.Type != "write-at" || event.Offset != "0" || string(event.Data) != "0007" {
		t.Fatal(event)
	}
	event = streamEvent(t, s, nil)
	if event.Type != "end-output" || event.Bytes != "11" {
		t.Fatal(event)
	}
	if event = streamEvent(t, s, nil); event.Type != "done" {
		t.Fatal(event)
	}
	if _, err := s.Next(nil); !errors.Is(err, io.EOF) {
		t.Fatal(err)
	}
}

func TestOutputFailureAndCancellation(t *testing.T) {
	for _, cancel := range []bool{false, true} {
		s := startOutput(func(s *OutputStream) error {
			w := &outputSink{stream: s, id: "test"}
			_, err := w.Write(make([]byte, clientcontext.MaxChunkBytes+1))
			return err
		})
		event := streamEvent(t, s, nil)
		if event.Type != "write-chunk" || len(event.Data) != clientcontext.MaxChunkBytes {
			t.Fatal(event.Type, len(event.Data))
		}
		failure := errors.New("disk full")
		if cancel {
			s.Cancel()
		}
		event = streamEvent(t, s, failure)
		if cancel {
			if event.Type != "cancelled" {
				t.Fatal(event)
			}
		} else if event.Type != "error" || !errors.Is(event.Err, failure) {
			t.Fatal(event)
		}
		if _, err := s.Next(nil); !errors.Is(err, io.EOF) {
			t.Fatal(err)
		}
	}
}

func TestStreamedContextFilesRestoreOriginalKeys(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 10, LogQ: []int{55, 40}, LogP: []int{55}, LogDefaultScale: 40, RingType: ring.ConjugateInvariant})
	if err != nil {
		t.Fatal(err)
	}
	c := &Client{Keys: keyState{Params: params}, plan: planFor(false)}
	s, err := c.StartSetup()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Cancel()
	files := map[string]*os.File{}
	positions := map[string]uint64{}
	defer func() {
		for _, f := range files {
			f.Close()
		}
	}()
	for {
		event := streamEvent(t, s, nil)
		switch event.Type {
		case "begin-output":
			f, err := os.CreateTemp(t.TempDir(), event.OutputID)
			if err != nil {
				t.Fatal(err)
			}
			files[event.OutputID] = f
		case "write-chunk", "write-at":
			offset, err := strconv.ParseUint(event.Offset, 10, 64)
			if err != nil || len(event.Data) > clientcontext.MaxChunkBytes {
				t.Fatal(event)
			}
			if event.Type == "write-chunk" {
				if offset != positions[event.OutputID] {
					t.Fatal("append cursor changed by patch")
				}
				positions[event.OutputID] += uint64(len(event.Data))
			} else if offset+uint64(len(event.Data)) > positions[event.OutputID] {
				t.Fatal("unreserved patch")
			}
			if _, err = files[event.OutputID].WriteAt(event.Data, int64(offset)); err != nil {
				t.Fatal(err)
			}
		case "end-output":
			if event.Bytes != strconv.FormatUint(positions[event.OutputID], 10) {
				t.Fatal(event)
			}
		case "done":
			full, err := os.ReadFile(files["full"].Name())
			if err != nil {
				t.Fatal(err)
			}
			restored, err := clientcontext.Restore(full, c.plan)
			if err != nil {
				t.Fatal(err)
			}
			if !restored.PublicKey.Equal(c.Keys.PK) || !restored.SecretKey.Equal(c.Keys.SK) {
				t.Fatal("keys changed")
			}
			fresh := &Client{Keys: keyState{Params: params}, plan: c.plan}
			runFeatureStream(t, fresh.StartRestore(uint64(len(full))), full)
			if !fresh.Keys.SK.Equal(c.Keys.SK) || !fresh.Keys.PK.Equal(c.Keys.PK) {
				t.Fatal("streaming restore changed keys")
			}
			public, err := os.ReadFile(files["eval"].Name())
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.HasPrefix(public, []byte("CKKSCTX1")) || len(public) <= len(full) {
				t.Fatal("invalid evaluation output")
			}
			return
		default:
			t.Fatal(event.Type, event.Err)
		}
	}
}
