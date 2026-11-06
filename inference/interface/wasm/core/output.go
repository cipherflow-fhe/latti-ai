package core

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strconv"

	"github.com/cipherflow-fhe/lattigo/clientcontext"
)

// OutputEvent borrows Data until the consumer acknowledges it with Next.
// Byte positions cross the JS boundary as unsigned decimal strings.
type OutputEvent struct {
	Type     string
	OutputID string
	Offset   string
	Data     []byte
	Bytes    string
	Err      error
	SourceID string
	MaxBytes int
	Role     string
	Format   string
}

// OutputStream permits exactly one unacknowledged event. The producer cannot
// reuse a write buffer or generate the next key until the sink acknowledges it.
type OutputStream struct {
	ctx     context.Context
	cancel  context.CancelFunc
	events  chan OutputEvent
	ack     chan inputAck
	done    chan struct{}
	waiting bool
	ended   bool
}

func startOutput(run func(*OutputStream) error) *OutputStream {
	ctx, cancel := context.WithCancel(context.Background())
	s := &OutputStream{ctx: ctx, cancel: cancel, events: make(chan OutputEvent), ack: make(chan inputAck), done: make(chan struct{})}
	go func() {
		defer close(s.done)
		err := run(s)
		typ := "done"
		if err != nil {
			typ = "error"
		}
		if errors.Is(err, context.Canceled) {
			typ = "cancelled"
		}
		select {
		case s.events <- OutputEvent{Type: typ, Err: err}:
		case <-s.ctx.Done():
		}
	}()
	return s
}

type inputAck struct {
	Err  error
	Data []byte
	EOF  bool
}

func (s *OutputStream) exchange(event OutputEvent) (inputAck, error) {
	select {
	case <-s.ctx.Done():
		return inputAck{}, s.ctx.Err()
	case s.events <- event:
	}
	select {
	case <-s.ctx.Done():
		return inputAck{}, s.ctx.Err()
	case reply := <-s.ack:
		return reply, reply.Err
	}
}
func (s *OutputStream) emit(event OutputEvent) error { _, err := s.exchange(event); return err }

// Next acknowledges the previous event, including an I/O failure, before
// pulling the next one. Calls on a stream must be serialized by its owner.
func (s *OutputStream) Next(ack error) (OutputEvent, error) {
	return s.next(inputAck{Err: ack})
}

// NextInput supplies the one outstanding bounded read, after Worker disk I/O.
func (s *OutputStream) NextInput(ack error, data []byte, eof bool) (OutputEvent, error) {
	if len(data) > clientcontext.MaxChunkBytes {
		return OutputEvent{}, errors.New("input chunk too large")
	}
	return s.next(inputAck{Err: ack, Data: data, EOF: eof})
}
func (s *OutputStream) next(ack inputAck) (OutputEvent, error) {
	if s.ended {
		return OutputEvent{}, io.EOF
	}
	if s.waiting {
		select {
		case s.ack <- ack:
		case <-s.ctx.Done():
		}
	}
	select {
	case event := <-s.events:
		s.waiting = event.Type != "done" && event.Type != "error" && event.Type != "cancelled"
		s.ended = !s.waiting
		if s.ended {
			<-s.done
			s.cancel()
		}
		return event, nil
	case <-s.ctx.Done():
		<-s.done
		s.ended = true
		return OutputEvent{Type: "cancelled", Err: s.ctx.Err()}, nil
	}
}

// Cancel stops blocked I/O immediately; an active cryptographic primitive
// finishes before the producer exits. Next returns the cancellation terminal.
func (s *OutputStream) Cancel() { s.cancel() }

type outputSink struct {
	stream *OutputStream
	id     string
	pos    uint64
}

func (w *outputSink) Write(data []byte) (int, error) {
	total := 0
	for len(data) != 0 {
		n := min(len(data), clientcontext.MaxChunkBytes)
		if uint64(n) > ^uint64(0)-w.pos {
			return total, fmt.Errorf("output length overflow")
		}
		if err := w.stream.emit(OutputEvent{Type: "write-chunk", OutputID: w.id, Offset: strconv.FormatUint(w.pos, 10), Data: data[:n]}); err != nil {
			return total, err
		}
		w.pos += uint64(n)
		total += n
		data = data[n:]
	}
	return total, nil
}

func (w *outputSink) Patch(offset uint64, data []byte) error {
	if offset > w.pos || uint64(len(data)) > w.pos-offset || len(data) > clientcontext.MaxChunkBytes {
		return fmt.Errorf("patch outside reserved output")
	}
	return w.stream.emit(OutputEvent{Type: "write-at", OutputID: w.id, Offset: strconv.FormatUint(offset, 10), Data: data})
}

func (s *OutputStream) output(id string, write func(clientcontext.Sink) (uint64, error)) error {
	role, format := "ciphertext", "native-feature-v6"
	switch id {
	case "eval":
		role, format = "public-eval", "CKKSCTX1"
	case "full":
		role, format = "private-full", "client-full-v6"
	case "values":
		role, format = "plaintext", "float64-le"
	}
	if err := s.emit(OutputEvent{Type: "begin-output", OutputID: id, Role: role, Format: format}); err != nil {
		return err
	}
	w := &outputSink{stream: s, id: id}
	n, err := write(w)
	if err != nil {
		return err
	}
	if n != w.pos {
		return fmt.Errorf("output length mismatch")
	}
	return s.emit(OutputEvent{Type: "end-output", OutputID: id, Bytes: strconv.FormatUint(n, 10)})
}
