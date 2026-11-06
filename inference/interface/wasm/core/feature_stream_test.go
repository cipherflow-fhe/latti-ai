package core

import (
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"testing"
)

func runFeatureStream(t *testing.T, s *OutputStream, source []byte) []byte {
	t.Helper()
	defer s.Cancel()
	out, err := os.CreateTemp(t.TempDir(), "feature-")
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	event := streamEvent(t, s, nil)
	for {
		switch event.Type {
		case "need-input":
			at, err := strconv.ParseUint(event.Offset, 10, 64)
			if err != nil || at > uint64(len(source)) || event.MaxBytes < 1 || event.MaxBytes > 4*1024*1024 || uint64(event.MaxBytes) > uint64(len(source))-at {
				t.Fatal("invalid bounded read", event)
			}
			data := append([]byte{}, source[at:at+uint64(event.MaxBytes)]...)
			event, err = s.NextInput(nil, data, at+uint64(len(data)) == uint64(len(source)))
			if err != nil {
				t.Fatal(err)
			}
			continue
		case "begin-output":
		case "write-chunk", "write-at":
			at, err := strconv.ParseInt(event.Offset, 10, 64)
			if err != nil || len(event.Data) > 4*1024*1024 {
				t.Fatal("invalid bounded write", event)
			}
			if _, err = out.WriteAt(event.Data, at); err != nil {
				t.Fatal(err)
			}
		case "end-output":
			info, err := out.Stat()
			if err != nil || strconv.FormatInt(info.Size(), 10) != event.Bytes {
				t.Fatal("stream length mismatch", event, err)
			}
		case "done":
			raw, err := os.ReadFile(out.Name())
			if err != nil {
				t.Fatal(err)
			}
			return raw
		default:
			t.Fatal("stream failed", event.Type, event.Err)
		}
		event = streamEvent(t, s, nil)
	}
}

func checkFeatureStreams(t *testing.T, c *Client, values []float64) {
	t.Helper()
	input := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(input[i*8:], math.Float64bits(v))
	}
	encrypt, err := c.StartEncrypt("input", uint64(len(input)))
	if err != nil {
		t.Fatal(err)
	}
	raw := runFeatureStream(t, encrypt, input)
	// The independent whole-feature test reader checks the native wire shape.
	got, err := c.Decrypt("output", raw)
	if err != nil {
		t.Fatal(err)
	}
	decrypt, err := c.StartDecrypt("output", uint64(len(raw)))
	if err != nil {
		t.Fatal(err)
	}
	output := runFeatureStream(t, decrypt, raw)
	if len(output) != len(input) || len(got) != len(values) {
		t.Fatal("stream output size mismatch")
	}
	for i, want := range values {
		value := math.Float64frombits(binary.LittleEndian.Uint64(output[i*8:]))
		if math.Abs(value-want) > 0.001 || math.Abs(value-got[i]) > 1e-8 {
			t.Fatalf("stream value %d: %g want %g", i, value, want)
		}
	}
}
