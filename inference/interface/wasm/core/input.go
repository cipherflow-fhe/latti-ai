package core

import (
	"errors"
	"io"
	"strconv"

	"github.com/cipherflow-fhe/lattigo/clientcontext"
)

// inputSource reads only from a Worker-owned OPFS snapshot. At most one bounded
// block is live; all offsets are uint64 until a checked local slice conversion.
type inputSource struct {
	stream         *OutputStream
	id             string
	size, position uint64
	cache          []byte
	cacheAt        uint64
}

func (r *inputSource) readAt(offset uint64, n int) ([]byte, error) {
	if n < 0 || n > clientcontext.MaxChunkBytes || offset > r.size || uint64(n) > r.size-offset {
		return nil, io.ErrUnexpectedEOF
	}
	if n == 0 {
		return nil, nil
	}
	if len(r.cache) > 0 && offset >= r.cacheAt && offset-r.cacheAt+uint64(n) <= uint64(len(r.cache)) {
		at := int(offset - r.cacheAt)
		return r.cache[at : at+n], nil
	}
	want := min(uint64(clientcontext.MaxChunkBytes), r.size-offset)
	reply, err := r.stream.exchange(OutputEvent{Type: "need-input", SourceID: r.id, Offset: strconv.FormatUint(offset, 10), MaxBytes: int(want)})
	if err != nil {
		return nil, err
	}
	if uint64(len(reply.Data)) != want {
		return nil, errors.New("short OPFS input read")
	}
	if reply.EOF != (offset+want == r.size) {
		return nil, errors.New("input EOF does not match snapshot size")
	}
	r.cacheAt, r.cache = offset, reply.Data
	return r.cache[:n], nil
}
func (r *inputSource) Read(p []byte) (int, error) {
	if r.position == r.size {
		return 0, io.EOF
	}
	n := min(len(p), clientcontext.MaxChunkBytes, int(min(r.size-r.position, uint64(clientcontext.MaxChunkBytes))))
	b, err := r.readAt(r.position, n)
	if err != nil {
		return 0, err
	}
	copy(p, b)
	r.position += uint64(n)
	return n, nil
}
