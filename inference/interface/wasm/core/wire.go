package core

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"math/bits"

	"github.com/cipherflow-fhe/lattigo/ckks"
)

type ciphertexts []*ckks.Ciphertext

func (f feature) marshal(params ckks.Parameters) []byte {
	var out bytes.Buffer
	write := func(v int) { binary.Write(&out, binary.LittleEndian, uint32(v)) }
	for _, v := range []int{f.dim, f.channels, f.perCT, f.level} {
		write(v)
	}
	if f.block != 0 {
		write(f.shape[0])
		write(f.shape[1])
		write(f.block)
	} else {
		switch f.dim {
		case 0:
			write(f.skip[0])
			write(f.skip[0])
		case 1:
			write(f.shape[0])
			write(f.skip[0])
			write(f.fill)
		case 2:
			write(f.shape[0])
			write(f.shape[1])
			write(f.skip[0])
			write(f.skip[1])
		}
	}
	write(len(f.ciphertexts))
	for _, ct := range f.ciphertexts {
		raw := ct.ToBytes(&params.Parameters)
		binary.Write(&out, binary.LittleEndian, uint64(len(raw))) // native 64-bit size_t
		out.Write(raw)
	}
	write(0) // uncompressed ciphertexts
	if f.dim == 0 {
		for i := 0; i < 4; i++ {
			write(0)
		}
	}
	return out.Bytes()
}

func unmarshalFeature(raw []byte, params ckks.Parameters, matrix bool) (f feature, err error) {
	r := bytes.NewReader(raw)
	read := func() int {
		var n uint32
		if err == nil {
			err = binary.Read(r, binary.LittleEndian, &n)
		}
		return int(n)
	}
	f.dim, f.channels, f.perCT, f.level = read(), read(), read(), read()
	f.skip = []int{1, 1}
	if matrix {
		f.shape = []int{read(), read()}
		f.block = read()
	} else {
		switch f.dim {
		case 0:
			f.skip = []int{read(), read()}
		case 1:
			f.shape = []int{read()}
			f.skip[0] = read()
			f.fill = read()
		case 2:
			f.shape = []int{read(), read()}
			f.skip = []int{read(), read()}
		default:
			return f, errors.New("invalid feature dimension")
		}
	}
	count := read()
	if err != nil {
		return f, err
	}
	if count < 1 || count > r.Len()/8 || f.level > params.MaxLevel() {
		return f, errors.New("invalid ciphertext count or level")
	}
	for i := 0; i < count; i++ {
		var size uint64
		if err = binary.Read(r, binary.LittleEndian, &size); err != nil {
			return f, err
		}
		if size > uint64(r.Len()) {
			return f, io.ErrUnexpectedEOF
		}
		data := make([]byte, int(size)+1)
		if _, err = io.ReadFull(r, data[:size]); err != nil {
			return f, err
		}
		if err = validateCiphertext(data[:size], params); err != nil {
			return f, err
		}
		ct := new(ckks.Ciphertext)
		ct.FromBytes(data) // legacy decoder's byte lookahead has a sentinel
		f.ciphertexts = append(f.ciphertexts, ct)
	}
	if read() != 0 {
		return f, errors.New("expected native uncompressed ciphertexts")
	}
	if f.dim == 0 && r.Len() != 0 {
		if read() != 0 {
			return f, errors.New("invalid scalar batch layout")
		}
		read()
		read()
		read()
	}
	if err != nil {
		return f, err
	}
	if r.Len() != 0 {
		return f, errors.New("trailing ciphertext bytes")
	}
	return f, nil
}

// Bound legacy bit decoder allocations using the task's trusted parameters.
func validateCiphertext(raw []byte, params ckks.Parameters) error {
	if len(raw) < 15 || binary.LittleEndian.Uint32(raw[8:12]) != uint32(params.N()) || raw[12] != 2 || int(raw[13]) > params.MaxLevel() {
		return errors.New("invalid CKKS ciphertext header")
	}
	scale := math.Float64frombits(binary.LittleEndian.Uint64(raw[:8]))
	if scale <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		return errors.New("invalid ciphertext scale")
	}
	level := int(raw[13])
	offset := 15 + level
	if offset > len(raw) {
		return io.ErrUnexpectedEOF
	}
	for i := 0; i <= level; i++ {
		if raw[14+i] != byte(bits.Len64(params.Q()[i])) {
			return errors.New("ciphertext modulus mismatch")
		}
	}
	for poly := 0; poly < 2; poly++ {
		if offset+3 > len(raw) {
			return io.ErrUnexpectedEOF
		}
		drop := int(raw[offset+2])
		offset += 3
		for i := 0; i <= level; i++ {
			width := int(raw[14+i])
			if level == 0 {
				width -= drop
			}
			if width < 1 || width > 64 {
				return errors.New("invalid ciphertext bit width")
			}
			offset += params.N() * width / 8
		}
	}
	if offset != len(raw) {
		return errors.New("ciphertext payload length mismatch")
	}
	return nil
}
