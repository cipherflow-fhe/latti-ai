package core

import (
	"bytes"
	"encoding/binary"
	"errors"
	"github.com/cipherflow-fhe/lattigo/core/rlwe"
	"github.com/cipherflow-fhe/lattigo/schemes/ckks"
	"io"
)

func (f feature) marshal(params ckks.Parameters) ([]byte, error) {
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
		raw, err := ct.MarshalBinary()
		if err != nil {
			return nil, err
		}
		binary.Write(&out, binary.LittleEndian, uint64(len(raw))) // native 64-bit size_t
		out.Write(raw)
	}
	if f.dim == 0 {
		for i := 0; i < 4; i++ {
			write(0)
		}
	}
	return out.Bytes(), nil
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
			// Native scalar features repeat the same skip value in two wire
			// positions. The second position is legacy metadata, not a
			// second logical dimension.
			first := read()
			_ = read()
			f.skip = []int{first, 1}
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
		data := raw[len(raw)-r.Len() : len(raw)-r.Len()+int(size)]
		if _, err = r.Seek(int64(size), io.SeekCurrent); err != nil {
			return f, err
		}
		if err = validateCiphertext(data, params); err != nil {
			return f, err
		}
		ct := new(rlwe.Ciphertext)
		if err = ct.UnmarshalBinary(data); err != nil {
			return f, err
		}
		if ct.Level() != f.level {
			return f, errors.New("ciphertext level differs from feature header")
		}
		f.ciphertexts = append(f.ciphertexts, ct)
	}
	if f.dim == 0 && r.Len() != 0 {
		if read() > 1 {
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
