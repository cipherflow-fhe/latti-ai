package core

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"

	"github.com/cipherflow-fhe/lattigo/core/rlwe"
	"github.com/cipherflow-fhe/lattigo/schemes/ckks"
)

type ciphertexts []*rlwe.Ciphertext

// validateCiphertext checks native v6 vector dimensions before ReadFrom can
// allocate from untrusted lengths. Metadata uses the v6 parser and fixed size.
func validateCiphertext(raw []byte, params ckks.Parameters) error {
	r := bytes.NewReader(raw)
	flag, err := r.ReadByte()
	if err != nil {
		return err
	}
	if flag != 1 {
		return errors.New("CKKS ciphertext metadata required")
	}
	var metadata rlwe.MetaData
	data := make([]byte, metadata.BinarySize())
	if _, err = io.ReadFull(r, data); err != nil {
		return err
	}
	if err = metadata.UnmarshalBinary(data); err != nil {
		return err
	}
	scale := metadata.Scale.Float64()
	if !metadata.IsBatched || !metadata.IsNTT || metadata.IsMontgomery ||
		metadata.LogDimensions.Rows != 0 || metadata.LogDimensions.Cols < 0 || metadata.LogDimensions.Cols > params.LogMaxSlots() ||
		math.IsNaN(scale) || math.IsInf(scale, 0) || scale <= 0 {
		return errors.New("invalid CKKS ciphertext metadata")
	}
	read := func() (uint64, error) { var n uint64; err := binary.Read(r, binary.LittleEndian, &n); return n, err }
	polys, err := read()
	if err != nil {
		return err
	}
	if polys < 2 || polys > 3 {
		return errors.New("invalid ciphertext degree")
	}
	var levelCount uint64
	for range polys {
		levels, err := read()
		if err != nil {
			return err
		}
		if levels < 1 || levels > uint64(params.QCount()) || levelCount != 0 && levels != levelCount {
			return errors.New("invalid ciphertext modulus count")
		}
		levelCount = levels
		for range levels {
			n, err := read()
			if err != nil {
				return err
			}
			if n != uint64(params.N()) {
				return errors.New("invalid ciphertext ring degree")
			}
			length := n * 8
			if length > uint64(r.Len()) {
				return io.ErrUnexpectedEOF
			}
			if _, err := r.Seek(int64(length), io.SeekCurrent); err != nil {
				return err
			}
		}
	}
	if r.Len() != 0 {
		return errors.New("trailing ciphertext bytes")
	}
	return nil
}
