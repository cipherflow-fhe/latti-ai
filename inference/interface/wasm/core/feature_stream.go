package core

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"slices"
	"sort"

	"github.com/cipherflow-fhe/lattigo/clientcontext"
	"github.com/cipherflow-fhe/lattigo/core/rlwe"
	"github.com/cipherflow-fhe/lattigo/schemes/ckks"
)

func (f feature) writeHeader(w io.Writer, count int) error {
	fields := []int{f.dim, f.channels, f.perCT, f.level}
	if f.block != 0 {
		fields = append(fields, f.shape[0], f.shape[1], f.block)
	} else {
		switch f.dim {
		case 0:
			fields = append(fields, f.skip[0], f.skip[0])
		case 1:
			fields = append(fields, f.shape[0], f.skip[0], f.fill)
		case 2:
			fields = append(fields, f.shape[0], f.shape[1], f.skip[0], f.skip[1])
		default:
			return errors.New("invalid feature dimension")
		}
	}
	fields = append(fields, count)
	for _, v := range fields {
		if v < 0 || uint64(v) > math.MaxUint32 {
			return errors.New("feature field overflow")
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(v)); err != nil {
			return err
		}
	}
	return nil
}

func readFeatureHeader(r io.Reader, matrix bool, params ckks.Parameters) (f feature, count int, err error) {
	read := func() int {
		var v uint32
		if err == nil {
			err = binary.Read(r, binary.LittleEndian, &v)
		}
		if v > 1<<26 {
			err = errors.New("feature field exceeds layout budget")
			return 0
		}
		return int(v)
	}
	f.dim, f.channels, f.perCT, f.level = read(), read(), read(), read()
	f.skip = []int{1, 1}
	if matrix {
		f.shape = []int{read(), read()}
		f.block = read()
		if f.dim != 2 || f.block < 1 {
			return f, 0, errors.New("invalid matrix header")
		}
	} else {
		switch f.dim {
		case 0:
			f.skip[0] = read()
			_ = read()
		case 1:
			f.shape = []int{read()}
			f.skip[0] = read()
			f.fill = read()
		case 2:
			f.shape = []int{read(), read()}
			f.skip = []int{read(), read()}
		default:
			return f, 0, errors.New("invalid feature dimension")
		}
	}
	count = read()
	if err != nil {
		return f, 0, err
	}
	if count < 1 || f.level > params.MaxLevel() || !matrix && (f.channels < 1 || f.perCT < 1 || f.perCT > params.MaxSlots() || product(f.skip) == 0) {
		return f, 0, errors.New("invalid feature header")
	}
	return f, count, nil
}

// StartEncrypt consumes staged little-endian float64 values, retaining one
// plaintext/ciphertext and one bounded OPFS source block at a time.
func (c *Client) StartEncrypt(name string, bytes uint64) (*OutputStream, error) {
	p, ok := c.Task.Inputs[name]
	if !ok || c.encryptor == nil {
		return nil, errors.New("bound input and client keys required")
	}
	if bytes == 0 || bytes%8 != 0 || bytes/8 > 1<<26 {
		return nil, errors.New("invalid numeric input length")
	}
	valueCount := int(bytes / 8)
	f, count, err := c.packingWalk(p, valueCount, -1, nil)
	if err != nil {
		return nil, err
	}
	return startOutput(func(s *OutputStream) error {
		source := &inputSource{stream: s, id: "values", size: bytes}
		return s.output("ciphertext", func(sink clientcontext.Sink) (uint64, error) {
			out := bufio.NewWriterSize(sink, 64*1024)
			if err := f.writeHeader(out, count); err != nil {
				return 0, err
			}
			for current := 0; current < count; current++ {
				vector := make([]float64, c.Keys.Params.MaxSlots())
				var inputErr error
				_, _, err := c.packingWalk(p, valueCount, current, func(ct, slot, index int) {
					if ct != current || inputErr != nil {
						return
					}
					if slot < 0 || slot >= len(vector) || index < 0 || index >= valueCount {
						inputErr = errors.New("packing exceeds layout")
						return
					}
					data, err := source.readAt(uint64(index)*8, 8)
					if err != nil {
						inputErr = err
						return
					}
					value := math.Float64frombits(binary.LittleEndian.Uint64(data))
					if math.IsNaN(value) || math.IsInf(value, 0) {
						inputErr = errors.New("non-finite input")
						return
					}
					vector[slot] = value
				})
				if err != nil {
					return 0, err
				}
				if inputErr != nil {
					return 0, inputErr
				}
				pt := ckks.NewPlaintext(c.Keys.Params, p.Level)
				pt.Scale = c.Keys.Params.DefaultScale()
				if err = c.encoder.Encode(vector, pt); err != nil {
					return 0, err
				}
				ct, err := c.encryptor.EncryptNew(pt)
				if err != nil {
					return 0, err
				}
				if err = binary.Write(out, binary.LittleEndian, uint64(ct.BinarySize())); err != nil {
					return 0, err
				}
				if _, err = ct.WriteTo(out); err != nil {
					return 0, err
				}
			}
			if f.dim == 0 {
				if _, err := out.Write(make([]byte, 16)); err != nil {
					return 0, err
				}
			}
			if err := out.Flush(); err != nil {
				return 0, err
			}
			return sink.(*outputSink).pos, nil
		})
	}), nil
}

type decodedValue struct {
	index int
	value float64
}

// StartDecrypt stages float64 output on disk. Sparse packing is rewritten as
// bounded contiguous patches, never as a complete decoded Feature array.
func (c *Client) StartDecrypt(name string, bytes uint64) (*OutputStream, error) {
	p, ok := c.Task.Outputs[name]
	if !ok || c.decryptor == nil {
		return nil, errors.New("bound output and client keys required")
	}
	return startOutput(func(s *OutputStream) error {
		source := &inputSource{stream: s, id: "ciphertext", size: bytes}
		f, count, err := readFeatureHeader(source, p.DataType == "feature_mat", c.Keys.Params)
		if err != nil {
			return err
		}
		if f.dim != p.Dim || p.Dim > 0 && p.DataType != "feature_mat" && !slices.Equal(f.shape, p.Shape) {
			return errors.New("output shape differs from task")
		}
		valueCount, err := c.unpackingWalk(p, f, count, -1, nil)
		if err != nil {
			return err
		}
		if p.DataType == "feature_mat" && p.SelectFirstToken {
			channels := p.Shape[0]
			if p.Transposed != nil && !*p.Transposed {
				channels = p.Shape[1]
			}
			if valueCount < channels {
				return errors.New("CLS output is shorter than channel count")
			}
			valueCount = channels
		}
		if valueCount < 1 || valueCount > 1<<26 || uint64(count) > bytes/8 {
			return errors.New("output size exceeds budget")
		}
		return s.output("values", func(sink clientcontext.Sink) (uint64, error) {
			outputBytes := uint64(valueCount) * 8
			zeros := make([]byte, min(outputBytes, uint64(clientcontext.MaxChunkBytes)))
			for remaining := outputBytes; remaining > 0; {
				n := min(remaining, uint64(len(zeros)))
				if _, err = sink.Write(zeros[:int(n)]); err != nil {
					return 0, err
				}
				remaining -= n
			}
			var md rlwe.MetaData
			maxCT := uint64(1+md.BinarySize()+8) + 3*(8+uint64(c.Keys.Params.QCount())*(8+uint64(c.Keys.Params.N())*8))
			for current := 0; current < count; current++ {
				var size uint64
				if err = binary.Read(source, binary.LittleEndian, &size); err != nil {
					return 0, err
				}
				if size > maxCT || size > source.size-source.position {
					return 0, errors.New("ciphertext length exceeds parameter bound")
				}
				raw := make([]byte, int(size))
				if _, err = io.ReadFull(source, raw); err != nil {
					return 0, err
				}
				if err = validateCiphertext(raw, c.Keys.Params); err != nil {
					return 0, err
				}
				ct := new(rlwe.Ciphertext)
				if err = ct.UnmarshalBinary(raw); err != nil {
					return 0, err
				}
				if ct.Level() != f.level {
					return 0, errors.New("ciphertext level differs from feature header")
				}
				decoded := make([]complex128, c.Keys.Params.MaxSlots())
				if err = c.encoder.Decode(c.decryptor.DecryptNew(ct), decoded); err != nil {
					return 0, err
				}
				values := make([]decodedValue, 0, c.Keys.Params.MaxSlots())
				var layoutErr error
				_, err = c.unpackingWalk(p, f, count, current, func(ct, slot, index int) {
					if ct != current || layoutErr != nil || p.SelectFirstToken && index >= valueCount {
						return
					}
					if slot < 0 || slot >= len(decoded) || index < 0 || index >= valueCount || len(values) >= c.Keys.Params.MaxSlots() {
						layoutErr = errors.New("output exceeds current ciphertext workspace")
						return
					}
					value := real(decoded[slot])
					if math.IsNaN(value) || math.IsInf(value, 0) {
						layoutErr = errors.New("non-finite decoded output")
						return
					}
					values = append(values, decodedValue{index, value})
				})
				if err != nil {
					return 0, err
				}
				if layoutErr != nil {
					return 0, layoutErr
				}
				sort.Slice(values, func(i, j int) bool { return values[i].index < values[j].index })
				for begin := 0; begin < len(values); {
					end := begin + 1
					for end < len(values) && values[end].index == values[end-1].index+1 && (end-begin)*8 < clientcontext.MaxChunkBytes {
						end++
					}
					chunk := make([]byte, (end-begin)*8)
					for i, v := range values[begin:end] {
						binary.LittleEndian.PutUint64(chunk[i*8:], math.Float64bits(v.value))
					}
					if err = sink.Patch(uint64(values[begin].index)*8, chunk); err != nil {
						return 0, err
					}
					begin = end
				}
			}
			remaining := source.size - source.position
			if f.dim == 0 && remaining != 0 {
				if remaining != 16 {
					return 0, errors.New("invalid scalar batch metadata length")
				}
				var batch [4]uint32
				if err = binary.Read(source, binary.LittleEndian, &batch); err != nil {
					return 0, err
				}
				if batch[0] > 1 {
					return 0, errors.New("invalid scalar batch metadata flag")
				}
			}
			if source.position != source.size {
				return 0, errors.New("trailing feature bytes")
			}
			return outputBytes, nil
		})
	}), nil
}
