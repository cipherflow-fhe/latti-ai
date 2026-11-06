package core

import (
	"errors"
	"github.com/cipherflow-fhe/lattigo/schemes/ckks"
	"math"
)

func (c *Client) Encrypt(name string, values []float64) ([]byte, error) {
	if len(c.Task.Inputs) == 0 {
		return nil, errors.New("task configuration is required for encryption")
	}
	if c.encryptor == nil {
		return nil, errors.New("setup or restore required")
	}
	p, ok := c.Task.Inputs[name]
	if !ok {
		return nil, errors.New("unknown input name")
	}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, errors.New("non-finite input")
		}
	}
	f, slots, err := c.pack(p, values)
	if err != nil {
		return nil, err
	}
	for _, vector := range slots {
		pt := ckks.NewPlaintext(c.Keys.Params, p.Level)
		pt.Scale = c.Keys.Params.DefaultScale()
		if err := c.encoder.Encode(vector, pt); err != nil {
			return nil, err
		}
		ct, err := c.encryptor.EncryptNew(pt)
		if err != nil {
			return nil, err
		}
		f.ciphertexts = append(f.ciphertexts, ct)
	}
	return f.marshal(c.Keys.Params)
}

func (c *Client) Decrypt(name string, raw []byte) (values []float64, err error) {
	if len(c.Task.Outputs) == 0 {
		return nil, errors.New("task configuration is required for decryption")
	}
	if c.decryptor == nil {
		return nil, errors.New("setup or restore required")
	}
	p, ok := c.Task.Outputs[name]
	if !ok {
		return nil, errors.New("unknown output name")
	}
	f, err := unmarshalFeature(raw, c.Keys.Params, p.DataType == "feature_mat")
	if err != nil {
		return nil, err
	}
	if f.dim != p.Dim {
		return nil, errors.New("output dimension mismatch")
	}
	slots := make([][]float64, len(f.ciphertexts))
	for i, ct := range f.ciphertexts {
		decoded := make([]complex128, c.Keys.Params.MaxSlots())
		if err := c.encoder.Decode(c.decryptor.DecryptNew(ct), decoded); err != nil {
			return nil, err
		}
		slots[i] = make([]float64, len(decoded))
		for j, value := range decoded {
			slots[i][j] = real(value)
		}
	}
	return c.unpack(p, f, slots)
}
