package core

import (
	"encoding/binary"
	"testing"

	"github.com/cipherflow-fhe/lattigo/core/rlwe"
)

func TestV6WireTailAndAllocationBounds(t *testing.T) {
	for _, feature := range []string{
		`{"dim":0,"channel":1,"pack_num":8,"skip":512,"level":1,"ckks_parameter_id":"p"}`,
		`{"dim":1,"channel":1,"shape":[2],"level":1,"ckks_parameter_id":"p"}`,
		`{"dim":2,"channel":1,"shape":[2,2],"level":1,"ckks_parameter_id":"p"}`,
	} {
		c, err := NewClient(configFor(t, feature, "ordinary", ""), testParameters(t, "p", 1))
		if err != nil {
			t.Fatal(err)
		}
		if err = c.GenerateKeys(); err != nil {
			t.Fatal(err)
		}
		p := c.Task.Inputs["input"]
		count := 1
		if p.Dim > 0 {
			count = product(p.Shape)
		}
		raw, err := c.Encrypt("input", make([]float64, count))
		if err != nil {
			t.Fatal(err)
		}
		offset := 28
		if p.Dim == 1 {
			offset = 32
		}
		if p.Dim == 2 {
			offset = 36
		}
		size := int(binary.LittleEndian.Uint64(raw[offset:]))
		ctRaw := raw[offset+8 : offset+8+size]
		tail := raw[offset+8+size:]
		wantTail := 0
		if p.Dim == 0 {
			wantTail = 16
		}
		if len(tail) != wantTail {
			t.Fatalf("dim %d: %d trailing bytes, want %d", p.Dim, len(tail), wantTail)
		}
		if p.Dim == 0 {
			if _, err = c.Decrypt("output", raw[:len(raw)-16]); err != nil {
				t.Fatal("native scalar EOF form rejected", err)
			}
		}
		for _, bad := range [][]byte{append(append([]byte{}, raw...), 0), raw[:len(raw)-1]} {
			if _, err = c.Decrypt("output", bad); err == nil {
				t.Fatal("invalid tail accepted")
			}
		}
		// No allocation may use any attacker-controlled polynomial/vector length.
		var md rlwe.MetaData
		polys := 1 + md.BinarySize()
		levels := polys + 8
		coeffs := levels + 8
		for _, at := range []int{polys, levels, coeffs} {
			corrupted := append([]byte{}, ctRaw...)
			binary.LittleEndian.PutUint64(corrupted[at:], ^uint64(0))
			if err = validateCiphertext(corrupted, c.Keys.Params); err == nil {
				t.Fatal("unbounded vector accepted")
			}
		}
	}
}
