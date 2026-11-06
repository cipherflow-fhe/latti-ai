package core

import (
	"math"
	"testing"

	"github.com/cipherflow-fhe/lattigo/schemes/ckks"
)

func configFor(t *testing.T, param string, style string, extra string) []byte {
	t.Helper()
	return []byte(`{"evaluation_keys":[],"use_btp":false,"pack_style":"` + style + `","block_shape":[32,32],` + extra + `"task_input_param":{"input":` + param + `},"task_output_param":{"output":` + param + `}}`)
}

func TestRestoreDecryptsCiphertextsFromPreviousSession(t *testing.T) {
	for _, tc := range []struct {
		name, param, style, extra string
		count                     int
	}{
		{"scalar", `{"dim":0,"channel":3,"pack_num":8,"skip":512,"level":1,"ckks_parameter_id":"p"}`, "ordinary", "", 3},
		{"one-dimensional", `{"dim":1,"channel":3,"shape":[4],"skip":[2],"level":1,"ckks_parameter_id":"p"}`, "ordinary", "", 12},
		{"one-dimensional-multiplexed", `{"dim":1,"channel":3,"shape":[4],"skip":[2],"level":1,"ckks_parameter_id":"p"}`, "multiplexed", "", 12},
		{"image", `{"dim":2,"channel":3,"shape":[4,4],"level":1,"ckks_parameter_id":"p"}`, "ordinary", "", 48},
		{"image-multiplexed", `{"dim":2,"channel":3,"shape":[4,4],"level":1,"ckks_parameter_id":"p"}`, "multiplexed", "", 48},
		{"image-interleaved", `{"dim":2,"channel":1,"shape":[128,64],"level":1,"ckks_parameter_id":"p"}`, "multiplexed", "", 8192},
		{"matrix", `{"dim":2,"data_type":"feature_mat","shape":[4,8],"head_shape":[4,4],"matmul_block_size":2,"n_heads":2,"level":1,"ckks_parameter_id":"p"}`, "ordinary", "", 32},
		{"matrix-diagonal", `{"dim":2,"data_type":"feature_mat","shape":[64,64],"head_shape":[32,64],"n_heads":2,"level":1,"ckks_parameter_id":"p"}`, "ordinary", `"mat_pack_style":"par_diagonal_pack",`, 4096},
	} {
		t.Run(tc.name, func(t *testing.T) {
			task := configFor(t, tc.param, tc.style, tc.extra)
			parameters := testParameters(t, "p", 1)
			client, err := NewClient(task, parameters)
			if err != nil {
				t.Fatal(err)
			}
			err = client.GenerateKeys()
			if err != nil {
				t.Fatal(err)
			}
			values := make([]float64, tc.count)
			for i := range values {
				values[i] = float64(i%19-9) / 32
			}
			ciphertext, err := client.Encrypt("input", values)
			if err != nil {
				t.Fatal(err)
			}
			full, err := fullBytes(t, client)
			if err != nil {
				t.Fatal(err)
			}
			fresh, err := NewClient(task, parameters)
			if err != nil {
				t.Fatal(err)
			}
			if err = fresh.Restore(full); err != nil {
				t.Fatal(err)
			}
			got, err := fresh.Decrypt("output", ciphertext)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(values) {
				t.Fatalf("length %d, want %d", len(got), len(values))
			}
			for i := range got {
				if math.Abs(got[i]-values[i]) > 0.001 {
					t.Fatalf("value %d = %g, want %g", i, got[i], values[i])
				}
			}
			checkFeatureStreams(t, fresh, values)
			if _, err = fresh.Encrypt("missing", values); err == nil {
				t.Fatal("unknown input accepted")
			}
			if _, err = fresh.Encrypt("input", []float64{math.NaN()}); err == nil {
				t.Fatal("invalid data accepted")
			}
			if _, err = fresh.Decrypt("output", ciphertext[:len(ciphertext)-1]); err == nil {
				t.Fatal("truncated ciphertext accepted")
			}
			other, err := NewClient(task, testParameters(t, "p", 0))
			if err != nil {
				t.Fatal(err)
			}
			if err = other.Restore(full); err == nil {
				t.Fatal("wrong task ring accepted")
			}

		})
	}
}

func TestScalarWireTreatsRedundantSkipAsMetadata(t *testing.T) {
	feature := `{"dim":0,"channel":2,"pack_num":2,"skip":2048,"level":0,"ckks_parameter_id":"param0"}`
	task := []byte(`{"evaluation_keys":[],"use_btp":false,"pack_style":"multiplexed","task_input_param":{"input":` + feature + `},"task_output_param":{"output":` + feature + `}}`)
	client, err := NewClient(task, testParameters(t, "param0", 1))
	if err != nil {
		t.Fatal(err)
	}
	if err = client.GenerateKeys(); err != nil {
		t.Fatal(err)
	}
	raw, err := client.Encrypt("input", []float64{1.25, -2})
	if err != nil {
		t.Fatal(err)
	}
	values, err := client.Decrypt("output", raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 2 || math.Abs(values[0]-1.25) > 0.001 || math.Abs(values[1]+2) > 0.001 {
		t.Fatalf("scalar round trip with repeated skip: %v", values)
	}
}

func mustParams(t *testing.T, params ckks.Parameters) []byte {
	t.Helper()
	raw, err := params.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
