package core

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/cipherflow-fhe/lattigo/ckks"
)

func configFor(t *testing.T, param string, style string, extra string) []byte {
	t.Helper()
	return []byte(`{"use_btp":false,"pack_style":"` + style + `","block_shape":[32,32],` + extra + `"task_input_param":{"input":` + param + `},"task_output_param":{"output":` + param + `}}`)
}

func TestRestoreDecryptsCiphertextsFromPreviousSession(t *testing.T) {
	for _, tc := range []struct {
		name, param, style, extra string
		count                     int
	}{
		{"scalar", `{"dim":0,"channel":3,"pack_num":8,"skip":256,"level":1,"ckks_parameter_id":"p"}`, "ordinary", "", 3},
		{"one-dimensional", `{"dim":1,"channel":3,"shape":[4],"skip":[2],"level":1,"ckks_parameter_id":"p"}`, "ordinary", "", 12},
		{"one-dimensional-multiplexed", `{"dim":1,"channel":3,"shape":[4],"skip":[2],"level":1,"ckks_parameter_id":"p"}`, "multiplexed", "", 12},
		{"image", `{"dim":2,"channel":3,"shape":[4,4],"level":1,"ckks_parameter_id":"p"}`, "ordinary", "", 48},
		{"image-multiplexed", `{"dim":2,"channel":3,"shape":[4,4],"level":1,"ckks_parameter_id":"p"}`, "multiplexed", "", 48},
		{"image-interleaved", `{"dim":2,"channel":1,"shape":[64,64],"level":1,"ckks_parameter_id":"p"}`, "multiplexed", "", 4096},
		{"matrix", `{"dim":2,"data_type":"feature_mat","shape":[4,8],"head_shape":[4,4],"matmul_block_size":2,"n_heads":2,"level":1,"ckks_parameter_id":"p"}`, "ordinary", "", 32},
		{"matrix-diagonal", `{"dim":2,"data_type":"feature_mat","shape":[64,64],"head_shape":[32,64],"n_heads":2,"level":1,"ckks_parameter_id":"p"}`, "ordinary", `"mat_pack_style":"par_diagonal_pack",`, 4096},
	} {
		t.Run(tc.name, func(t *testing.T) {
			task := configFor(t, tc.param, tc.style, tc.extra)
			parameters := []byte(`{"p":{"poly_modulus_degree":4096}}`)
			client, err := NewClient(task, parameters)
			if err != nil {
				t.Fatal(err)
			}
			stream, err := client.BeginSetup()
			if err != nil {
				t.Fatal(err)
			}
			// Consume a real header; persistence needs just this round's SK/PK.
			if _, err = stream.Next(); err != nil {
				t.Fatal(err)
			}
			stream = nil
			values := make([]float64, tc.count)
			for i := range values {
				values[i] = float64(i%19-9) / 32
			}
			ciphertext, err := client.Encrypt("input", values)
			if err != nil {
				t.Fatal(err)
			}
			full, err := client.Full()
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
			if _, err = fresh.Encrypt("missing", values); err == nil {
				t.Fatal("unknown input accepted")
			}
			if _, err = fresh.Encrypt("input", []float64{math.NaN()}); err == nil {
				t.Fatal("invalid data accepted")
			}
			if _, err = fresh.Decrypt("output", ciphertext[:len(ciphertext)-1]); err == nil {
				t.Fatal("truncated ciphertext accepted")
			}
			var wrong map[string]any
			json.Unmarshal(parameters, &wrong)
			wrong["p"].(map[string]any)["log_slots"] = 10
			encoded, _ := json.Marshal(wrong)
			other, err := NewClient(task, encoded)
			if err != nil {
				t.Fatal(err)
			}
			if err = other.Restore(full); err == nil {
				t.Fatal("wrong task parameters accepted")
			}
		})
	}
}

func TestBTPSetupUsesPublicEvaluationFormat(t *testing.T) {
	task := []byte(`{"use_btp":true,"pack_style":"multiplexed","task_input_param":{"input":{"dim":0,"channel":1,"pack_num":32,"skip":1024,"level":0,"ckks_parameter_id":"p"}},"task_output_param":{"output":{"dim":0,"channel":1,"pack_num":32,"skip":1024,"level":0,"ckks_parameter_id":"p"}}}`)
	parameters := []byte(`{"p":{"poly_modulus_degree":65536}}`)
	client, err := NewClient(task, parameters)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := client.BeginSetup()
	if err != nil {
		t.Fatal(err)
	}
	payload, err := stream.Next()
	if err != nil {
		t.Fatal(err)
	}
	if len(payload) == 0 || payload[4+len(mustParams(t, client.Keys.Params))] != 0 {
		t.Fatal("BTP evaluation payload did not force has_sk=0")
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
