package core

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestNativeFeatureInterop(t *testing.T) {
	runner := os.Getenv("LATTI_NATIVE_WIRE_FIXTURE")
	if runner == "" {
		t.Skip("set LATTI_NATIVE_WIRE_FIXTURE to the current CMake wasm_wire_fixture executable")
	}
	for _, ringType := range []int{0, 1} {
		for _, tc := range []struct {
			name, param, style, extra string
			count                     int
		}{
			{"scalar", `{"dim":0,"channel":3,"pack_num":8,"skip":SLOT_SKIP,"level":1,"ckks_parameter_id":"p"}`, "ordinary", "", 3},
			{"1d", `{"dim":1,"channel":3,"shape":[4],"skip":[2],"level":1,"ckks_parameter_id":"p"}`, "multiplexed", "", 12},
			{"2d", `{"dim":2,"channel":3,"shape":[4,4],"level":1,"ckks_parameter_id":"p"}`, "multiplexed", "", 48},
			{"interleaved", `{"dim":2,"channel":1,"shape":[128,64],"level":1,"ckks_parameter_id":"p"}`, "multiplexed", "", 8192},
			{"matrix-block", `{"dim":2,"data_type":"feature_mat","shape":[4,8],"head_shape":[4,4],"matmul_block_size":2,"n_heads":2,"level":1,"ckks_parameter_id":"p"}`, "ordinary", "", 32},
			{"matrix-diagonal", `{"dim":2,"data_type":"feature_mat","shape":[64,64],"head_shape":[32,64],"n_heads":2,"level":1,"ckks_parameter_id":"p"}`, "ordinary", `"mat_pack_style":"par_diagonal_pack",`, 4096},
			{"matrix-block-first-token", `{"dim":2,"data_type":"feature_mat","shape":[4,8],"head_shape":[4,4],"matmul_block_size":2,"n_heads":2,"select_first_token":true,"is_transposed":false,"level":1,"ckks_parameter_id":"p"}`, "ordinary", "", 32},
			{"matrix-diagonal-first-token", `{"dim":2,"data_type":"feature_mat","shape":[64,64],"head_shape":[32,64],"n_heads":2,"select_first_token":true,"is_transposed":true,"level":1,"ckks_parameter_id":"p"}`, "ordinary", `"mat_pack_style":"par_diagonal_pack",`, 4096},
		} {
			t.Run(strconv.Itoa(ringType)+"/"+tc.name, func(t *testing.T) {
				root := t.TempDir()
				parameters := testParameters(t, "p", ringType)
				skip := 512
				if ringType == 0 {
					skip = 256
				}
				task := configFor(t, strings.ReplaceAll(tc.param, "SLOT_SKIP", strconv.Itoa(skip)), tc.style, tc.extra)
				client, err := NewClient(task, parameters)
				if err != nil {
					t.Fatal(err)
				}
				if err = client.GenerateKeys(); err != nil {
					t.Fatal(err)
				}
				values := make([]float64, tc.count)
				text := make([]string, tc.count)
				numeric := make([]byte, tc.count*8)
				for i := range values {
					values[i] = float64(i%19-9) / 32
					text[i] = strconv.FormatFloat(values[i], 'g', -1, 64)
					binary.LittleEndian.PutUint64(numeric[i*8:], math.Float64bits(values[i]))
				}
				stream, err := client.StartEncrypt("input", uint64(len(numeric)))
				if err != nil {
					t.Fatal(err)
				}
				ciphertext := runFeatureStream(t, stream, numeric)
				full, err := fullBytes(t, client)
				if err != nil {
					t.Fatal(err)
				}
				expected := values
				if strings.Contains(tc.name, "first-token") {
					channels := client.Task.Outputs["output"].Shape[0]
					if !*client.Task.Outputs["output"].Transposed {
						channels = client.Task.Outputs["output"].Shape[1]
					}
					expected = values[:channels]
				}
				encoded, _ := json.Marshal(expected)
				for name, data := range map[string][]byte{"task_config.json": task, "ckks_parameter.json": parameters, "output-values.json": encoded, "values.csv": []byte(strings.Join(text, ",") + "\n"), "go.ct": ciphertext, "full.bin": full} {
					if err = os.WriteFile(filepath.Join(root, name), data, 0600); err != nil {
						t.Fatal(err)
					}
				}
				if out, err := exec.Command(runner, root).CombinedOutput(); err != nil {
					t.Fatalf("native v6 interop: %v\n%s", err, out)
				}
				native, err := os.ReadFile(filepath.Join(root, "native.ct"))
				if err != nil {
					t.Fatal(err)
				}
				stream, err = client.StartDecrypt("output", uint64(len(native)))
				if err != nil {
					t.Fatal(err)
				}
				decoded := runFeatureStream(t, stream, native)
				if len(decoded) != len(expected)*8 {
					t.Fatal("native ciphertext output size differs")
				}
				for i, want := range expected {
					value := math.Float64frombits(binary.LittleEndian.Uint64(decoded[i*8:]))
					if math.Abs(value-want) > 0.001 {
						t.Fatalf("native-to-Go value %d: %g != %g", i, value, want)
					}
				}
			})
		}
	}
}
