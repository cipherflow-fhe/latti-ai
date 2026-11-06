package core

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/cipherflow-fhe/lattigo/ring"
	"github.com/cipherflow-fhe/lattigo/schemes/ckks"
)

func testParameters(t *testing.T, id string, ringType int) []byte {
	t.Helper()
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 12, LogQ: []int{55, 40, 40}, LogP: []int{55}, LogDefaultScale: 40, RingType: ring.Type(ringType)})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(map[string]any{id: map[string]any{"poly_modulus_degree": p.N(), "q": p.Q(), "p": p.P(), "log_default_scale": 40, "ring_type": ringType, "n_slots": p.MaxSlots()}})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestPregenerationBindsTaskWithoutReplacingKeys(t *testing.T) {
	for _, ringType := range []int{0, 1} {
		parameters := testParameters(t, "p", ringType)
		keys, err := NewKeyClient(parameters, false)
		if err != nil {
			t.Fatal(err)
		}
		if err = keys.GenerateKeys(); err != nil {
			t.Fatal(err)
		}
		sk, pk := keys.Keys.SK, keys.Keys.PK
		full, err := fullBytes(t, keys)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = keys.Encrypt("input", []float64{1}); err == nil {
			t.Fatal("task required")
		}
		for _, feature := range []string{
			`{"dim":0,"channel":3,"pack_num":8,"skip":512,"level":1,"ckks_parameter_id":"p"}`,
			`{"dim":2,"channel":1,"shape":[2,2],"level":1,"ckks_parameter_id":"p"}`,
		} {
			// Match the scalar layout to the real slot capacity of each ring.
			if ringType == 0 {
				feature = strings.ReplaceAll(feature, `"skip":512`, `"skip":256`)
			}
			task := configFor(t, feature, "multiplexed", "")
			if err = keys.BindTask(task, parameters); err != nil {
				t.Fatal(err)
			}
			if keys.Keys.SK != sk || keys.Keys.PK != pk {
				t.Fatal("bindTask replaced keys")
			}
			values := []float64{1.25, -2, 0.03}
			if keys.Task.Inputs["input"].Dim == 2 {
				values = append(values, 0.5)
			}
			raw, err := keys.Encrypt("input", values)
			if err != nil {
				t.Fatal(err)
			}
			fresh, err := NewKeyClient(parameters, false)
			if err != nil {
				t.Fatal(err)
			}
			if err = fresh.Restore(full); err != nil {
				t.Fatal(err)
			}
			if err = fresh.BindTask(task, parameters); err != nil {
				t.Fatal(err)
			}
			got, err := fresh.Decrypt("output", raw)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(values) {
				t.Fatal("output length")
			}
			for i, want := range values {
				if math.Abs(got[i]-want) > 0.001 {
					t.Fatalf("value %d: %g != %g", i, got[i], want)
				}
			}
		}
	}
}

func TestPregenerationRejectsIncompleteParameters(t *testing.T) {
	for _, config := range []string{`{}`, `{"p":{"poly_modulus_degree":4096}}`, `{"p":{"param_name":"N16QP1546H192H32"}}`, `{"a":{},"b":{}}`} {
		if _, err := NewKeyClient([]byte(config), false); err == nil {
			t.Fatal("incomplete profile accepted")
		}
	}
}

func TestTaskRejectsMismatchedReferences(t *testing.T) {
	feature := `{"dim":0,"channel":1,"pack_num":8,"skip":512,"level":1,"ckks_parameter_id":"p"}`
	task := configFor(t, feature, "ordinary", "")
	var config map[string]any
	json.Unmarshal(task, &config)
	config["task_output_param"].(map[string]any)["output"].(map[string]any)["ckks_parameter_id"] = "missing"
	raw, _ := json.Marshal(config)
	if _, err := NewClient(raw, testParameters(t, "p", 1)); err == nil {
		t.Fatal("unknown output profile accepted")
	}
	keys, err := NewKeyClient(testParameters(t, "p", 1), false)
	if err != nil {
		t.Fatal(err)
	}
	if err = keys.BindTask(task, testParameters(t, "p", 0)); err == nil {
		t.Fatal("task ring mismatch accepted")
	}
	config = map[string]any{}
	json.Unmarshal(task, &config)
	config["use_btp"] = true
	raw, _ = json.Marshal(config)
	if err = keys.BindTask(raw, testParameters(t, "p", 1)); err == nil {
		t.Fatal("missing bootstrapping plan accepted")
	}
}
