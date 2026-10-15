package core

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

const fixedParameters = `{"param0":{"poly_modulus_degree":65536,"n_mult_level":9,"log_default_scale":40,"param_name":"N16QP1546H192H32"}}`

func TestPregenerationRestoresAcrossTaskLayouts(t *testing.T) {
	keys, err := NewClient(nil, []byte(fixedParameters))
	if err != nil {
		t.Fatal(err)
	}
	stream, err := keys.BeginSetup()
	if err != nil {
		t.Fatal(err)
	}
	header, err := stream.Next()
	if err != nil {
		t.Fatal(err)
	}
	parameterLength := len(mustParams(t, keys.Keys.Params))
	if header[4+parameterLength] != 0 || header[5+parameterLength] != 1 {
		t.Fatal("evaluation context must include PK and exclude SK")
	}
	full, err := keys.Full()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = keys.Encrypt("input", []float64{1}); err == nil || !strings.Contains(err.Error(), "task") {
		t.Fatalf("key-only encryption must require a task, got %v", err)
	}
	if _, err = keys.Decrypt("output", nil); err == nil || !strings.Contains(err.Error(), "task") {
		t.Fatalf("key-only decryption must require a task, got %v", err)
	}
	for _, feature := range []string{
		`{"dim":0,"channel":3,"pack_num":32,"skip":1024,"level":4,"ckks_parameter_id":"param0"}`,
		`{"dim":2,"channel":1,"shape":[2,2],"pack_num":32,"skip":[1,1],"level":4,"ckks_parameter_id":"param0"}`,
	} {
		task := []byte(`{"use_btp":true,"pack_style":"multiplexed","block_shape":[32,32],"task_input_param":{"input":` + feature + `},"task_output_param":{"output":` + feature + `}}`)
		client, err := NewClient(task, []byte(fixedParameters))
		if err != nil {
			t.Fatal(err)
		}
		if err = client.Restore(full); err != nil {
			t.Fatal(err)
		}
		values := []float64{1.25, -2, 0.03}
		if client.Task.Inputs["input"].Dim == 2 {
			values = append(values, 0.5)
		}
		raw, err := client.Encrypt("input", values)
		if err != nil {
			t.Fatal(err)
		}
		fresh, err := NewClient(task, []byte(fixedParameters))
		if err != nil {
			t.Fatal(err)
		}
		if err = fresh.Restore(full); err != nil {
			t.Fatal(err)
		}
		got, err := fresh.Decrypt("output", raw)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(values) {
			t.Fatalf("got %d values, want %d", len(got), len(values))
		}
		for i, want := range values {
			if math.Abs(got[i]-want) > 0.001 {
				t.Fatalf("value %d: %g, want %g", i, got[i], want)
			}
		}
	}
}

func TestPregenerationRejectsParameterOverrides(t *testing.T) {
	for field, value := range map[string]any{
		"param_name": "PN16QP1761", "poly_modulus_degree": 4096,
		"n_mult_level": 8, "log_default_scale": 39, "log_slots": 14,
		"q": []uint64{17}, "p": []uint64{17},
	} {
		t.Run(field, func(t *testing.T) {
			var config map[string]map[string]any
			json.Unmarshal([]byte(fixedParameters), &config)
			config["param0"][field] = value
			encoded, _ := json.Marshal(config)
			if _, err := NewClient(nil, encoded); err == nil {
				t.Fatal("incompatible fixed parameters accepted")
			}
		})
	}
	for _, config := range []string{`{}`, `{"param0":{"poly_modulus_degree":65536}}`,
		`{"param0":{"param_name":"N16QP1546H192H32"},"param1":{"param_name":"N16QP1546H192H32"}}`} {
		if _, err := NewClient(nil, []byte(config)); err == nil {
			t.Fatal("invalid parameter profile accepted")
		}
	}
}

func TestFixedProfileRejectsIncompatibleTask(t *testing.T) {
	feature := `{"dim":0,"channel":1,"pack_num":32,"skip":1024,"level":4,"ckks_parameter_id":"param0"}`
	task := []byte(`{"use_btp":true,"pack_style":"multiplexed","task_input_param":{"input":` + feature + `},"task_output_param":{"output":` + feature + `}}`)
	for _, change := range []func(map[string]any){
		func(v map[string]any) { v["use_btp"] = false },
		func(v map[string]any) {
			v["task_output_param"].(map[string]any)["output"].(map[string]any)["ckks_parameter_id"] = "missing"
		},
	} {
		var v map[string]any
		json.Unmarshal(task, &v)
		change(v)
		encoded, _ := json.Marshal(v)
		if _, err := NewClient(encoded, []byte(fixedParameters)); err == nil {
			t.Fatal("task incompatible with the fixed profile accepted")
		}
	}
}
