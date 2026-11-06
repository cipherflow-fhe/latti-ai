package core

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBindChecksGraphKeyRequirements(t *testing.T) {
	params := testParameters(t, "p", 1)
	client, err := NewKeyClient(params, false)
	if err != nil {
		t.Fatal(err)
	}
	feature := `{"dim":0,"channel":1,"level":1,"ckks_parameter_id":"p"}`
	var task map[string]any
	if err = json.Unmarshal(configFor(t, feature, "ordinary", ""), &task); err != nil {
		t.Fatal(err)
	}
	required := KeyRequirement{Role: "residual", Kind: "glk", GaloisElement: client.Keys.Params.GaloisElement(1), LevelQ: 1, LevelP: 0}
	bind := func(requirement KeyRequirement) error {
		task["evaluation_keys"] = []KeyRequirement{requirement}
		raw, e := json.Marshal(task)
		if e != nil {
			t.Fatal(e)
		}
		return client.BindTask(raw, params)
	}
	if err = bind(required); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*KeyRequirement){
		func(r *KeyRequirement) { r.LevelQ = client.Keys.Params.MaxLevelQ() + 1 },
		func(r *KeyRequirement) { r.LevelP = client.Keys.Params.MaxLevelP() + 1 },
		func(r *KeyRequirement) { r.BaseTwoDecomposition = 4 },
		func(r *KeyRequirement) { r.Compressed = true },
		func(r *KeyRequirement) { r.GaloisElement = 3 },
		func(r *KeyRequirement) { r.Role = "bootstrap" },
	} {
		bad := required
		change(&bad)
		if err = bind(bad); err == nil {
			t.Fatalf("unsupported key accepted: %+v", bad)
		}
	}
	client.plan.DefaultRotations = false
	if err = bind(required); err == nil {
		t.Fatal("missing Galois key accepted")
	}
	client.plan.DefaultRotations = true
	delete(task, "evaluation_keys")
	raw, _ := json.Marshal(task)
	if err = client.BindTask(raw, params); err == nil || !strings.Contains(err.Error(), "evaluation_keys") {
		t.Fatalf("missing graph requirements: %v", err)
	}
}

// Exercise graph-generated requirements with production parameters without
// generating keys. A fixture label or profile hash cannot replace this check.
func TestGeneratedTaskKeyRequirements(t *testing.T) {
	directory := os.Getenv("LATTI_GENERATED_TASK_CLIENT")
	if directory == "" {
		t.Skip("set LATTI_GENERATED_TASK_CLIENT to a freshly generated client directory")
	}
	read := func(name string) []byte {
		data, err := os.ReadFile(filepath.Join(directory, name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	task, params := read("task_config.json"), read("ckks_parameter.json")
	client, err := NewClient(task, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(client.Task.EvaluationKeys) == 0 {
		t.Fatal("production graph did not export any key requirements")
	}
	if err = client.BindTask(task, params); err != nil {
		t.Fatal(err)
	}
}
