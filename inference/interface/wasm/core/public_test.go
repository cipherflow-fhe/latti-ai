package core

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
)

func TestPublicImportRevokesSecretCapabilitiesAndEncryptsParameterModeInput(t *testing.T) {
	parameters := testParameters(t, "p", 1)
	model, err := NewKeyClient(parameters, false)
	if err != nil {
		t.Fatal(err)
	}
	if err = model.GenerateKeys(); err != nil {
		t.Fatal(err)
	}
	export, err := model.StartExportPublic()
	if err != nil {
		t.Fatal(err)
	}
	public := runFeatureStream(t, export, nil)
	provider, err := NewKeyClient(parameters, false)
	if err != nil {
		t.Fatal(err)
	}
	if err = provider.GenerateKeys(); err != nil {
		t.Fatal(err)
	}
	runFeatureStream(t, provider.StartImportPublic(uint64(len(public))), public)
	a, _ := model.KeyIdentity()
	b, _ := provider.KeyIdentity()
	if a != b || provider.Keys.SK != nil || provider.decryptor != nil {
		t.Fatal("public import retained secret capability")
	}
	skip, _ := json.Marshal(provider.Keys.Params.MaxSlots() / 2)
	p := FeatureParam{Dim: 0, Channel: 2, PackNum: 2, Level: 1, Skip: skip}
	provider.Task = Task{DeploymentMode: "server_provisioned_runner", InputMode: "client_ciphertext", Inputs: map[string]FeatureParam{"input": p}, Outputs: map[string]FeatureParam{"output": p}}
	model.Task = provider.Task
	model.Task.PackStyle = "ordinary"
	provider.Task.PackStyle = "ordinary"
	values := make([]byte, 16)
	binary.LittleEndian.PutUint64(values, math.Float64bits(0.25))
	binary.LittleEndian.PutUint64(values[8:], math.Float64bits(-0.5))
	stream, err := provider.StartEncrypt("input", 16)
	if err != nil {
		t.Fatal(err)
	}
	ciphertext := runFeatureStream(t, stream, values)
	got, err := model.Decrypt("output", ciphertext)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || math.Abs(got[0]-0.25) > 1e-5 || math.Abs(got[1]+0.5) > 1e-5 {
		t.Fatal(got)
	}
	if _, err = provider.StartDecrypt("output", uint64(len(ciphertext))); err == nil {
		t.Fatal("public provider decrypted")
	}
	if _, err = provider.StartProvision([]byte("[]")); err == nil {
		t.Fatal("public provider provisioned")
	}
	if err = provider.GenerateKeys(); err != nil {
		t.Fatal(err)
	}
	invalid := provider.StartImportPublic(1)
	event := streamEvent(t, invalid, nil)
	if event.Type != "error" || provider.decryptor != nil || provider.Keys.SK != nil || provider.encryptor != nil {
		t.Fatal("failed public import retained previous key capabilities")
	}

}
