// Command profile derives the bundled CI PN15 profile from the current native
// parameter table without converting uint64 values through JavaScript numbers.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/cipherflow-fhe/latti-ai/client-wasm/core"
)

func main() {
	if len(os.Args) != 3 {
		panic("usage: profile parameter.json output.json")
	}
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	var table map[string]map[string]map[string]json.RawMessage
	if err = json.Unmarshal(raw, &table); err != nil {
		panic(err)
	}
	entry := table["CKKS"]["15"]
	if entry == nil {
		panic("missing native PN15 parameters")
	}
	entry["poly_modulus_degree"] = json.RawMessage("32768")
	entry["ring_type"] = json.RawMessage("1")
	entry["n_slots"] = json.RawMessage("32768")
	parameters, err := json.Marshal(map[string]any{"param0": entry})
	if err != nil {
		panic(err)
	}
	client, err := core.NewKeyClient(parameters, true)
	if err != nil {
		panic(err)
	}
	profile, err := client.Profile()
	if err != nil {
		panic(err)
	}
	sha := sha256.Sum256(raw)
	revision, err := exec.Command("git", "-C", "../../lattisense", "rev-parse", "HEAD").Output()
	if err != nil {
		panic(err)
	}
	output, err := json.MarshalIndent(map[string]any{"name": "PN15Default-CI", "parameters": string(parameters), "bootstrap": true, "profile": profile, "provenance": map[string]any{"source": "inference/lattisense/frontend/parameter.json#CKKS.15", "sha256": hex.EncodeToString(sha[:]), "revision": strings.TrimSpace(string(revision))}}, "", "  ")
	if err != nil {
		panic(err)
	}
	if err = os.WriteFile(os.Args[2], append(output, '\n'), 0644); err != nil {
		panic(err)
	}
	fmt.Println(profile["profileId"])
}
