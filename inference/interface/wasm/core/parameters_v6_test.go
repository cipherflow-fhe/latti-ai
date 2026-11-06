package core

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/cipherflow-fhe/lattigo/ring"
)

func TestTaskParametersV6(t *testing.T) {
	generator := ring.NewNTTFriendlyPrimesGenerator(55, 1<<14)
	primes, err := generator.NextAlternatingPrimes(3)
	if err != nil {
		t.Fatal(err)
	}
	entry := fmt.Sprintf(`{"poly_modulus_degree":4096,"q":[%d,%d],"p":[%d],"log_default_scale":32}`, primes[0], primes[1], primes[2])
	data := []byte(fmt.Sprintf(`{"ci":%s,"same":%s,"standard":%s}`, entry, entry, strings.Replace(entry, `"log_default_scale":32`, `"ring_type":0,"log_default_scale":32`, 1)))
	params, err := parseTaskParameters(data, "ci")
	if err != nil {
		t.Fatal(err)
	}
	if params.RingType() != ring.ConjugateInvariant || params.MaxSlots() != 4096 || params.Q()[0] != primes[0] || primes[0] <= 1<<53 {
		t.Fatal("CI parameters or uint64 modulus changed")
	}
	if err := requireMatchingTaskParameters(data, []string{"ci", "same"}, params); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"missing", "standard"} {
		if err := requireMatchingTaskParameters(data, []string{id}, params); err == nil {
			t.Fatalf("accepted incompatible parameter %q", id)
		}
	}
	standard, err := parseTaskParameters(data, "standard")
	if err != nil || standard.RingType() != ring.Standard || standard.MaxSlots() != 2048 {
		t.Fatalf("standard parameters: %v", err)
	}
}

func TestTaskParametersRejectInvalidInput(t *testing.T) {
	const valid = `{"poly_modulus_degree":4096,"q":[274876334081,4293918721],"p":[549754109953],"log_default_scale":32}`
	var fields map[string]any
	if err := json.Unmarshal([]byte(valid), &fields); err != nil {
		t.Fatal(err)
	}
	for name, value := range map[string]any{"poly_modulus_degree": 4097, "q": []int{}, "ring_type": 2, "n_slots": 2048, "log_slots": 11, "log_default_scale": -1} {
		t.Run(name, func(t *testing.T) {
			copyFields := make(map[string]any, len(fields))
			for k, v := range fields {
				copyFields[k] = v
			}
			copyFields[name] = value
			data, err := json.Marshal(map[string]any{"param0": copyFields})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := parseTaskParameters(data, "param0"); err == nil {
				t.Fatal("invalid parameters accepted")
			}
		})
	}
}
