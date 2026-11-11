package core

import (
	"encoding/json"
	"fmt"
	"math/bits"

	"github.com/cipherflow-fhe/lattigo/core/rlwe"
	"github.com/cipherflow-fhe/lattigo/ring"
	"github.com/cipherflow-fhe/lattigo/schemes/ckks"
)

// taskParameters matches the CKKS entries emitted by the model compiler.
// In particular, Q and P are decoded directly as uint64, never via float64.
type taskParameters struct {
	SecretHammingWeight int      `json:"secret_hamming_weight"`
	Degree              int      `json:"poly_modulus_degree"`
	Q                   []uint64 `json:"q"`
	P                   []uint64 `json:"p"`
	LogScale            *int     `json:"log_default_scale"`
	RingType            *int     `json:"ring_type"`
	Slots               *int     `json:"n_slots"`
	LogSlots            *int     `json:"log_slots"`
}

func parseTaskParameters(data []byte, id string) (ckks.Parameters, error) {
	var entries map[string]taskParameters
	if err := json.Unmarshal(data, &entries); err != nil {
		return ckks.Parameters{}, fmt.Errorf("invalid CKKS parameter JSON: %w", err)
	}
	entry, ok := entries[id]
	if !ok {
		return ckks.Parameters{}, fmt.Errorf("missing CKKS parameter %q", id)
	}
	if entry.Degree < 2 || entry.Degree&(entry.Degree-1) != 0 || len(entry.Q) == 0 || len(entry.P) == 0 || entry.LogScale == nil || *entry.LogScale < 0 || *entry.LogScale > 128 {
		return ckks.Parameters{}, fmt.Errorf("invalid CKKS parameters for %q", id)
	}
	ringType := ring.ConjugateInvariant
	if entry.RingType != nil {
		switch *entry.RingType {
		case 0:
			ringType = ring.Standard
		case 1:
			ringType = ring.ConjugateInvariant
		default:
			return ckks.Parameters{}, fmt.Errorf("invalid ring type for %q", id)
		}
	}
	logN := bits.Len(uint(entry.Degree)) - 1
	if logN > rlwe.MaxLogN || entry.SecretHammingWeight < 0 || entry.SecretHammingWeight > entry.Degree {
		return ckks.Parameters{}, fmt.Errorf("invalid CKKS degree or secret Hamming weight for %q", id)
	}
	literal := ckks.ParametersLiteral{
		LogN: logN, Q: entry.Q, P: entry.P,
		RingType: ringType, LogDefaultScale: *entry.LogScale,
	}
	if entry.SecretHammingWeight != 0 {
		literal.Xs = ring.Ternary{H: entry.SecretHammingWeight}
	}
	params, err := ckks.NewParametersFromLiteral(literal)
	if err != nil {
		return ckks.Parameters{}, fmt.Errorf("invalid CKKS parameters for %q: %w", id, err)
	}
	if (entry.Slots != nil && *entry.Slots != params.MaxSlots()) || (entry.LogSlots != nil && *entry.LogSlots != params.LogMaxSlots()) {
		return ckks.Parameters{}, fmt.Errorf("inconsistent CKKS slot capacity for %q", id)
	}
	return params, nil
}

func requireMatchingTaskParameters(data []byte, ids []string, expected ckks.Parameters) error {
	for _, id := range ids {
		actual, err := parseTaskParameters(data, id)
		if err != nil {
			return err
		}
		if !actual.Equal(&expected) {
			return fmt.Errorf("CKKS parameter %q does not match the client profile", id)
		}
	}
	return nil
}
