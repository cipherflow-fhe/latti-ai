package core

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strconv"

	"github.com/cipherflow-fhe/lattigo/clientcontext"
)

// Profile reports exact moduli as decimal strings. Identity includes the
// native parameters and the shared bootstrapping configuration/key plan.
func (c *Client) Profile() (map[string]any, error) {
	parameter, err := c.Keys.Params.MarshalBinary()
	if err != nil {
		return nil, err
	}
	plan, err := json.Marshal(c.plan)
	if err != nil {
		return nil, err
	}
	hash := sha256.New()
	hash.Write(parameter)
	hash.Write(plan)
	capabilities, err := clientcontext.DescribeKeyPlan(c.Keys.Params, c.plan)
	if err != nil {
		return nil, err
	}
	encodedCapabilities, err := json.Marshal(capabilities)
	if err != nil {
		return nil, err
	}
	hash.Write(encodedCapabilities)
	result := map[string]any{"ringType": int(c.Keys.Params.RingType()), "logN": c.Keys.Params.LogN(), "maxSlots": c.Keys.Params.MaxSlots(), "logDefaultScale": c.Keys.Params.LogDefaultScale(), "keyPlan": map[string]any{"defaultRotations": c.plan.DefaultRotations, "bootstrapping": c.plan.Bootstrapping}}
	stringify := func(values []uint64) []string {
		out := make([]string, len(values))
		for i, v := range values {
			out[i] = strconv.FormatUint(v, 10)
		}
		return out
	}
	result["q"], result["p"] = stringify(c.Keys.Params.Q()), stringify(c.Keys.Params.P())
	result["evaluationKeys"] = capabilities
	if c.plan.Bootstrapping {
		btp, err := clientcontext.BootstrappingParameters(c.Keys.Params)
		if err != nil {
			return nil, err
		}
		data, err := btp.MarshalBinary()
		if err != nil {
			return nil, err
		}
		hash.Write(data)
		// Preserve all expanded circuit parameters without JSON number rounding.
		result["bootstrappingParameters"] = hex.EncodeToString(data)
	}
	result["nativeParameters"] = hex.EncodeToString(parameter)
	result["profileId"] = hex.EncodeToString(hash.Sum(nil))
	return result, nil
}
