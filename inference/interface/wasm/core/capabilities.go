package core

import (
	"fmt"

	"github.com/cipherflow-fhe/lattigo/clientcontext"
)

// KeyRequirement is exported from the generated computation graph, rather than
// inferred from its input/output packing or task-local parameter identifier.
type KeyRequirement struct {
	Role                 string `json:"role"`
	Kind                 string `json:"kind"`
	GaloisElement        uint64 `json:"galoisElement,string,omitempty"`
	LevelQ               int    `json:"levelQ"`
	LevelP               int    `json:"levelP"`
	BaseTwoDecomposition int    `json:"baseTwoDecomposition"`
	Compressed           bool   `json:"compressed"`
}

func (c *Client) requireTaskKeys(required []KeyRequirement) error {
	available, err := clientcontext.DescribeKeyPlan(c.Keys.Params, c.plan)
	if err != nil {
		return err
	}
	for _, requirement := range required {
		found := false
		if requirement.LevelQ < 0 || requirement.LevelP < -1 {
			return fmt.Errorf("invalid evaluation-key level")
		}
		for _, key := range available {
			if key.Role == requirement.Role && key.Kind == requirement.Kind && key.GaloisElement == requirement.GaloisElement &&
				key.LevelQ >= requirement.LevelQ && (requirement.LevelP == -1 || key.LevelP == requirement.LevelP) &&
				key.BaseTwoDecomposition == requirement.BaseTwoDecomposition && key.Compressed == requirement.Compressed {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("client profile does not cover evaluation key: %s/%s/%d at Q=%d P=%d", requirement.Role, requirement.Kind, requirement.GaloisElement, requirement.LevelQ, requirement.LevelP)
		}
	}
	return nil
}
