package core

func (c *Client) pack(p FeatureParam, values []float64) (feature, [][]float64, error) {
	f, count, err := c.packingWalk(p, len(values), -1, nil)
	if err != nil {
		return f, nil, err
	}
	slots := make([][]float64, count)
	for i := range slots {
		slots[i] = make([]float64, c.Keys.Params.MaxSlots())
	}
	_, _, err = c.packingWalk(p, len(values), -1, func(ct, slot, index int) { slots[ct][slot] = values[index] })
	return f, slots, err
}
func (c *Client) unpack(p FeatureParam, f feature, slots [][]float64) ([]float64, error) {
	count, err := c.unpackingWalk(p, f, len(slots), -1, nil)
	if err != nil {
		return nil, err
	}
	values := make([]float64, count)
	_, err = c.unpackingWalk(p, f, len(slots), -1, func(ct, slot, index int) { values[index] = slots[ct][slot] })
	return values, err
}
