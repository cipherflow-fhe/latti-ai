package core

import "errors"

type feature struct {
	dim, channels, perCT, level int
	shape, skip                 []int
	fill, block                 int
	ciphertexts                 ciphertexts
}

func (c *Client) pack(p FeatureParam, values []float64) (feature, [][]float64, error) {
	f := feature{dim: p.Dim, channels: p.Channel, level: p.Level, shape: p.Shape, skip: []int{1, 1}, fill: 1}
	nslots := c.Keys.Params.N() / 2
	if p.DataType == "feature_mat" {
		return c.packMatrix(p, values)
	}
	if len(values) == 0 || (p.Dim > 0 && len(values) != p.Channel*product(p.Shape)) {
		return f, nil, errors.New("input count does not match task")
	}
	switch p.Dim {
	case 0:
		if p.PackNum < 1 || p.PackNum > nslots {
			return f, nil, errors.New("invalid pack_num")
		}
		f.channels, f.skip[0] = len(values), nslots/p.PackNum
		f.perCT = nslots / f.skip[0]
	case 1:
		f.skip = p.skip()
		if f.skip[0] < 1 {
			return f, nil, errors.New("invalid skip")
		}
		f.perCT = nslots / (p.Shape[0] * f.skip[0])
		if c.Task.PackStyle == "multiplexed" {
			f.perCT = nslots / p.Shape[0]
		}
	case 2:
		f.perCT = nslots / product(p.Shape)
	}
	if p.Dim == 2 && c.Task.PackStyle == "multiplexed" && product(p.Shape) > nslots {
		if len(c.Task.BlockShape) != 2 || product(c.Task.BlockShape) == 0 || product(c.Task.BlockShape) > nslots || p.Shape[0]%c.Task.BlockShape[0] != 0 || p.Shape[1]%c.Task.BlockShape[1] != 0 {
			return f, nil, errors.New("invalid interleaved block shape")
		}
		stride := []int{p.Shape[0] / c.Task.BlockShape[0], p.Shape[1] / c.Task.BlockShape[1]}
		f.perCT = 1
		slots := make([][]float64, p.Channel*product(stride))
		for i := range slots {
			slots[i] = make([]float64, nslots)
			channel, grid := i/product(stride), i%product(stride)
			for y := 0; y < p.Shape[0]; y++ {
				for x := 0; x < p.Shape[1]; x++ {
					if y%stride[0] == grid/stride[1] && x%stride[1] == grid%stride[1] {
						slots[i][y/stride[0]*c.Task.BlockShape[1]+x/stride[1]] = values[channel*product(p.Shape)+y*p.Shape[1]+x]
					}
				}
			}
		}
		return f, slots, nil
	}
	if f.perCT < 1 {
		return f, nil, errors.New("feature does not fit slots")
	}
	slots := make([][]float64, ceil(f.channels, f.perCT))
	for i := range slots {
		slots[i] = make([]float64, nslots)
		for j := 0; j < f.perCT; j++ {
			channel := i*f.perCT + j
			if channel >= f.channels && (p.Dim == 0 || c.Task.PackStyle == "multiplexed") {
				continue
			}
			channel %= f.channels
			length := 1
			if p.Dim > 0 {
				length = product(p.Shape)
			}
			for k := 0; k < length; k++ {
				slot := j * f.skip[0]
				if p.Dim == 1 {
					slot = j*length*f.skip[0] + k*f.skip[0]
					if c.Task.PackStyle == "multiplexed" {
						slot = (j/f.skip[0])*length*f.skip[0] + k*f.skip[0] + j%f.skip[0]
					}
				} else if p.Dim == 2 {
					slot = j*length + k
				}
				if slot >= nslots {
					return f, nil, errors.New("packing exceeds slots")
				}
				slots[i][slot] = values[channel*length+k]
			}
		}
	}
	return f, slots, nil
}

func (c *Client) unpack(p FeatureParam, f feature, slots [][]float64) ([]float64, error) {
	if p.DataType == "feature_mat" {
		return c.unpackMatrix(p, f, slots)
	}
	if f.channels < 1 || f.perCT < 1 || product(f.skip) == 0 {
		return nil, errors.New("invalid feature layout")
	}
	length := 1
	if p.Dim > 0 {
		length = product(f.shape)
	}
	if length < 1 || f.channels > (1<<26)/length {
		return nil, errors.New("invalid output size")
	}
	values := make([]float64, f.channels*length)
	if p.Dim == 2 && c.Task.PackStyle == "multiplexed" && len(c.Task.BlockShape) == 2 && product(p.Shape) > product(c.Task.BlockShape) {
		block := c.Task.BlockShape
		if product(block) == 0 || f.shape[0]%block[0] != 0 || f.shape[1]%block[1] != 0 {
			return nil, errors.New("invalid output block shape")
		}
		stride := []int{f.shape[0] / block[0], f.shape[1] / block[1]}
		if len(slots) != f.channels*product(stride) {
			return nil, errors.New("interleaved ciphertext count mismatch")
		}
		for i, vector := range slots {
			for y := 0; y < block[0]; y++ {
				for x := 0; x < block[1]; x++ {
					if y*block[1]+x >= len(vector) {
						return nil, errors.New("output exceeds slots")
					}
					grid := i % product(stride)
					values[(i/product(stride))*length+(y*stride[0]+grid/stride[1])*f.shape[1]+x*stride[1]+grid%stride[1]] = vector[y*block[1]+x]
				}
			}
		}
		return values, nil
	}
	fill := []int{1, 1}
	copy(fill, p.InvalidFill)
	if product(fill) == 0 {
		return nil, errors.New("invalid output fill")
	}
	if len(slots) != ceil(f.channels, f.perCT) {
		return nil, errors.New("ciphertext count mismatch")
	}
	for channel := 0; channel < f.channels; channel++ {
		for k := 0; k < length; k++ {
			j := channel % f.perCT
			slot := j * f.skip[0]
			if p.Dim == 0 {
				slot = j * p.skip()[0]
			}
			if p.Dim == 1 {
				slot = j*length*f.skip[0] + k*f.skip[0]
				if c.Task.PackStyle == "multiplexed" {
					perBlock := f.skip[0] / fill[0]
					if perBlock == 0 {
						return nil, errors.New("invalid output skip")
					}
					slot = j/perBlock*length*f.skip[0] + k*f.skip[0] + j%perBlock
				}
			} else if p.Dim == 2 {
				y, x := k/f.shape[1], k%f.shape[1]
				slot = j*length*product(f.skip) + y*f.shape[1]*product(f.skip) + x*f.skip[1]
				if c.Task.PackStyle == "multiplexed" {
					perBlock := product(f.skip) / product(fill)
					cols := f.skip[1] / fill[1]
					if perBlock == 0 || cols == 0 {
						return nil, errors.New("invalid output skip")
					}
					local := j % perBlock
					slot = j/perBlock*length*product(f.skip) + (y*f.skip[0]+local/cols)*f.shape[1]*f.skip[1] + x*f.skip[1] + local%cols
				}
			}
			vector := slots[channel/f.perCT]
			if slot < 0 || slot >= len(vector) {
				return nil, errors.New("output exceeds slots")
			}
			values[channel*length+k] = vector[slot]
		}
	}
	return values, nil
}
