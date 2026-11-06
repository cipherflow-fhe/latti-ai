package core

import "errors"

type feature struct {
	dim, channels, perCT, level int
	shape, skip                 []int
	fill, block                 int
	ciphertexts                 ciphertexts
}

func (c *Client) packingWalk(p FeatureParam, valueCount int, selected int, visit func(int, int, int)) (feature, int, error) {
	f := feature{dim: p.Dim, channels: p.Channel, level: p.Level, shape: p.Shape, skip: []int{1, 1}, fill: 1}
	nslots := c.Keys.Params.MaxSlots()
	if p.DataType == "feature_mat" {
		if valueCount != product(p.Shape) {
			return f, 0, errors.New("matrix input count mismatch")
		}
		return c.matrixWalk(p, false, f, selected, visit)
	}
	if valueCount == 0 || (p.Dim > 0 && valueCount != p.Channel*product(p.Shape)) {
		return f, 0, errors.New("input count does not match task")
	}
	switch p.Dim {
	case 0:
		if p.PackNum < 1 || p.PackNum > nslots {
			return f, 0, errors.New("invalid pack_num")
		}
		f.channels, f.skip[0] = valueCount, nslots/p.PackNum
		f.perCT = nslots / f.skip[0]
	case 1:
		f.skip = p.skip()
		if f.skip[0] < 1 {
			return f, 0, errors.New("invalid skip")
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
			return f, 0, errors.New("invalid interleaved block shape")
		}
		stride := []int{p.Shape[0] / c.Task.BlockShape[0], p.Shape[1] / c.Task.BlockShape[1]}
		f.perCT = 1
		count := p.Channel * product(stride)
		first, last := 0, count
		if selected >= 0 {
			first, last = selected, selected+1
		}
		for i := first; visit != nil && i < last; i++ {
			channel, grid := i/product(stride), i%product(stride)
			for y := 0; y < p.Shape[0]; y++ {
				for x := 0; x < p.Shape[1]; x++ {
					if y%stride[0] == grid/stride[1] && x%stride[1] == grid%stride[1] {
						visit(i, y/stride[0]*c.Task.BlockShape[1]+x/stride[1], channel*product(p.Shape)+y*p.Shape[1]+x)
					}
				}
			}
		}
		return f, count, nil
	}
	if f.perCT < 1 {
		return f, 0, errors.New("feature does not fit slots")
	}
	count := ceil(f.channels, f.perCT)
	first, last := 0, count
	if selected >= 0 {
		first, last = selected, selected+1
	}
	for i := first; visit != nil && i < last; i++ {
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
					return f, 0, errors.New("packing exceeds slots")
				}
				visit(i, slot, channel*length+k)
			}
		}
	}
	return f, count, nil
}

func (c *Client) unpackingWalk(p FeatureParam, f feature, count int, selected int, visit func(int, int, int)) (int, error) {
	if p.DataType == "feature_mat" {
		layout, expected, err := c.matrixWalk(p, true, f, selected, visit)
		if err != nil {
			return 0, err
		}
		if expected != count {
			return 0, errors.New("matrix ciphertext count mismatch")
		}
		return product(layout.shape), nil
	}
	if f.channels < 1 || f.perCT < 1 || product(f.skip) == 0 {
		return 0, errors.New("invalid feature layout")
	}
	length := 1
	if p.Dim > 0 {
		length = product(f.shape)
	}
	if length < 1 || f.channels > (1<<26)/length {
		return 0, errors.New("invalid output size")
	}
	valueCount := f.channels * length
	if p.Dim == 2 && c.Task.PackStyle == "multiplexed" && len(c.Task.BlockShape) == 2 && product(p.Shape) > product(c.Task.BlockShape) {
		block := c.Task.BlockShape
		if product(block) == 0 || f.shape[0]%block[0] != 0 || f.shape[1]%block[1] != 0 {
			return 0, errors.New("invalid output block shape")
		}
		stride := []int{f.shape[0] / block[0], f.shape[1] / block[1]}
		if count != f.channels*product(stride) {
			return 0, errors.New("interleaved ciphertext count mismatch")
		}
		first, last := 0, count
		if selected >= 0 {
			first, last = selected, selected+1
		}
		for i := first; visit != nil && i < last; i++ {
			for y := 0; y < block[0]; y++ {
				for x := 0; x < block[1]; x++ {
					if y*block[1]+x >= c.Keys.Params.MaxSlots() {
						return 0, errors.New("output exceeds slots")
					}
					grid := i % product(stride)
					visit(i, y*block[1]+x, (i/product(stride))*length+(y*stride[0]+grid/stride[1])*f.shape[1]+x*stride[1]+grid%stride[1])
				}
			}
		}
		return valueCount, nil
	}
	fill := []int{1, 1}
	copy(fill, p.InvalidFill)
	if product(fill) == 0 {
		return 0, errors.New("invalid output fill")
	}
	if count != ceil(f.channels, f.perCT) {
		return 0, errors.New("ciphertext count mismatch")
	}
	first, last := 0, f.channels
	if selected >= 0 {
		first, last = selected*f.perCT, min((selected+1)*f.perCT, f.channels)
	}
	for channel := first; visit != nil && channel < last; channel++ {
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
						return 0, errors.New("invalid output skip")
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
						return 0, errors.New("invalid output skip")
					}
					local := j % perBlock
					slot = j/perBlock*length*product(f.skip) + (y*f.skip[0]+local/cols)*f.shape[1]*f.skip[1] + x*f.skip[1] + local%cols
				}
			}
			if slot < 0 || slot >= c.Keys.Params.MaxSlots() {
				return 0, errors.New("output exceeds slots")
			}
			visit(channel/f.perCT, slot, channel*length+k)
		}
	}
	return valueCount, nil
}
