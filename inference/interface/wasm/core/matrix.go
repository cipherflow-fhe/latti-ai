package core

import "errors"

// Matrix indexing mirrors FeatureMatEncrypted's parallel block/diagonal pack.
// Visit is used in both directions; diagonal CSVs follow native's transpose.
func (c *Client) matrixWalk(p FeatureParam, output bool, f feature, visit func(int, int, int)) (feature, int, error) {
	nslots := c.Keys.Params.N() / 2
	f.dim, f.shape, f.level = 2, p.Shape, p.Level
	if p.Heads < 2 || len(p.Shape) != 2 {
		return f, 0, errors.New("matrix requires parallel heads")
	}
	height, width := p.Shape[0], p.Shape[1]
	if c.Task.MatrixStyle == "par_diagonal_pack" {
		if len(p.HeadShape) != 2 || product(p.HeadShape) == 0 {
			return f, 0, errors.New("missing matrix head_shape")
		}
		transposed := p.Transposed == nil || *p.Transposed
		mPre, nPre := p.HeadShape[0], p.HeadShape[1]
		if !transposed {
			mPre, nPre = nPre, mPre
		}
		m, n, H := pow2(mPre), pow2(nPre), pow2(p.Heads)
		if n < m || nslots%(H*n) != 0 {
			return f, 0, errors.New("invalid diagonal matrix layout")
		}
		chunk := nslots / (H * n)
		if chunk == 0 || m%chunk != 0 {
			return f, 0, errors.New("diagonal matrix does not fit slots")
		}
		extent := height
		if !transposed {
			extent = width
		}
		perMB, mbs := m/chunk, ceil(extent, p.Heads*mPre)
		f.block = m
		if visit != nil {
			for mb := 0; mb < mbs; mb++ {
				for local := 0; local < perMB; local++ {
					for diagonal := 0; diagonal < chunk; diagonal++ {
						for t := 0; t < n; t++ {
							for h := 0; h < p.Heads; h++ {
								row, col := (local*chunk+diagonal+t)%m, t
								if transposed {
									row += mb*p.Heads*mPre + h*mPre
								} else {
									row, col = (local*chunk+diagonal+t)%n, mb*p.Heads*mPre+h*mPre+t%m
								}
								if row < height && col < width && (transposed && row%(p.Heads*mPre)%mPre < mPre || !transposed && col%(p.Heads*mPre)%mPre < mPre) {
									// Native reads width x height CSV then transposes before pack,
									// and transposes the decoded matrix back before flattening.
									index := col*height + row
									visit(mb*perMB+local, diagonal*H*n+t*H+h, index)
								}
							}
						}
					}
				}
			}
		}
		return f, mbs * perMB, nil
	}
	if width%p.Heads != 0 {
		return f, 0, errors.New("matrix width is not divisible by heads")
	}
	headDim := width / p.Heads
	if len(p.HeadShape) == 2 && p.HeadShape[1] > 0 {
		headDim = p.HeadShape[1]
	}
	d := p.Block
	if d == 0 {
		d = headDim
	}
	if output && f.block != 0 {
		d = f.block
	}
	m := height
	if output {
		if len(p.HeadShape) == 2 && product(p.HeadShape) > 0 {
			if height != p.HeadShape[0]*p.Heads || width != p.HeadShape[1] {
				m = p.HeadShape[0]
			}
		} else if height%p.Heads == 0 && height > d {
			m = height / p.Heads
		}
	}
	if d < 1 || d*d > nslots {
		return f, 0, errors.New("invalid matrix block size")
	}
	H := pow2(p.Heads)
	S, chunk, groups := H, H*d*d, 1
	if nslots < chunk {
		S = nslots / (d * d)
		chunk = nslots
		if S == 1 {
			H = p.Heads
		}
		groups = H / S
	}
	if S < 1 || H%S != 0 || nslots%chunk != 0 {
		return f, 0, errors.New("invalid matrix head packing")
	}
	rows, cols := ceil(m, d), ceil(headDim, d)
	perMB := rows * cols * groups
	mbs := ceil(width, headDim*p.Heads)
	f.block = d
	if visit != nil {
		for mb := 0; mb < mbs; mb++ {
			for bj := 0; bj < cols; bj++ {
				for bi := 0; bi < rows; bi++ {
					for group := 0; group < groups; group++ {
						for hLocal := 0; hLocal < S; hLocal++ {
							for col := 0; col < d; col++ {
								for row := 0; row < d; row++ {
									h := group*S + hLocal
									r, x := bi*d+row, bj*d+col
									globalCol := mb*p.Heads*headDim + h*headDim + x
									if h < p.Heads && r < m && x < headDim && globalCol < width {
										for repeat := 0; repeat < nslots/chunk; repeat++ {
											if output && repeat != 0 {
												break
											}
											visit(mb*perMB+(bi+rows*bj)*groups+group, repeat*chunk+(row+d*col)*S+hLocal, r*width+globalCol)
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	f.shape = []int{m, width}
	return f, mbs * perMB, nil
}

func (c *Client) packMatrix(p FeatureParam, values []float64) (feature, [][]float64, error) {
	f, count, err := c.matrixWalk(p, false, feature{}, nil)
	if err != nil {
		return f, nil, err
	}
	if len(values) != product(p.Shape) {
		return f, nil, errors.New("matrix input count mismatch")
	}
	slots := make([][]float64, count)
	for i := range slots {
		slots[i] = make([]float64, c.Keys.Params.N()/2)
	}
	_, _, err = c.matrixWalk(p, false, f, func(ct, slot, index int) { slots[ct][slot] = values[index] })
	return f, slots, err
}

func (c *Client) unpackMatrix(p FeatureParam, f feature, slots [][]float64) ([]float64, error) {
	f, count, err := c.matrixWalk(p, true, f, nil)
	if err != nil {
		return nil, err
	}
	if len(slots) != count {
		return nil, errors.New("matrix ciphertext count mismatch")
	}
	values := make([]float64, product(f.shape))
	var boundsError error
	_, _, err = c.matrixWalk(p, true, f, func(ct, slot, index int) {
		if slot >= len(slots[ct]) || index >= len(values) {
			boundsError = errors.New("matrix output exceeds layout")
			return
		}
		values[index] = slots[ct][slot]
	})
	if boundsError != nil {
		return nil, boundsError
	}
	return values, err
}
