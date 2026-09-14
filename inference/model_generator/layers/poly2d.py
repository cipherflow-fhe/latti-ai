from inference.model_generator.layers.poly_base import PolyBase
from inference.lattisense.frontend.custom_task import *

op_class = 'Poly2D'


class Poly2D(PolyBase):
    """Stockmeyer polynomial DAG for multiplexed 2D feature packing."""

    def __init__(self, input_shape, order, skip, n_channel_per_ct, upsample_factor=[1, 1], block_expansion=[1, 1]):
        PolyBase.__init__(self, order)
        self.input_shape = input_shape
        self.order = order
        self.skip = skip
        self.n_channel_per_ct = n_channel_per_ct
        self.upsample_factor = upsample_factor
        self.block_expansion = block_expansion
        self.pre_skip = [skip[0] * upsample_factor[0], skip[1] * upsample_factor[1]]
        self.block_shape = [
            int(input_shape[0] * skip[0] / block_expansion[0]),
            int(input_shape[1] * skip[1] / block_expansion[1]),
        ]

    def call_stockmeyer_feature2d(self, x, weight_pt):
        return self._run_stockmeyer_core(x, lambda i, j: weight_pt[i][j])

    def call(self, x, weight_pt):
        return self.call_stockmeyer_feature2d(x, weight_pt)

    def call_custom_compute(self, x, poly_data_source, layer_id=''):
        return self.call_stockmeyer_lazy(x, poly_data_source, layer_id)

    def call_stockmeyer_lazy(self, x, poly_data_source, layer_id=''):
        cache = {}

        def get_weight(i, j):
            key = (i, j)
            if key not in cache:
                node = CkksPlaintextRingtNode(f'encode_pt_{i}_{j}')
                custom_compute(
                    inputs=[poly_data_source],
                    output=node,
                    type='encode_pt',
                    attributes={'op_class': op_class, 'type': 'weight_pt', 'i': i, 'j': j},
                )
                cache[key] = node
            return cache[key]

        return self._run_stockmeyer_core(x, get_weight)

    def _run_stockmeyer_core(self, x, get_weight):
        if not x:
            return []
        self.init_stockmeyer(x[0].level)
        powers = self.stockmeyer_powers
        baby = self.stockmeyer_baby_steps
        groups = self.stockmeyer_n_baby_polys
        result = []
        for xi in x:
            xp = {1: xi}
            for power, info in sorted(powers.items()):
                if power > 1:
                    xp[power] = rescale(relin(mult(xp[info.decomp_a], xp[info.decomp_a])))
            nodes = []
            for g in range(groups):
                base = g * baby
                target = self.stockmeyer_baby_poly_output_level[g]
                scale = self.stockmeyer_baby_poly_output_scale[g]
                acc = None
                if base + 1 <= self.order:
                    p = xp[1]
                    if p.level > target + 1:
                        p = drop_level(p, p.level - target - 1)
                    acc = rescale(mult(p, get_weight(base + 1, len(result))))
                if base + 2 <= self.order:
                    p = xp[2]
                    if p.level > target + 1:
                        p = drop_level(p, p.level - target - 1)
                    term = rescale(mult(p, get_weight(base + 2, len(result))))
                    acc = term if acc is None else add(acc, term)
                if base + 3 <= self.order:
                    p = xp[1]
                    if p.level > target + 2:
                        p = drop_level(p, p.level - target - 2)
                    c3_scale = target / self.stockmeyer_powers[2].scale * float(self._param().q[target + 1])
                    c3x = rescale(mult(p, get_weight(base + 3, len(result))))
                    x2 = xp[2]
                    if x2.level > target + 1:
                        x2 = drop_level(x2, x2.level - target - 1)
                    term = rescale(relin(mult(c3x, x2)))
                    acc = term if acc is None else add(acc, term)
                if acc is None:
                    nodes.append((None, base, target, scale))
                    continue
                nodes.append((add(acc, get_weight(base, len(result))), None, target, scale))
            combine = baby
            while len(nodes) > 1:
                nxt = []
                for i in range(0, len(nodes), 2):
                    if i + 1 == len(nodes):
                        nxt.append(nodes[i])
                        continue
                    left, right = nodes[i], nodes[i + 1]
                    giant = xp[combine]
                    if giant.level > left[2] + 1:
                        giant = drop_level(giant, giant.level - left[2] - 1)
                    if right[0] is None:
                        term = rescale(mult(giant, get_weight(right[1], len(result))))
                    else:
                        rc = right[0]
                        if rc.level > left[2] + 1:
                            rc = drop_level(rc, rc.level - left[2] - 1)
                        term = rescale(relin(mult(rc, giant)))
                    nxt.append((add(left[0], term), None, left[2], left[3]))
                nodes = nxt
                combine *= 2
            result.append(nodes[0][0])
        return result
