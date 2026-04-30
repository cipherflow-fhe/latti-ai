# softmax.py
from inference.lattisense.frontend.custom_task import *
from inference.model_generator.deploy_cmds import *

op_class = "SoftmaxLayer"

class SoftmaxLayer:
    def __init__(self, num_channels: int, skip: int = 1, N: int = 65536):
        self.num_channels = num_channels
        self.skip = skip
        self.N = N

    def make_pt_nodes(self, layer_id: str):
        return [], [], None

    def call(self, x: list[CkksCiphertextNode], weight_pt, bias_pt, N: int, repack_mask_pt=None):
        exp_nodes = []
        for ct in x:
            x2 = rescale(relinearize(mult(ct, ct)))
            x3 = rescale(relinearize(mult(x2, ct)))
            x4 = rescale(relinearize(mult(x3, ct)))
            x5 = rescale(relinearize(mult(x4, ct)))
            one = CkksPlaintextNode([1.0])
            exp = add_plain(ct, one)
            half = CkksPlaintextNode([0.5])
            exp = add(exp, mult_plain(x2, half))
            sixth = CkksPlaintextNode([1.0/6.0])
            exp = add(exp, mult_plain(x3, sixth))
            twenty4 = CkksPlaintextNode([1.0/24.0])
            exp = add(exp, mult_plain(x4, twenty4))
            one20 = CkksPlaintextNode([1.0/120.0])
            exp = add(exp, mult_plain(x5, one20))
            exp_nodes.append(exp)

        total = exp_nodes[0]
        for e in exp_nodes[1:]:
            total = add(total, e)
        n_effective = (self.num_channels + self.skip - 1) // self.skip
        sum_node = total
        for i in range(1, n_effective):
            rot = rotate_cols(total, i * self.skip)[0]
            sum_node = add(sum_node, rot)

        inv_coeffs = [9.999, -9.999, 9.999, -9.999]
        c0 = CkksPlaintextNode([inv_coeffs[0]])
        inv = encrypt_asymmetric(c0)
        c1 = CkksPlaintextNode([inv_coeffs[1]])
        inv = add(inv, mult_plain(sum_node, c1))
        x2 = rescale(relinearize(mult(sum_node, sum_node)))
        c2 = CkksPlaintextNode([inv_coeffs[2]])
        inv = add(inv, mult_plain(x2, c2))
        x3 = rescale(relinearize(mult(x2, sum_node)))
        c3 = CkksPlaintextNode([inv_coeffs[3]])
        inv = add(inv, mult_plain(x3, c3))

        n_slots = N // 2
        inv_bcast = inv
        for i in range(1, n_slots):
            rot = rotate_cols(inv, i)[0]
            inv_bcast = add(inv_bcast, rot)

        result = []
        for e in exp_nodes:
            mul3 = mult(e, inv_bcast)
            mul = rescale(relinearize(mul3))
            result.append(mul)
        return result

    def call_custom_compute(self, x: list[CkksCiphertextNode], conv_data_source, N: int):
        return self.call(x, None, None, N)
