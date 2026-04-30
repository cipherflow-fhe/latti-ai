
 
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
 
from inference.lattisense.frontend.custom_task import *
 
import math
 
op_class = 'SoftmaxLayerBase'
 
 
class SoftmaxLayerBase:
    """
    Parameters
    ----------
    n_channel : int
        Total number of output channels (logits).
    n_channel_per_ct : int
        Number of logical channels packed into each ciphertext.
        Must be a power of two (required by the rotate-and-add halving
        used in sum_slots / broadcast_slots).
    skip : int
        Physical slot stride between consecutive logical channels
        inside a ciphertext (Feature0DEncrypted::skip).
    exp_order : int
        Chebyshev polynomial degree used for exp approximation.
    inv_order : int
        Chebyshev polynomial degree used for reciprocal approximation.
    n_goldschmidt_iter : int
        Number of Goldschmidt refinement iterations (default 2,
        matching the C++ implementation).
    """
 
    def __init__(
        self,
        n_channel: int,
        n_channel_per_ct: int,
        skip: int,
        exp_order: int,
        inv_order: int,
        input_min: float,
        input_max: float,
        n_goldschmidt_iter: int = 2,
    ):
        if n_channel_per_ct & (n_channel_per_ct - 1) != 0:
            raise ValueError(
                f'n_channel_per_ct must be a power of two, got {n_channel_per_ct}'
            )
 
        self.n_channel = n_channel
        self.n_channel_per_ct = n_channel_per_ct
        self.skip = skip
        self.exp_order = exp_order
        self.inv_order = inv_order
        self.exp_domain_a = input_min
        self.exp_domain_b = input_max
        self.inv_domain_a = n_channel * math.exp(input_min) + 0.1
        self.inv_domain_b = n_channel * math.exp(input_max) + 0.5
        self.n_goldschmidt_iter = n_goldschmidt_iter
        self.n_ct = math.ceil(n_channel / n_channel_per_ct)
 
    # ──────────────────────────────────────────────────────────────────────────
    # Slot-level helpers (native graph ops: rotate_cols + add)
    # ──────────────────────────────────────────────────────────────────────────
 
    @staticmethod
    def _sum_slots(
        ct: CkksCiphertextNode,
        n_channel_per_ct: int,
        skip: int,
    ) -> CkksCiphertextNode:
        """
        Reduce ``n_channel_per_ct`` logical channels into physical slot 0
        via successive forward rotations and additions.
 
        Mirrors C++ ``SoftmaxLayerbase::sum_slots``:
          for step in 1, 2, 4, …, n_channel_per_ct/2:
              res += rotate(res, +step*skip)
        """
        res = ct
        step = 1
        while step < n_channel_per_ct:
            rotated = rotate_cols(res, [step * skip])[0]
            res = add(res, rotated)
            step <<= 1
        return res
 
    @staticmethod
    def _broadcast_slots(
        ct: CkksCiphertextNode,
        n_channel_per_ct: int,
        skip: int,
    ) -> CkksCiphertextNode:
        """
        Broadcast physical slot 0 to all ``n_channel_per_ct`` logical
        channels via successive backward rotations and additions.
 
        Mirrors C++ ``SoftmaxLayerbase::broadcast_slots``:
          for step in 1, 2, 4, …, n_channel_per_ct/2:
              res += rotate(res, -step*skip)
        """
        res = ct
        step = 1
        while step < n_channel_per_ct:
            rotated = rotate_cols(res, [-(step * skip)])[0]
            res = add(res, rotated)
            step <<= 1
        return res
 

    # ──────────────────────────────────────────────────────────────────────────
    # Plaintext node factory (non-lazy)
    # ──────────────────────────────────────────────────────────────────────────

    def make_pt_nodes(self, layer_id: str | int):
        """
        Create the two slot-mask plaintext nodes for this layer.

        slot0_mask_pt  : used in Step 3 to zero out all slots except slot 0
                        after the cross-CT global sum.
        slot0_mask_pt2 : used in Step 6 to re-mask the Goldschmidt result
                        before broadcasting.
        """
        slot0_mask_pt  = CkksPlaintextRingtNode(f'slot0_mask_{layer_id}')
        slot0_mask_pt2 = CkksPlaintextRingtNode(f'slot0_mask2_{layer_id}')
        return slot0_mask_pt, slot0_mask_pt2
    # ──────────────────────────────────────────────────────────────────────────
    # Main graph-building entry point
    # ──────────────────────────────────────────────────────────────────────────
 
    def call(
        self,
        x: list[CkksCiphertextNode],
        slot0_mask_pt: CkksPlaintextRingtNode,       # 第一次掩码（Step 3）
        slot0_mask_pt2: CkksPlaintextRingtNode,      # 第二次掩码（Step 6）
    ) -> list[CkksCiphertextNode]:
        """
        Build the Softmax computation graph and return the output ciphertext
        nodes, one per input ciphertext (preserving the same packing layout).
 
        Parameters
        ----------
        x : list[CkksCiphertextNode]
            Encrypted input ciphertexts (len == n_ct).
        softmax_data_source :
            Layer data-source node that provides pre-encoded plaintext
            constants (Chebyshev coefficients, domain bounds, mask vectors,
            scalar constants). Passed to every custom_compute call so the
            backend executor can resolve the layer's private state.
 
        Returns
        -------
        list[CkksCiphertextNode]
            Softmax output ciphertexts in the same packing layout as ``x``.
        """
 
        # ── Step 1: exp(x_i) for each input ciphertext ────────────────────────
        exp_cts: list[CkksCiphertextNode] = []
        for i, ct in enumerate(x):
            exp_ct = poly_eval(x=ct, func='exp', degree=self.exp_order, 
                      left=self.exp_domain_a, right=self.exp_domain_b, output_id=f'exp_{i}')
            exp_cts.append(exp_ct)
 
        # ── Step 2: Cross-CT global sum ────────────────────────────────────────
        global_sum: CkksCiphertextNode = None
        for ct in exp_cts:
            reduced = self._sum_slots(ct, self.n_channel_per_ct, self.skip)
            if global_sum is None:
                global_sum = reduced
            else:
                global_sum = add(global_sum, reduced)
 
        # ── Step 3: Mask to slot 0 ─────────────────────────────────────────────
        global_sum = rescale(mult(global_sum, slot0_mask_pt))
 
        # ── Step 4: Initial 1/sum estimate via Chebyshev poly eval ────────────
        # Domain: [inv_domain_a, inv_domain_b] computed from (n_channel, input range).
        inv_sum_init = poly_eval(x=global_sum, func='reciprocal', degree=self.inv_order, 
                  left=self.inv_domain_a, right=self.inv_domain_b, output_id='inv_init')
        
        # ── Step 5: Goldschmidt iterations ────────────────────────────────────
        # w_0 = global_sum * inv_sum_init  (ct×ct mult + relin + rescale)
        if global_sum.level > inv_sum_init.level:
            global_sum_for_gs = drop_level(global_sum, global_sum.level - inv_sum_init.level)
        else:
            global_sum_for_gs = global_sum
        inv_sum = goldschmidt_reciprocal(x=global_sum_for_gs, init_guess=inv_sum_init,
                                         iterations=self.n_goldschmidt_iter, output_id='inv_final')
        
        # ── Step 6: Mask refined inverse, then broadcast ───────────────────────
        inv_sum = rescale(mult(inv_sum, slot0_mask_pt2))

        # broadcast_slots: slot 0 → all n_channel_per_ct channels
        broadcast_inv = self._broadcast_slots(inv_sum, self.n_channel_per_ct, self.skip)
 
        # ── Step 7: Normalize — exp_ct_i * broadcast_inv ──────────────────────
        result: list[CkksCiphertextNode] = []
        for exp_ct in exp_cts:
            # 对齐 level，对应 C++ 的 align_levels
            if exp_ct.level > broadcast_inv.level:
                exp_ct = drop_level(exp_ct, exp_ct.level - broadcast_inv.level)
            prod = rescale(relin(mult(exp_ct, broadcast_inv)))
            result.append(prod)
        return result
 