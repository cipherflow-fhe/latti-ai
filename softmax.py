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
        output = CkksCiphertextNode(f"softmax_out")
        custom_compute(
            inputs=x,
            outputs=[output],
            type="CustomOp",
            attributes={
                "op_class": op_class,
                "num_channels": str(self.num_channels),
                "skip": str(self.skip),
                "N": str(self.N)
            }
        )
        return [output]

    def call_custom_compute(self, x: list[CkksCiphertextNode], conv_data_source, N: int):
        return self.call(x, None, None, N)
