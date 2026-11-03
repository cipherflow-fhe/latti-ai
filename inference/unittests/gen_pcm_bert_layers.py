# Copyright (c) 2026 CipherFlow (Shenzhen) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Generate the PCM BERT layer regression tasks, exclusively under /tmp.

Run with the project Python, then pass the same directory to test_pcm_bert_layers.
By default arithmetic edges pass through a test-only callback that checks actual
CKKS metadata on CPU. Use --without-contract-checks for production-like GPU
graphs without extra ciphertext round trips. Production generation is unchanged.
"""

import argparse
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'inference' / 'lattisense'))

from inference.lattisense.frontend.custom_task import (
    Argument, CkksCiphertextNode, CustomDataNode, custom_compute,
    drop_level, process_custom_task,
)
from inference.model_generator.deploy_cmds import set_param
from inference.model_generator.layers.matrix_param_ops import _resolve
from inference.model_generator.parameter_sources import resolve_pcm_parameter_source

MODULES = [importlib.import_module('inference.model_generator.layers.' + name) for name in (
    'par_block_col_major_softmax', 'par_block_col_major_poly', 'par_block_col_major_poly_mult_ct',
    'par_block_col_major_layernorm',
)]


def instrument():
    for module in MODULES:
        for operation in ('add', 'sub', 'mult'):
            if not hasattr(module, operation):
                continue
            original = getattr(module, operation)

            def checked(a, b, *args, _op=operation, _original=original, **kwargs):
                rhs = _resolve(b, a)
                checked_a = CkksCiphertextNode(level=a.level)
                custom_compute([a] if a is rhs else [a, rhs], checked_a, 'check_contract',
                               {'operation': _op, 'plaintext': not isinstance(rhs, CkksCiphertextNode)})
                return _original(checked_a, b, *args, **kwargs)

            setattr(module, operation, checked)


def parameter_source(name, encrypted):
    source = CustomDataNode(type='pcm_layer_source', id=name)
    if encrypted:
        def build(attrs, parameter_name, level, trigger):
            secret = resolve_pcm_parameter_source({}, attrs)['visibility'] == 'secret'
            if secret:
                output = CkksCiphertextNode(parameter_name + '_ct', level=level)
            else:
                from inference.lattisense.frontend.custom_task import CkksPlaintextRingtNode
                output = CkksPlaintextRingtNode(parameter_name + '_pt')
            custom_compute([source, trigger], output, 'encode_secret' if secret else 'encode_pt',
                           dict(attrs, parameter_level=level))
            return output
        source.matrix_parameter_builder = build
    return source


def generate(root, check_contracts=True, encrypted=False):
    if check_contracts:
        instrument()
    softmax, polynomial, gelu, ln = MODULES
    for cfg, (rows, heads, cols, block, k) in enumerate(((35, 12, 37, 32, 1), (11, 3, 9, 8, 2))):
        for name in ('square', 'headsum', 'invinit', 'inviter', 'ctmul', 'gelumult',
                     'poly1', 'poly2', 'poly3', 'poly7', 'poly15', 'gelu_composition'):
            set_param('PN14QP438')
            shape, head = (rows, heads * cols * k), (rows, cols)
            common = dict(shape=shape, head_shape=head, n_heads=heads, n_slot=8192, block_size=block)
            if name.startswith('poly'):
                layer = polynomial.ParBlockColMajorPoly(**common, order=int(name[4:]))
            elif name == 'gelu_composition':
                layer = polynomial.ParBlockColMajorPoly(**common, order=1)
            else:
                cls = {'square': softmax.ParBlockColMajorMultipleSquare,
                       'headsum': softmax.ParBlockColMajorHeadColSum,
                       'invinit': softmax.ParBlockColMajorInverseInit,
                       'inviter': softmax.ParBlockColMajorInverseIter,
                       'ctmul': softmax.ParBlockColMajorCtMul,
                       'gelumult': gelu.ParBlockColMajorPolyMultCt}[name]
                layer = cls(**common)
            a = [CkksCiphertextNode(level=6) for _ in range(layer.total_cts)]
            b = [CkksCiphertextNode(level=6) for _ in a]
            source = parameter_source('layer', encrypted)
            inputs = [Argument('a', a), Argument('layer', [source])]
            if name == 'gelu_composition':
                sources = [source] + [parameter_source('layer' + str(i), encrypted)
                                     for i in range(1, 4)]
                output = a
                for i in range(3):
                    output = layer.call_custom_compute(output, sources[i])
                tail = gelu.ParBlockColMajorPolyMultCt(**common)
                output = tail.call_custom_compute(output, [drop_level(ct, 3) for ct in a], sources[3])
                inputs += [Argument('layer' + str(i), [sources[i]]) for i in range(1, 4)]
            elif name in ('inviter', 'ctmul', 'gelumult'):
                inputs.append(Argument('b', b))
                output = layer.call_custom_compute(a, b, source)
            else:
                output = layer.call_custom_compute(a, source)
            process_custom_task(inputs, [Argument('output', output)],
                                output_instruction_path=root / str(cfg) / name, fpga_acc=False)

        # LayerNorm's statistics use one broadcast ciphertext per token block.
        shape = (rows, heads * 2 * block)
        for name, cls in (('lnstats', ln.ParBlockColMajorLNStats),
                          ('lncenter', ln.ParBlockColMajorLNXCentered),
                          ('lninit', ln.ParBlockColMajorLNMinimaxInit),
                          ('lngs', ln.ParBlockColMajorLNGoldschmidt),
                          ('lnaffine', ln.ParBlockColMajorLNAffine)):
            set_param('PN14QP438')
            compact = (rows + block - 1) // block
            if name in ('lninit', 'lngs'):
                layer = cls(block, 8192)
                count = compact
            else:
                layer = cls(shape, block, heads, 8192)
                count = layer.total_cts
            a = [CkksCiphertextNode(level=6) for _ in range(count)]
            b = [CkksCiphertextNode(level=6) for _ in range(compact)]
            source = parameter_source('layer', encrypted)
            inputs = [Argument('a', a), Argument('layer', [source])]
            if name in ('lngs', 'lnaffine'):
                inputs.append(Argument('b', b))
                output = layer.call_custom_compute(a, b, source)
            else:
                output = layer.call_custom_compute(a, source)
            process_custom_task(inputs, [Argument('output', output)],
                                output_instruction_path=root / str(cfg) / name, fpga_acc=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--without-contract-checks', action='store_true')
    parser.add_argument('--encrypted-parameters', action='store_true')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to('/tmp'):
        parser.error('output must be under /tmp')
    generate(args.output, not args.without_contract_checks, args.encrypted_parameters)
