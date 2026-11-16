"""Build a real CKKS unrolled training DAG with existing LattiAI PS operators.

Column packing: each encrypted X column is zero-padded to SIMD capacity; each encrypted weight is replicated across slots. No plaintext
access occurs between forward, residual, gradient, and update stages.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'inference/lattisense'))
sys.path.insert(0, str(ROOT))
from frontend.custom_task import (
    CkksParam, CkksCiphertextNode, CkksPlaintextNode, Argument,
    Processor, set_fhe_param, mult, mult_relin, rescale, drop_level,
    add, sub, rotate_cols, process_custom_task, bootstrap,
)
from inference.model_generator.layers.poly_relu0d import PolyRelu0D


def align(x, level):
    return drop_level(x, x.metadata.level - level)


def product(x, y):
    level = min(x.metadata.level, y.metadata.level)
    return rescale(mult_relin(align(x, level), align(y, level)))


def build(output, features=30, degree=3, learning_rate=0.05, batch_size=426, steps=2,
          checkpoints=False):
    if features < 1 or degree not in (3, 5, 7, 9, 15):
        raise ValueError('features >= 1; degree must be 3, 5, 7, 9, or 15')
    param = CkksParam.create_default_param(log_n=15)
    set_fhe_param(param)
    level = param.max_level
    inputs = []
    def encrypted(name):
        node = CkksCiphertextNode(level=level, id=name)
        inputs.append(Argument(name, node))
        return node
    columns = [encrypted(f'x{j}') for j in range(features + 1)]
    weights = [encrypted(f'w{j}') for j in range(features + 1)]
    labels = encrypted('labels')

    if steps < 1:
        raise ValueError('steps must be positive')
    stage_nodes = []
    updates = []
    refreshes = []
    required_levels = PolyRelu0D.compute_bsgs_level_cost(degree) + 3
    for step in range(steps):
        # Fixed manual placement: refresh every two cubic updates, entirely in
        # the encrypted DAG. Higher degrees use a conservative per-step refresh.
        if weights[0].metadata.level < required_levels:
            # Boost small weights by 2 with addition before approximate
            # refresh, then undo the boost homomorphically. This reduces the
            # absolute bootstrap error at the weight scale, costing one level.
            weights = [add(w, w) for w in weights]
            weights = bootstrap([align(w, 0) for w in weights],
                                output_id=f'step{step}_refreshed_w')
            weights = [rescale(mult(w, 1 / 2)) for w in weights]
            refreshes.append({'before_step': step + 1, 'ciphertexts': len(weights),
                              'input_gain': 2, 'inverse_gain_rescale_levels': 1})
        # Stage 1: encrypted forward dot product, summed over feature ciphertexts.
        terms = [product(x, w) for x, w in zip(columns, weights)]
        logits = terms[0]
        for term in terms[1:]:
            logits = add(logits, term)
    
        # Stage 2: existing PS/BSGS implementation, public polynomial coefficients.
        poly = PolyRelu0D(degree, 1, 1)
        scales = poly.bsgs_coefficient_scales(logits.metadata.level, param)
        coefficients = []
        for k in range(degree + 1):
            node = CkksPlaintextNode(id=f'step{step}_coefficient{k}', scale=scales[k])
            inputs.append(Argument(f'step{step}_coefficient{k}', node))
            coefficients.append([node])
        probability = poly.call_bsgs_feature0d([logits], coefficients)[0]
    
        # Stage 3: encrypted residual. Labels must be encoded at probability scale
        # after dropping levels; manifest records all scales required by the runner.
        residual = sub(probability, align(labels, probability.metadata.level))
    
        # Stage 4: encrypted gradient; cyclic all-slot sum replicates the result.
        # Zero-padded feature slots contribute zero to the gradient.
        next_weights = []
        for j, (column, weight) in enumerate(zip(columns, weights)):
            gradient = product(column, residual)
            for shift in range(param.log_max_slots()):
                gradient = add(gradient, rotate_cols(gradient, 1 << shift)[0])
            scaled = rescale(mult(gradient, learning_rate / batch_size))
            updated = sub(align(weight, scaled.metadata.level), scaled)
            next_weights.append(updated)
        weights = next_weights
        stage_nodes.append({"step": step + 1, "logits": logits.id,
                            "probability": probability.id, "residual": residual.id,
                            "weights": [w.id for w in weights]})
        if checkpoints and step != steps - 1:
            updates.extend(Argument(f'step{step+1}_w{j}', w) for j, w in enumerate(weights))
    updates.extend(Argument(f'updated_w{j}', w) for j, w in enumerate(weights))
    manifest = {
        'backend': 'GPU', 'log_n': 15, 'features': features, 'degree': degree,
        'learning_rate': learning_rate, 'steps': steps, 'batch_size': batch_size,
        'bootstrapping_count': sum(r['ciphertexts'] for r in refreshes),
        'bootstrap_batch_count': len(refreshes), 'bootstrap_placements': refreshes,
        'checkpoints': checkpoints,
        'input_level': level, 'slot_count': 1 << param.log_max_slots(),
        'packing': 'zero-padded feature columns and labels; weights replicated',
        'inputs': {a.id: {'level': a.data[0].metadata.level, 'scale': a.data[0].metadata.scale}
                   for a in inputs},
        'status': 'graph generation only; runtime verification required',
    }
    # Export signatures before the frontend clears its global graph.
    manifest['outputs'] = [
        {'name': a.id, 'level': a.data[0].metadata.level, 'scale': a.data[0].metadata.scale}
        for a in updates
    ]
    manifest['level_consumption'] = level - weights[0].metadata.level
    manifest['level_consumption_note'] = 'input minus final level; excludes levels restored by bootstrap'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from training.model_compiler.primitive_graph import export_primitive_graph
    export_primitive_graph(output / "training_dag.json", stage_nodes)
    process_custom_task(input_args=inputs, output_args=updates,
                        output_instruction_path=str(output), processor=Processor.GPU)
    (output / 'experiment.json').write_text(json.dumps(manifest, indent=2))
    return manifest
