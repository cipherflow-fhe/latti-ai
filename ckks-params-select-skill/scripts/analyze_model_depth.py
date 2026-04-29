#!/usr/bin/env python3
"""Analyze ONNX model or compiled JSON DAG to extract CKKS-relevant information.

Extracts: multiplicative depth, layer types, input shapes, channel counts,
and whether bootstrapping is needed for given parameter sets.

Usage:
  python analyze_model_depth.py --onnx model.onnx
  python analyze_model_depth.py --dag task/server/nn_layers_ct_0.json
  python analyze_model_depth.py --onnx model.onnx --max-levels 9
"""

import argparse
import json
import math
import sys


# Level-consuming operations (each costs 1 CKKS level)
LEVEL_CONSUMING_OPS = {
    'Conv', 'ConvTranspose', 'MatMul', 'Gemm',  # linear layers
    'Mul',  # square / element-wise multiply
    'PolyRelu', 'Square',  # polynomial activations
}

# Default polynomial degree for PolyRelu (overridable via --poly-degree)
DEFAULT_POLY_DEGREE = 4

# Pass-through operations (no level consumption)
PASS_THROUGH_OPS = {
    'Add', 'Reshape', 'Flatten', 'Transpose', 'Concat', 'Pad',
    'AveragePool', 'GlobalAveragePool', 'MaxPool',
    'BatchNormalization', 'Identity', 'Dropout',
    'Relu', 'Clip',  # replaced by poly, counted above
}


def analyze_onnx(onnx_path, poly_degree=DEFAULT_POLY_DEGREE):
    """Analyze ONNX model for multiplicative depth and layer info."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        print('Error: onnx package required. Install with: pip install onnx')
        sys.exit(1)

    model = onnx.load(onnx_path)
    graph = model.graph

    print(f'ONNX Model: {onnx_path}')
    print(f'  Opset: {model.opset_import[0].version}')
    print(f'  Nodes: {len(graph.node)}')
    print()

    # Collect layer types
    op_counts = {}
    for node in graph.node:
        op = node.op_type
        op_counts[op] = op_counts.get(op, 0) + 1
        # Build a graph for topological depth analysis
    input_names = set()
    output_to_node = {}
    for inp in graph.input:
        input_names.add(inp.name)

    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node.name

    # Track depth per intermediate value
    value_depth = {}
    for name in input_names:
        value_depth[name] = 0

    total_depth = 0
    for node in graph.node:
        # Input depth = max depth of all inputs
        max_input_depth = 0
        for inp in node.input:
            if inp in value_depth:
                max_input_depth = max(max_input_depth, value_depth[inp])

        # Does this op consume a level?
        consumes = 0
        if node.op_type == 'PolyRelu':
            consumes = math.ceil(math.log2(poly_degree))
        elif node.op_type == 'Square':
            consumes = 1
        elif node.op_type == 'Mul':
            inputs = list(node.input)
            if len(inputs) >= 2 and inputs[0] == inputs[1]:
                consumes = 1  # square
            else:
                consumes = 0  # element-wise scalar multiply (no level)
        elif node.op_type in LEVEL_CONSUMING_OPS:
            consumes = 1

        node_depth = max_input_depth + consumes

        for out in node.output:
            value_depth[out] = node_depth

        total_depth = max(total_depth, node_depth)

    # Print results
    print('--- Layer Types ---')
    for op, count in sorted(op_counts.items()):
        marker = ' *' if op in LEVEL_CONSUMING_OPS or op == 'Mul' else ''
        print(f'  {op}: {count}{marker}')
    print()
    # Input shape
    print('--- Inputs ---')
    for inp in graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f'  {inp.name}: {shape}')
    print()

    print('--- Depth Analysis ---')
    print(f'  Estimated multiplicative depth: {total_depth}')
    print()

    return {
        'depth': total_depth,
        'op_counts': op_counts,
    }


def analyze_dag(dag_path):
    """Analyze compiled JSON DAG for multiplicative depth and layer info.

    Expects a JSON format with 'layer' and 'feature' top-level keys.
    Adapt the key names below to match your DAG format if different.
    """
    with open(dag_path, 'r') as f:
        data = json.load(f)

    layers = data.get('layer', {})
    features = data.get('feature', {})

    print(f'DAG: {dag_path}')
    print(f'  Layers: {len(layers)}')
    print(f'  Features: {len(features)}')
    print()

    # Layer types that consume CKKS levels
    level_consuming_types = {'conv2d', 'dense', 'square', 'poly_relu', 'upsample', 'matmul'}

    layer_types = {}
    depth_map = {}  # feature_id -> depth
    max_depth = 0

    # Topological analysis
    remaining = dict(layers)
    iterations = 0
    while remaining and iterations < 1000:
        iterations += 1
        progress = False
        to_remove = []

        for key, layer_info in remaining.items():
            inputs = layer_info.get('feature_input', [])
            if all(fi in depth_map for fi in inputs):
                input_depth = max(depth_map.get(fi, 0) for fi in inputs)
                ltype = layer_info.get('type', '')

                # Level consumption
                consumes = 1 if ltype in level_consuming_types else 0
                if ltype in ('bootstrapping', 'drop_level', 'batchnorm', 'batchnorm2d', 'identity'):
                    consumes = 0

                output_depth = input_depth + consumes
                outputs = layer_info.get('feature_output', [])
                for fo in outputs:
                    depth_map[fo] = output_depth
                    max_depth = max(max_depth, output_depth)

                layer_types[ltype] = layer_types.get(ltype, 0) + 1
                to_remove.append(key)
                progress = True

        for key in to_remove:
            del remaining[key]

        if not progress:
            print(f'  Warning: {len(remaining)} layers have unresolved dependencies')
            break
        # Print results
    print('--- Layer Types ---')
    for lt, count in sorted(layer_types.items()):
        marker = ' *' if lt in level_consuming_types else ''
        print(f'  {lt}: {count}{marker}')
    print()

    print('--- Depth Analysis ---')
    print(f'  Computed multiplicative depth: {max_depth}')
    print()

    return {
        'depth': max_depth,
        'layer_types': layer_types,
    }


def check_depth(depth, max_levels=None):
    """Check if the model fits within available levels."""
    print('--- Depth Check ---')
    print(f'  Multiplicative depth: {depth}')
    if max_levels is not None:
        if depth > max_levels:
            print(f'  EXCEEDS available levels: depth={depth} > max_levels={max_levels}')
            print(f'  Options: increase N (more levels), reduce depth, or use bootstrapping')
        else:
            print(f'  Fits within available levels: depth={depth} <= max_levels={max_levels}')
            print(f'  Levels remaining: {max_levels - depth}')
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze model for CKKS multiplicative depth')
    parser.add_argument('--onnx', help='Path to ONNX model file')
    parser.add_argument('--dag', help='Path to compiled JSON DAG')
    parser.add_argument('--max-levels', type=int, default=None,
                        help='Max available levels (to check if bootstrapping is needed)')
    parser.add_argument('--poly-degree', type=int, default=DEFAULT_POLY_DEGREE,
                        help='Polynomial degree for PolyRelu activations (default: 4)')
    args = parser.parse_args()

    if not args.onnx and not args.dag:
        print('Provide --onnx <path> or --dag <path>')
        parser.print_help()
        sys.exit(1)

    if args.onnx:
        print(f'Note: ONNX analysis estimates depth from op types. If polynomial')
        print(f'activations (PolyRelu) are used, specify --poly-degree (current: {args.poly_degree}).')
        print()
        result = analyze_onnx(args.onnx, poly_degree=args.poly_degree)
    else:
        result = analyze_dag(args.dag)

    check_depth(result['depth'], args.max_levels)


if __name__ == '__main__':
    main()