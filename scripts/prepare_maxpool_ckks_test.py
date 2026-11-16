"""Create a real CKKS test task with Retina spatial packing and automatic DP."""
import argparse
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import components
import transforms
from graph_partition_dp import compile_model_btp
from pipeline import prepare_graph, recompute_final_level
from processor import graph_to_task_config
from inference.model_generator.deploy_cmds import gen_custom_task
from training.model_compiler.maxpool_plan import build_maxpool_plan
from training.nn_tools.fixed_maxpool_profiles import get_maxpool_profile, available_maxpool_alphas


def prepare(root, channels, activation=None, calibration=None, *, alpha=7, upper_bound=1., B=1., precision_bits=40, gpu=False):
    profile = get_maxpool_profile(alpha)
    if not all(np.isfinite(v) and v > 0 for v in (upper_bound, B)):
        raise ValueError("upper_bound and B must be positive and finite")
    running_max = 1.
    if activation is not None:
        image = np.asarray(np.load(activation), dtype=np.float64)
        if image.ndim != 3 or image.shape[1:] != (256, 256):
            raise ValueError('Expected activation shape [channels,256,256]')
        channels = image.shape[0]
        running_max = float(json.loads(calibration.read_text())['running_max'])
        if not np.isfinite(running_max) or running_max <= 0:
            raise ValueError('Invalid calibrated running_max')
        if not np.isfinite(image).all() or np.max(np.abs(image)) > running_max:
            raise ValueError('Activation is nonfinite or outside calibrated range')
    else:
        image = np.random.default_rng(42).uniform(0., 1., (channels, 256, 256))
    config = components.config
    if gpu and precision_bits != 40:
        raise ValueError('GPU native CI validation uses PN15Default (40-bit scale)')
    config.fhe_param = components.PN15Default if gpu else {40: components.N16QP1546H192H32, 45: components.N16QP1547H192H32, 50: components.EXPERIMENTAL_N16S50H192H32}[precision_bits]
    config.block_shape = [128, 128]
    config.style = 'multiplexed'
    config.mpc_refresh = False
    config.use_gpu = gpu
    graph = components.LayerAbstractGraph()
    x = components.FeatureNode('x', 2, channels, 1, 'param0', 1, [256, 256])
    y = components.FeatureNode('y', 2, channels, 1, 'param0', 1, [128, 128])
    pool = components.PoolComputeNode('pool', 'maxpool2d', channels, channels,
        stride=[2, 2], kernel_shape=[3, 3], padding=[1, 1])
    pool.running_max = running_max
    pool.alpha, pool.upper_bound, pool.B = alpha, upper_bound, B
    graph.dag.add_edges_from([(x, pool), (pool, y)])
    for feature in (x, y):
        graph.dag.nodes[feature].update(skip=[1, 1], level=config.fhe_param.max_level, pack_num=1)
    graph = prepare_graph(graph)
    score, graph = compile_model_btp(graph, temperature=0)
    if graph is None:
        raise RuntimeError('No legal MaxPool DP plan')
    recompute_final_level(graph)
    transforms.insert_drop_level_layers(graph)
    server, client = root / 'server', root / 'client'
    server.mkdir(parents=True, exist_ok=True)
    client.mkdir(parents=True, exist_ok=True)
    graph.to_json({'param0': config.fhe_param}, str(server / 'nn_layers_ct_0.json'), score=score)
    graph_to_task_config(graph, str(server))
    shutil.copyfile(server / 'task_config.json', client / 'task_config.json')
    for directory in (server, client):
        (directory / 'ckks_parameter.json').write_text(json.dumps({'param0': config.fhe_param.to_dict()}))
    with h5py.File(server / 'model_parameters.h5', 'w'):
        pass
    gen_custom_task(str(server), param_name=config.fhe_param.name, style='multiplexed', use_gpu=gpu)
    np.savetxt(client / 'x.csv', image.reshape(-1), delimiter=',')
    (root / 'test_input.json').write_text(json.dumps({
        'activation': str(activation) if activation else None,
        'calibration': str(calibration) if calibration else None,
        'channels': channels, 'running_max': running_max,
        'alpha': alpha, 'upper_bound': upper_bound, 'B': B, 'degrees': profile['degrees'],
        'precision_bits': precision_bits, 'parameter_name': config.fhe_param.name, 'gpu': gpu,
        'input_min': float(image.min()), 'input_max': float(image.max()),
    }, indent=2))
    rr, cc = np.meshgrid(np.arange(128) * 2, np.arange(128) * 2, indexing='ij')
    taps, exact_taps = [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            valid = (rr + dr >= 0) & (rr + dr < 256) & (cc + dc >= 0) & (cc + dc < 256)
            values = image[:, np.where(valid, rr + dr, rr), np.where(valid, cc + dc, cc)]
            taps.append(values * upper_bound / running_max)
            exact_taps.append(np.where(valid, values, -np.inf))
    plan = build_maxpool_plan({'kernel_shape': [3, 3], 'stride': [2, 2], 'padding': [1, 1]}, [256, 256])
    coefficients = [[v for c in odd for v in (0., float(c))]
                    for odd in profile['odd_coefficients']]
    for r, round_plan in enumerate(plan['rounds']):
        bound = B * (1 if r == 0 else 1.05) * 2**r
        result = []
        for i, j in round_plan['pairs']:
            a, b = taps[i], taps[j]
            sign = (a - b) / (2 * bound)
            for coeff in coefficients:
                sign = np.polynomial.polynomial.polyval(sign, coeff)
            result.append(a + b + (a - b) * sign)
        result.extend(2 * taps[i] for i in round_plan['carry'])
        taps = result
    np.savetxt(root / 'polynomial.csv', (taps[0] * (running_max / upper_bound / 16)).reshape(-1), delimiter=',')
    np.savetxt(root / 'exact.csv', np.max(exact_taps, axis=0).reshape(-1), delimiter=',')
    print(f'Ready: {root}, channels={channels}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/maxpool-staged-ckks'))
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', help='Generate native CI PN15 GPU instructions')
    parser.add_argument('--precision-bits', type=int, choices=(40,45,50), default=40)
    parser.add_argument('--alpha', type=int, choices=available_maxpool_alphas(), default=7)
    parser.add_argument('--upper-bound', type=float, default=1.)
    parser.add_argument('--B', type=float, default=1.)
    parser.add_argument('--activation', type=Path)
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    if args.channels <= 0:
        parser.error('--channels must be positive')
    if (args.activation is None) != (args.calibration is None):
        parser.error('--activation and --calibration must be supplied together')
    prepare(args.output, args.channels, args.activation, args.calibration,
            alpha=args.alpha, upper_bound=args.upper_bound, B=args.B, precision_bits=args.precision_bits, gpu=args.gpu)
