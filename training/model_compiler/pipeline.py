# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import components
from components import LayerAbstractGraph, FheParameter, config
import processor
from processor import *
from graph_partition_dp import *


def init_config_with_args(poly_n=None, style=None, graph_type=None):
    """
    Initialize configuration based on command line arguments

    Args:
        poly_n: Polynomial modulus degree (POLY_N)
        style: Computation style (STYLE)
        graph_type: Graph type (GRAPH_TYPE)
    """
    if style is not None:
        config.style = style
    if graph_type is not None:
        config.graph_type = graph_type

    current_graph_type = config.graph_type

    # Automatically set MAX_LEVEL based on POLY_N and GRAPH_TYPE
    if current_graph_type == 'btp':
        poly_n_to_max_level = {65536: 9}
    else:
        poly_n_to_max_level = {8192: 5, 16384: 9, 32768: 17, 65536: 33}

    poly_n_to_block_shape = {65536: [128, 128], 32768: [128, 128], 16384: [64, 64], 8192: [64, 64]}

    current_poly_n = poly_n if poly_n is not None else config.poly_n
    max_level = poly_n_to_max_level.get(current_poly_n)
    if max_level is None:
        print(f'Warning: No MAX_LEVEL mapping for POLY_N={current_poly_n}, using value from config.json')
        max_level = config.fhe_param.max_level
    block_shape = poly_n_to_block_shape.get(current_poly_n, [64, 64])

    config.fhe_param = FheParameter(
        poly_modulus_degree=current_poly_n,
        n_mult_level=max_level,
        block_shape=block_shape,
    )

    print(f'Automatically set MAX_LEVEL={max_level} based on GRAPH_TYPE={current_graph_type}, POLY_N={current_poly_n}')
    print(f'Automatically set block_shape={block_shape} based on POLY_N={current_poly_n}')
    print(
        f'Configuration initialized: POLY_N={current_poly_n}, STYLE={config.style}, GRAPH_TYPE={current_graph_type}, MAX_LEVEL={max_level}, block_shape={block_shape}'
    )


def prepare_graph(raw_graph: LayerAbstractGraph) -> LayerAbstractGraph:
    """
    Prepare graph for compilation (common preparation steps)

    Args:
        raw_graph: Raw LayerAbstractGraph loaded from json

    Returns:
        Prepared LayerAbstractGraph
    """
    pt_graph = copy.deepcopy(raw_graph)

    substitute_layers_for_btp(pt_graph)
    # transforms.init_levels(pt_graph)
    # update_shape_for_btp(pt_graph)
    # update_skip_for_btp(pt_graph)
    # update_level_cost_for_btp(pt_graph)
    set_is_adaptive_avgpool(pt_graph)
    transforms.split_upsampling_layers(pt_graph)
    transforms.infer_shapes_and_skips(pt_graph)
    transforms.combine_convs_with_upsamples(pt_graph)
    transforms.set_level_costs(pt_graph)

    transforms.absorb_scale(pt_graph)

    return pt_graph


def try_no_btp(raw_graph: LayerAbstractGraph) -> tuple[bool, LayerAbstractGraph | None, float]:
    """
    Try no-BTP mode compilation with prepared graph

    Args:
        raw_graph: Raw LayerAbstractGraph

    Returns:
        (succeeded, graph, score): succeeded=True if no-BTP succeeded, graph and score are set on success
    """
    print('Step 2: Trying no-BTP mode...')

    # not btp style, set max level for polyrelu
    no_btp_params = [
        components.FheParameter(8192, 5, [64, 64]),
        components.FheParameter(16384, 9, [64, 64]),
        components.FheParameter(32768, 17, [128, 128]),
        components.FheParameter(65536, 33, [128, 256]),
    ]

    for params in no_btp_params:
        config.fhe_param = params
        print(
            f'Trying POLY_N={config.poly_n}, MAX_LEVEL={config.fhe_param.max_level}, block_shape={config.block_shape}'
        )

        # (1) Pre-process
        pt_graph = prepare_graph(raw_graph)

        # (2) Process
        result = process_with_no_btp(pt_graph)

        # (3) Post-process
        if result is not None:
            print(f'Success! Using POLY_N={config.poly_n}, MAX_LEVEL={config.fhe_param.max_level}')
            print('✓ No-BTP mode succeeded! Saving results...')
            restore_node_attributes(result.dag)
            result = post_process(result)
            print(f'\n=== No-BTP Results ===')
            print(f'Score: 0.0')
            return True, result, 0.0
        else:
            print(f'Level exceeded with POLY_N={config.poly_n}, trying next level...')

    print(f'Warning: Even with POLY_N=65536, level still exceeds limit!')
    print('✗ No-BTP mode failed, switching to BTP mode...')
    return False, None, float('inf')


def process_with_no_btp(graph: LayerAbstractGraph):
    return reset_level_and_check_level(graph)


def try_btp(
    num_experiments: int,
    raw_graph: LayerAbstractGraph,
    temperature: float,
    num_workers: int,
) -> tuple[bool, LayerAbstractGraph | None, float]:
    btp_param_list = [
        components.FheParameter(65536, 9, [128, 128]),
    ]
    valid_results = []
    for params in btp_param_list:
        config.fhe_param = params

        # (1) Pre-process
        pt_graph = prepare_graph(raw_graph)

        # (2) Process
        graph, score = run_btp_compilation(num_experiments, pt_graph, temperature, num_workers)

        # (3) Post-process
        if graph is not None:
            graph = post_process(graph)
            valid_results.append((score, graph))

    if not valid_results:
        return False, None, float('inf')

    best_score, best_graph = min(valid_results, key=lambda x: x[0])
    return True, best_graph, best_score


def run_btp_compilation(
    num_experiments: int,
    pt_graph: LayerAbstractGraph,
    temperature: float,
    num_workers: int,
) -> tuple[LayerAbstractGraph | None, float]:
    """
    Run BTP mode parallel compilation with prepared graph

    Args:
        num_experiments: Number of parallel compilation runs
        temperature: Temperature parameter for randomization
        pt_graph: Prepared graph for BTP compilation
        num_workers: Number of parallel worker processes

    Returns:
        (best_graph, best_score): best_graph is None if all runs failed
    """
    print(f'Step 4: Starting {num_experiments} parallel BTP compilations with {num_workers} processes...')

    # Prepare arguments for each run
    args_list = [(copy.deepcopy(pt_graph), temperature) for _ in range(num_experiments)]

    # Run compilations in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_single_compile, args_list))
    # results = [run_single_compile(*args_list)]

    # Filter out failed results
    valid_results = [(score, graph) for score, graph in results if graph is not None]
    failed_count = num_experiments - len(valid_results)

    print(f'\n=== Summary ===')
    print(f'Total runs: {num_experiments}')
    print(f'Successful: {len(valid_results)}')
    print(f'Failed (level limit exceeded): {failed_count}')

    if not valid_results:
        print('ERROR: All runs failed! No valid results to save.')
        return None, float('inf')

    # Find the best result
    best_score, best_graph = min(valid_results, key=lambda x: x[0])

    print(f'\n=== Results ===')
    print(f'Best score: {best_score}')
    return best_graph, best_score


def post_process(graph: LayerAbstractGraph):
    slot_num = config.poly_n / 2
    for node in graph.dag.nodes:
        if isinstance(node, ComputeNode):
            node.up_scale_str = list()
            node.down_scale_str = list()
            populate_pack_num(graph.dag, node, slot_num)

    transforms.set_graph_scale(graph)
    process_levels(graph)

    return graph


def dump_graph(
    graph: LayerAbstractGraph,
    output_dir: Path,
    score: float,
    use_btp: bool,
):
    task_dir = output_dir / 'task'
    server_dir = task_dir / 'server'
    client_dir = task_dir / 'client'
    ergs_dir = server_dir

    ergs_dir.mkdir(parents=True, exist_ok=True)
    client_dir.mkdir(parents=True, exist_ok=True)

    erg0_path = ergs_dir / 'nn_layers_ct_0.json'
    graph.to_json(dict(), str(erg0_path), score=score)

    if use_btp:
        graph_to_task_config(graph, str(server_dir))
    else:
        graph_to_task_config(graph, str(server_dir), False)

    server_task_config = server_dir / 'task_config.json'
    client_task_config = client_dir / 'task_config.json'
    if server_task_config.exists():
        shutil.copy(str(server_task_config), str(client_task_config))

    ckks_param = {'param0': {**config.fhe_param.to_dict(), 'pack_num': 4.0}}

    with open(server_dir / 'ckks_parameter.json', 'w') as f:
        json.dump(ckks_param, f, indent=4)

    with open(client_dir / 'ckks_parameter.json', 'w') as f:
        json.dump(ckks_param, f, indent=4)


def run_pipeline(
    num_experiments: int,
    input_file_path: Path,
    output_dir: Path,
    temperature: float,
    num_workers: int = 16,
):
    """
    Run multiple compilations in parallel and select the best result

    This is the main entry point for compilation. It tries no-BTP mode first,
    and falls back to BTP mode if needed.

    Args:
        num_experiments: Number of parallel compilation runs
        input_file_path: Input pt.json file path
        output_dir: Output directory (will contain erg0.json, task_config.json)
        temperature: Temperature parameter for randomization
        num_workers: Number of parallel worker processes
    """
    print(f'Starting compilation...')

    raw_graph = LayerAbstractGraph.from_json(str(input_file_path))

    use_btp = False
    succeeded, graph, score = try_no_btp(raw_graph)
    if not succeeded:
        use_btp = True
        succeeded, graph, score = try_btp(num_experiments, raw_graph, temperature, num_workers)
        if not succeeded:
            raise ValueError('Compilation failed.')

    dump_graph(graph, output_dir, score, use_btp=use_btp)

    return graph, score
