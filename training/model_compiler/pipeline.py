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
from components import LayerAbstractGraph, config
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
    # If command line arguments are provided, override config file values
    if poly_n is not None:
        config.poly_n = poly_n
    if style is not None:
        config.style = style
    if graph_type is not None:
        config.graph_type = graph_type

    # Get current values from config
    current_poly_n = config.poly_n
    current_style = config.style
    current_graph_type = config.graph_type

    # Automatically set MAX_LEVEL based on POLY_N and GRAPH_TYPE
    if current_graph_type == 'btp':
        # BTP version configuration
        poly_n_to_max_level = {65536: 9}
    else:
        # Non-BTP version configuration
        poly_n_to_max_level = {8192: 5, 16384: 9, 32768: 17, 65536: 33}

    max_level = poly_n_to_max_level.get(current_poly_n)
    if max_level is not None:
        config.max_level = max_level
        print(
            f'Automatically set MAX_LEVEL={max_level} based on GRAPH_TYPE={current_graph_type}, POLY_N={current_poly_n}'
        )
    else:
        print(f'Warning: No MAX_LEVEL mapping for POLY_N={current_poly_n}, using value from config.json')

    # Automatically set block_shape based on POLY_N
    poly_n_to_block_shape = {65536: [128, 128], 32768: [128, 128], 16384: [64, 64], 8192: [64, 64]}
    block_shape = poly_n_to_block_shape.get(current_poly_n, [64, 64])
    config.block_shape = block_shape
    print(f'Automatically set block_shape={block_shape} based on POLY_N={current_poly_n}')

    print(
        f'Configuration initialized: POLY_N={current_poly_n}, STYLE={current_style}, GRAPH_TYPE={current_graph_type}, MAX_LEVEL={config.max_level}, block_shape={config.block_shape}'
    )


def prepare_graph(input_file_path: Path) -> LayerAbstractGraph:
    """
    Prepare graph for compilation (common preparation steps)

    Args:
        input_file_path: Input pt.json file path

    Returns:
        Prepared LayerAbstractGraph
    """
    pt_graph = LayerAbstractGraph.from_json(str(input_file_path))

    substitute_layers_for_btp(pt_graph)
    transforms.init_levels(pt_graph)
    set_is_adaptive_avgpool(pt_graph)
    update_shape_for_btp(pt_graph)
    update_skip_for_btp(pt_graph)
    update_level_cost_for_btp(pt_graph)
    transforms.absorb_scale(pt_graph)

    return pt_graph


def try_no_btp(pt_graph: LayerAbstractGraph) -> tuple[bool, LayerAbstractGraph | None, float]:
    """
    Try no-BTP mode compilation with prepared graph

    Args:
        pt_graph: Prepared LayerAbstractGraph

    Returns:
        (succeeded, graph, score): succeeded=True if no-BTP succeeded, graph and score are set on success
    """
    print('Step 2: Trying no-BTP mode...')
    result = process_with_no_btp(pt_graph)

    if result:
        print('✓ No-BTP mode succeeded! Saving results...')

        total_graph = result
        restore_node_attributes(total_graph.dag)

        print(f'\n=== No-BTP Results ===')
        print(f'Score: 0.0')
        return True, total_graph, 0.0

    print('✗ No-BTP mode failed, switching to BTP mode...')
    return False, None, float('inf')


def process_with_no_btp(graph: LayerAbstractGraph):
    current_graph_type = config.graph_type

    # not btp style, set max level for polyrelu
    poly_n_to_max_level = {8192: 5, 16384: 9, 32768: 17, 65536: 33}

    poly_n_to_block_shape = {8192: [64, 64], 16384: [64, 64], 32768: [128, 128], 65536: [128, 256]}
    poly_n_levels = [8192, 16384, 32768, 65536]  # always start trying from 8192

    result = None
    for poly_n in poly_n_levels:
        # Update configuration
        config.poly_n = poly_n
        config.max_level = poly_n_to_max_level[poly_n]
        config.block_shape = poly_n_to_block_shape[poly_n]

        print(f'Trying POLY_N={poly_n}, MAX_LEVEL={config.max_level}, block_shape={config.block_shape}')

        # Check whether the level meets the requirements
        result = reset_level_and_check_level(graph)

        if result is not None:
            print(f'Success! Using POLY_N={poly_n}, MAX_LEVEL={config.max_level}')
            break
        else:
            print(f'Level exceeded with POLY_N={poly_n}, trying next level...')

    if result is None:
        print(f'Warning: Even with POLY_N=65536, level still exceeds limit!')

    return result


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
    print('Step 3: Restoring to BTP parameters (POLY_N=65536, MAX_LEVEL=9)...')

    config.poly_n = 65536
    config.max_level = 9
    config.block_shape = [128, 128]

    print(f'Step 4: Starting {num_experiments} parallel BTP compilations with {num_workers} processes...')

    # Prepare arguments for each run
    args_list = [(copy.deepcopy(pt_graph), temperature) for _ in range(num_experiments)]

    # Run compilations in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_single_compile, args_list))

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

    poly_n = config.poly_n
    poly_to_mod = {8192: 30, 16384: 34, 32768: 40, 65536: 45}
    mod_bit = poly_to_mod[poly_n]
    ckks_param = {
        'param0': {
            'poly_modulus_degree': poly_n,
            'n_mult_level': config.max_level,
            'coeff_modulus_bit_length': mod_bit,
            'special_prime_bit_length': mod_bit,
            'pack_num': 4.0,
        }
    }

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

    print('Step 1: Preparing graph...')
    pt_graph = prepare_graph(input_file_path)

    use_btp = False
    succeeded, graph, score = try_no_btp(pt_graph)
    if not succeeded:
        use_btp = True
        graph, score = run_btp_compilation(num_experiments, pt_graph, temperature, num_workers)
        if graph is None:
            raise ValueError('Compilation failed.')

    graph = post_process(graph)

    dump_graph(graph, output_dir, score, use_btp=use_btp)

    return graph, score
