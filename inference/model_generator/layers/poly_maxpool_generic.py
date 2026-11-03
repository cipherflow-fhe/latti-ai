"""General deferred-half polynomial MaxPool DAG composition."""

from inference.model_generator.layers.poly_maxpool_taps import (
    call_normalized_maxpool_taps,
    call_interleaved_normalized_maxpool_taps,
)
from inference.model_generator.layers.poly_maxpool2d_layer import (
    reduce_poly_maxpool_stockmeyer,
    call_poly_maxpool_reduction_round,
)


def flatten_maxpool_state(groups, channels):
    """Feature boundary order: reduction group first, then CNN channel.

    Each entry is one dense spatial plane in its own ciphertext. These are
    live reduction groups, not packed CNN channels. A generic Bootstrap must
    refresh every entry in this list without reordering it.
    """
    if channels <= 0 or not groups or any(len(group) != channels for group in groups):
        raise ValueError('Invalid MaxPool live group/channel counts')
    nodes = [ct for group in groups for ct in group]
    if len({ct.level for ct in nodes}) != 1:
        raise ValueError('MaxPool live state levels differ')
    return nodes


def call_maxpool_round_stage(nodes, plan, round_index, weights, channels):
    """Execute a single compiler round on a flattened live ciphertext set.

    Generic drop-level/Bootstrap nodes may run before this function. This
    function neither chooses nor inserts a Bootstrap.
    """
    if not 0 <= round_index < len(plan['rounds']):
        raise ValueError('Invalid MaxPool round index')
    expected_groups = plan['kernel_shape'][0] * plan['kernel_shape'][1]
    if round_index:
        previous = plan['rounds'][round_index - 1]
        expected_groups = len(previous['pairs']) + len(previous['carry'])
    if channels <= 0 or len(nodes) != expected_groups * channels:
        raise ValueError('Incomplete MaxPool live state at stage boundary')
    groups = [nodes[i:i + channels] for i in range(0, len(nodes), channels)]
    flatten_maxpool_state(groups, channels)
    outputs = call_poly_maxpool_reduction_round(
        groups, plan['rounds'][round_index], weights,
        input_shape=plan['output_shape'], skip=[1, 1], n_channel_per_ct=1,
    )
    return flatten_maxpool_state(outputs, channels)


def call_interleaved_deferred_half_maxpool_dag(
    x_cts, plan, tap_specs, mask_pt_nodes, weights_by_round, channels,
):
    """Normalize interleaved inputs and reduce to one output plane/channel.

    The tap builder writes a dense output plane, so the old stride-2 anchor
    repacker must not be applied. Channel packing and restoration are separate
    operations. This fused path requires sufficient levels; it does not execute
    bootstraps from a compiler stage graph.
    """
    if list(plan['kernel_shape']) != [3, 3] or list(plan['stride']) != [2, 2]:
        raise NotImplementedError('Interleaved reduction requires 3x3 stride-2 MaxPool')
    if not x_cts or channels <= 0 or len(x_cts) != channels * 4:
        raise ValueError('Expected four interleaved ciphertexts per channel')
    if len(tap_specs) != 9 or len(weights_by_round) != len(plan['rounds']):
        raise ValueError('Incorrect tap or polynomial round count')
    levels = {ct.level for ct in x_cts}
    if len(levels) != 1:
        raise ValueError('Input ciphertext levels differ')
    start = next(iter(levels))
    cost = 1 + 7 * len(plan['rounds'])
    if start - cost < 1:
        raise ValueError('Insufficient levels for fused interleaved MaxPool reduction')
    taps = call_interleaved_normalized_maxpool_taps(
        x_cts, tap_specs, mask_pt_nodes, channels,
    )
    if any(ct.level != start - 1 for row in taps for ct in row):
        raise ValueError('Incorrect normalized tap level')
    result = reduce_poly_maxpool_stockmeyer(
        taps, plan, weights_by_round,
        input_shape=plan['output_shape'], skip=[1, 1], n_channel_per_ct=1,
    )
    if len(result) != channels or any(ct.level != start - cost for ct in result):
        raise ValueError('Incorrect interleaved reduction output count or level')
    return result


def call_deferred_half_maxpool_dag(
    x_cts,
    plan,
    tap_specs,
    mask_pt_nodes,
    weights_by_round,
    input_shape,
    skip,
    n_channel_per_ct,
):
    """
    Compose two existing CKKS DAG stages:

        1. Rotate + fused padding/normalization masks
        2. Deferred-half polynomial Max reduction

    Returns:
        Ciphertext nodes at output-anchor positions,
        scaled by 2**compare_depth.

    Does NOT perform output repacking or scale restoration.
    """

    if not x_cts:
        raise ValueError("Empty MaxPool ciphertext input")

    if tuple(skip) != (1, 1):
        raise NotImplementedError(
            "Generic MaxPool currently requires skip=(1,1)"
        )

    if tuple(input_shape) != tuple(plan["input_shape"]):
        raise ValueError("MaxPool input shape mismatch")

    expected_taps = (
        plan["kernel_shape"][0]
        * plan["kernel_shape"][1]
    )

    if len(tap_specs) != expected_taps:
        raise ValueError("Incorrect MaxPool tap count")

    depth = plan["compare_depth"]

    if len(weights_by_round) != depth:
        raise ValueError("Incorrect polynomial round count")

    levels = {ct.level for ct in x_cts}

    if len(levels) != 1:
        raise ValueError("Input ciphertext levels differ")

    input_level = next(iter(levels))

    # Estimated complete path:
    # fused masks: 1
    # each deferred-half binary max: 7
    # repacking: 1
    # restoration: 1
    # reserve >= 1 output level
    required = 1 + 7 * depth + 2 + 1

    if input_level < required:
        raise ValueError(
            f"Insufficient CKKS levels: "
            f"input={input_level}, required={required}"
        )

    taps = call_normalized_maxpool_taps(
        x_cts,
        tap_specs,
        mask_pt_nodes,
    )

    tap_level = input_level - 1

    if any(
        ct.level != tap_level
        for row in taps
        for ct in row
    ):
        raise ValueError("Incorrect normalized tap level")

    result = reduce_poly_maxpool_stockmeyer(
        taps=taps,
        plan=plan,
        weights_by_round=weights_by_round,
        input_shape=input_shape,
        skip=skip,
        n_channel_per_ct=n_channel_per_ct,
    )

    expected_level = tap_level - 7 * depth

    if any(ct.level != expected_level for ct in result):
        raise ValueError(
            "Incorrect MaxPool reduction output level"
        )

    return result
