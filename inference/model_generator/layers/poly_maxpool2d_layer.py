
from inference.lattisense.frontend.custom_task import *
from inference.model_generator.layers.poly2d import Poly2D
from inference.model_generator.layers.mult_scaler import MultScalarLayer

from inference.model_generator.layers.range_norm_maxpool import (
    PolyMaxPoolConfig,
    scaling_plan,
    apply_range_normalization,
    restore_range_after_repack,
)


def _evaluate_sign_stage(diff, weight_pt, degree, input_shape, skip, n_channel_per_ct):
    """Evaluate one fixed-profile sign polynomial stage."""
    poly = Poly2D(input_shape, degree, skip, n_channel_per_ct)
    return poly.call_stockmeyer_feature2d(diff, weight_pt)


def _evaluate_sign_chain(diff, stages, input_shape, skip, n_channel_per_ct):
    """Evaluate an ordered fixed-profile polynomial chain.

    ``stages`` is a sequence of ``{'degree': odd int, 'weights': ...}``.
    Keeping the descriptor explicit prevents a coefficient table from being
    silently evaluated with the wrong Poly2D degree.
    """
    if not stages:
        raise ValueError('MaxPool sign chain must contain at least one stage')
    # Validate the complete chain before emitting any polynomial nodes.
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {'degree', 'weights'}:
            raise ValueError('Invalid MaxPool sign stage descriptor')
        degree = stage['degree']
        if isinstance(degree, bool) or not isinstance(degree, int) or degree <= 0 or degree % 2 == 0:
            raise ValueError('MaxPool sign stage degree must be a positive odd integer')
        Poly2D.compute_stockmeyer_level_cost(degree)
        weights = stage['weights']
        if len(weights) != degree + 1 or any(len(row) != len(diff) for row in weights):
            raise ValueError('MaxPool sign stage weights must match degree and ciphertext count')
    current = diff
    for stage in stages:
        current = _evaluate_sign_stage(
            current, stage['weights'], stage['degree'], input_shape, skip, n_channel_per_ct,
        )
    return current


def binary_max_stockmeyer(
    a,
    b,
    weight_pt0,
    weight_pt1,
    half_pt,
    input_shape,
    skip,
    n_channel_per_ct,
    sign_stages=None,
):
    """
    Approximate element-wise max(a, b).

    Default P(d) = p1(p0(d)), both degree 7. sign_stages overrides
    the default with an ordered chain of bound polynomial weights.
    max(a,b) ≈ (a + b + (a-b)*P(a-b)) / 2.

    Deferred-half level cost: sum(Poly2D stage costs) + 1.
    An explicit half_pt consumes one additional level.
    """

    assert len(a) == len(b)
    if half_pt is not None:
        assert len(half_pt) == len(a)
    # Step 1: d = a - b
    diff = [sub(x, y) for x, y in zip(a, b)]

    # Step 2: p0(d)
    if sign_stages is None:
        sign_stages = [
            {'degree': 7, 'weights': weight_pt0},
            {'degree': 7, 'weights': weight_pt1},
        ]
    sign_approx = _evaluate_sign_chain(
        diff, sign_stages, input_shape, skip, n_channel_per_ct,
    )

    result = []

    for x, y, d, p in zip(a, b, diff, sign_approx):
        # Step 4: d * P(d)
        d_aligned = drop_level(
            d, d.metadata.level - p.metadata.level
        )
        product = rescale(
            relin(mult(d_aligned, p))
        )

        # Step 5: a + b
        summation = add(x, y)
        summation = drop_level(
            summation,
            summation.metadata.level - product.metadata.level
        )

        # Step 6: a + b + d*P(d)
        result.append(add(summation, product))

    # Step 7: multiply by 0.5
    if half_pt is None:
        return result
    return MultScalarLayer().call(result, half_pt)


def repack_maxpool2x2_multiplexed(
    x,
    select_tensor_pt,
    n_channel,
    n_channel_per_ct,
    input_shape,
    skip,
):
    """
    Select and repack 2x2 MaxPooling results.

    Input:
        Results of two rounds of binary Max.
        Original multiplexed feature layout.

    Output:
        Repacked ciphertext nodes for a 2x2,
        stride-2 pooling layer.

    Level cost: 1 (mask multiplication + rescale).
    """

    stride = (2, 2)
    shape = input_shape

    out_channels_per_ct = (
        n_channel_per_ct * stride[0] * stride[1]
    )

    expected_n_ct = (
        n_channel + n_channel_per_ct - 1
    ) // n_channel_per_ct

    assert len(x) == expected_n_ct
    assert all(ct.metadata.level >= 1 for ct in x)
    assert len(select_tensor_pt) >= min(
        n_channel, out_channels_per_ct
    )

    result_tmp = []

    # Stage 1: Rotation and mask selection
    for idx, ct in enumerate(x):
        n_valid = min(
            n_channel_per_ct,
            n_channel - idx * n_channel_per_ct,
        )

        steps = []

        for i in range(n_valid):
            channel_id = idx * n_channel_per_ct + i

            rp = channel_id % out_channels_per_ct

            r_num0 = (
                rp // (
                    skip[0] * skip[1]
                    * stride[0] * stride[1]
                )
            ) * (
                skip[0] * skip[1]
                * shape[0] * shape[1]
            )

            r_num1 = (
                (
                    rp % (
                        skip[0] * skip[1]
                        * stride[0] * stride[1]
                    )
                ) // (stride[1] * skip[1])
            ) * shape[1] * skip[1]

            r_num2 = rp % (skip[1] * stride[1])

            lp = channel_id % n_channel_per_ct

            l_num0 = (
                lp // (skip[0] * skip[1])
            ) * (
                skip[0] * skip[1]
                * shape[0] * shape[1]
            )

            l_num1 = (
                (
                    lp % (skip[0] * skip[1])
                ) // skip[1]
            ) * shape[1] * skip[1]

            l_num2 = lp % skip[1]

            step = (
                -r_num0 - r_num1 - r_num2
                + l_num0 + l_num1 + l_num2
            )

            steps.append(step)

        # Reuse rotations for identical steps
        non_zero_steps = sorted(
            set(s for s in steps if s != 0)
        )

        rotated = {0: ct}

        if non_zero_steps:
            rotated_list = rotate_cols(
                ct, non_zero_steps
            )

            for step, rotated_ct in zip(
                non_zero_steps, rotated_list
            ):
                rotated[step] = rotated_ct

        # Apply 0/1 selection masks
        for i in range(n_valid):
            channel_id = (
                idx * n_channel_per_ct + i
            )

            out_pos = (
                channel_id % out_channels_per_ct
            )

            select_pt = select_tensor_pt[out_pos]
            selected_ct = rotated[steps[i]]

            masked = rescale(
                mult(selected_ct, select_pt)
            )

            result_tmp.append(masked)

    # Stage 2: Repack channels
    output_ct = []

    for start in range(
        0, n_channel, out_channels_per_ct
    ):
        end = min(
            start + out_channels_per_ct,
            n_channel,
        )

        packed = result_tmp[start]

        for i in range(start + 1, end):
            packed = add(packed, result_tmp[i])

        output_ct.append(packed)

    return output_ct


def poly_maxpool2x2_range_norm_stockmeyer(
    x,
    input_norm_pt,
    output_restore_pt,
    weight_pt0_r1,
    weight_pt1_r1,
    weight_pt0_r2,
    weight_pt1_r2,
    half_pt_r1,
    half_pt_r2,
    select_tensor_pt,
    n_channel,
    n_channel_per_ct,
    input_shape,
    skip,
    running_max,
    config=None,
):
    """
    Range-normalized CKKS polynomial 2x2 MaxPooling.

    Pipeline:
        1. Normalize encrypted input
        2. Horizontal approximate Max
        3. Vertical approximate Max
        4. Repack output
        5. Restore output scale

    The normalization constants and polynomial
    coefficients are encoded before FHE execution.
    """

    if config is None:
        config = PolyMaxPoolConfig()

    plan = scaling_plan(
        running_max=running_max,
        config=config,
    )

    # ---------------------------------------------
    # Stage 1: Normalize encrypted input
    # ---------------------------------------------
    normalized_x = apply_range_normalization(
        x,
        input_norm_pt,
    )

    # ---------------------------------------------
    # Stage 2: Horizontal approximate Max
    # ---------------------------------------------
    horizontal_step = skip[1]

    right_x = [
        rotate_cols(ct, [horizontal_step])[0]
        for ct in normalized_x
    ]

    horizontal_max = binary_max_stockmeyer(
        normalized_x,
        right_x,
        weight_pt0_r1,
        weight_pt1_r1,
        half_pt_r1,
        input_shape,
        skip,
        n_channel_per_ct,
    )

    # ---------------------------------------------
    # Stage 3: Vertical approximate Max
    # ---------------------------------------------
    vertical_step = (
        input_shape[1] * skip[0] * skip[1]
    )

    lower_x = [
        rotate_cols(ct, [vertical_step])[0]
        for ct in horizontal_max
    ]

    vertical_max = binary_max_stockmeyer(
        horizontal_max,
        lower_x,
        weight_pt0_r2,
        weight_pt1_r2,
        half_pt_r2,
        input_shape,
        skip,
        n_channel_per_ct,
    )

    # ---------------------------------------------
    # Stage 4: Repack 2x2 pooling output
    # ---------------------------------------------
    packed_output = repack_maxpool2x2_multiplexed(
        vertical_max,
        select_tensor_pt,
        n_channel,
        n_channel_per_ct,
        input_shape,
        skip,
    )

    # ---------------------------------------------
    # Stage 5: Restore original output scale
    # ---------------------------------------------
    output = restore_range_after_repack(
        packed_output,
        output_restore_pt,
    )

    return output, plan


def reduce_poly_maxpool_stockmeyer(
    taps,
    plan,
    weights_by_round,
    input_shape,
    skip,
    n_channel_per_ct,
):
    """
    Build a deferred-half MaxPool CKKS reduction DAG.

    taps:
        One ciphertext list per kernel position.
        Each tap must already have the correct boundary handling
        and range normalization.

    weights_by_round:
        [{"p0": ..., "p1": ...}, ...]
        One coefficient set for each reduction round.

    Returns ciphertexts scaled by 2**compare_depth.
    The caller must fold the inverse factor into output_restore.
    """
    kh, kw = plan["kernel_shape"]
    expected_taps = kh * kw
    rounds = plan["rounds"]

    if len(taps) != expected_taps:
        raise ValueError("Incorrect number of MaxPool taps")

    if len(weights_by_round) != len(rounds):
        raise ValueError("Incorrect number of polynomial rounds")

    n_ct = len(taps[0])
    if n_ct == 0:
        raise ValueError("Empty MaxPool ciphertext list")

    current = [list(tap) for tap in taps]

    if any(len(tap) != n_ct for tap in current):
        raise ValueError("Inconsistent ciphertext counts")

    for round_index, round_plan in enumerate(rounds):
        weights = weights_by_round[round_index]
        current = call_poly_maxpool_reduction_round(
            current, round_plan, weights, input_shape, skip, n_channel_per_ct,
        )

    if len(current) != 1:
        raise ValueError("Incomplete MaxPool reduction")

    return current[0]


def call_poly_maxpool_reduction_round(current, round_plan, weights, input_shape, skip, n_channel_per_ct):
    """Execute one existing deferred-half round, preserving all live groups.

    The returned group list is the complete state to refresh at a DP boundary.
    No Bootstrap is scheduled here. The fused entry point uses this same code.
    """
    if not current or not current[0]:
        raise ValueError("Empty MaxPool round state")
    n_ct = len(current[0])
    if any(len(group) != n_ct for group in current):
        raise ValueError("Inconsistent round ciphertext counts")
    levels = {ct.metadata.level for group in current for ct in group}
    chain = weights.get('sign_stages')
    cost = (sum(Poly2D.compute_stockmeyer_level_cost(s['degree']) for s in chain) + 1
            if chain else 7)
    if len(levels) != 1 or next(iter(levels)) < cost:
        raise ValueError(f"MaxPool round requires aligned inputs and at least {cost} levels")
    used = [index for pair in round_plan['pairs'] for index in pair] + list(round_plan['carry'])
    if sorted(used) != list(range(len(current))):
        raise ValueError("Round must consume every live group exactly once")
    next_values = []
    target_level = None

    for i, j in round_plan["pairs"]:
        left = current[i]
        right = current[j]

        result = binary_max_stockmeyer(
            a=left,
            b=right,
            weight_pt0=weights["p0"],
            weight_pt1=weights["p1"],
            half_pt=None,
            input_shape=input_shape,
            skip=skip,
            n_channel_per_ct=n_channel_per_ct,
            sign_stages=chain,
        )

        if len(result) != n_ct:
            raise ValueError("Binary Max ciphertext count mismatch")

        levels = {ct.metadata.level for ct in result}
        if len(levels) != 1:
            raise ValueError("Binary Max output levels differ")

        result_level = result[0].metadata.level

        if target_level is None:
            target_level = result_level
        elif result_level != target_level:
            raise ValueError("Binary Max round level mismatch")

        next_values.append(result)

    if target_level is None:
        raise ValueError("Reduction round contains no comparisons")

    # Carry nodes must be doubled AND aligned to the
    # ciphertext level of the comparison results.
    for carry_index in round_plan["carry"]:
        carry = []

        for ct in current[carry_index]:
            if ct.metadata.level < target_level:
                raise ValueError("Carry ciphertext level too low")

            aligned = ct
            if ct.metadata.level > target_level:
                aligned = drop_level(
                    ct, ct.metadata.level - target_level
                )

            carry.append(add(aligned, aligned))

        next_values.append(carry)

    return next_values
