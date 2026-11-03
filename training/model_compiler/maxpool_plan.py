"""Shared MaxPool2D planning utilities."""

def build_maxpool_plan(layer, input_shape):
    kernel = tuple(map(int, layer["kernel_shape"]))
    stride = tuple(map(int, layer["stride"]))
    padding = tuple(map(int, layer["padding"]))
    shape = tuple(map(int, input_shape))

    if any(len(v) != 2 for v in (kernel, stride, padding, shape)):
        raise ValueError("MaxPool2D expects 2D parameters")

    if any(v <= 0 for v in kernel + stride + shape):
        raise ValueError("Kernel, stride and shape must be positive")

    if any(v < 0 for v in padding):
        raise ValueError("Padding must be nonnegative")

    if any(padding[i] > kernel[i] // 2 for i in range(2)):
        raise NotImplementedError("Unsupported MaxPool padding")

    if int(layer.get("ceil_mode", 0)) != 0:
        raise NotImplementedError("ceil_mode=True is not supported yet")

    dilation = layer.get("dilations", [1, 1])
    if list(dilation) != [1, 1]:
        raise NotImplementedError("Dilated MaxPool is not supported yet")

    output_shape = tuple(
        (shape[i] + 2 * padding[i] - kernel[i]) // stride[i] + 1
        for i in range(2)
    )

    if any(v <= 0 for v in output_shape):
        raise ValueError("Invalid MaxPool output shape")

    # Build a balanced binary Max reduction plan.
    count = kernel[0] * kernel[1]
    rounds = []

    while count > 1:
        pairs = [
            (i, i + 1)
            for i in range(0, count - 1, 2)
        ]

        carry = [count - 1] if count % 2 else []

        rounds.append({
            "pairs": pairs,
            "carry": carry,
        })

        count = len(pairs) + len(carry)

    return {
        "kernel_shape": kernel,
        "stride": stride,
        "padding": padding,
        "input_shape": shape,
        "output_shape": output_shape,
        "binary_max_count": kernel[0] * kernel[1] - 1,
        "compare_depth": len(rounds),
        "rounds": rounds,
    }


def build_maxpool_access_plan(plan, skip=(1, 1)):
    """
    Generate neighborhood offsets and valid output regions.

    This is a logical slot-access plan for a contiguous
    feature layout. Big-size / cross-ciphertext access
    requires separate handling.
    """
    h, w = plan["input_shape"]
    oh, ow = plan["output_shape"]
    kh, kw = plan["kernel_shape"]
    sh, sw = plan["stride"]
    ph, pw = plan["padding"]

    if len(skip) != 2 or any(v <= 0 for v in skip):
        raise ValueError("Invalid skip")

    # Output anchors are at (out_row*sh, out_col*sw).
    # Require anchors to lie inside the source feature.
    if (oh - 1) * sh >= h or (ow - 1) * sw >= w:
        raise NotImplementedError(
            "Output anchors outside input require "
            "a different layout strategy"
        )

    accesses = []

    for kr in range(kh):
        for kc in range(kw):
            dr = kr - ph
            dc = kc - pw

            row_lo = max(0, -(dr // sh))
            row_hi = min(oh - 1, (h - 1 - dr) // sh)

            col_lo = max(0, -(dc // sw))
            col_hi = min(ow - 1, (w - 1 - dc) // sw)

            rotation_step = (
                dr * w * skip[0] * skip[1]
                + dc * skip[1]
            )

            accesses.append({
                "kernel_position": (kr, kc),
                "relative_offset": (dr, dc),
                "rotation_step": rotation_step,
                "valid_region": (
                    row_lo, row_hi, col_lo, col_hi
                ),
            })

    return accesses



def execute_maxpool_reduction(plan, inputs, binary_max, carry_transform=None):
    """
    Execute a planned balanced binary Max reduction.

    inputs:
        One element per kernel position.

    binary_max:
        Callable(left, right, round_index, pair_index).

    The callback can be implemented by an exact plaintext
    maximum or, later, a CKKS polynomial Max DAG builder.
    """
    expected = (
        plan["kernel_shape"][0]
        * plan["kernel_shape"][1]
    )

    if len(inputs) != expected:
        raise ValueError(
            f"Expected {expected} kernel inputs, "
            f"got {len(inputs)}"
        )

    current = list(inputs)

    for round_index, round_plan in enumerate(plan["rounds"]):
        next_values = []

        for pair_index, (i, j) in enumerate(round_plan["pairs"]):
            next_values.append(
                binary_max(
                    current[i],
                    current[j],
                    round_index,
                    pair_index,
                )
            )

        for i in round_plan["carry"]:
            value = current[i]
            if carry_transform is not None:
                value = carry_transform(value, round_index)
            next_values.append(value)

        current = next_values

    assert len(current) == 1
    return current[0]



def build_deferred_half_ckks_schedule(
    plan,
    input_level,
    running_max,
    upper_bound=1.0,
    k_first=1.0,
    k_later=1.05,
    extra_pre_levels=0,
):
    """
    Plan degree-7/degree-7 deferred-half MaxPool levels.

    Assumptions:
      - normalization: 1 level
      - each binary Max: 7 levels
      - output repacking: 1 level
      - output restoration: 1 level

    Does not generate CKKS masks or plaintext weights.
    """
    import math

    if not isinstance(input_level, int) or input_level < 0:
        raise ValueError("Invalid input level")

    if not isinstance(extra_pre_levels, int) or extra_pre_levels < 0:
        raise ValueError("Invalid extra preprocessing levels")

    if not math.isfinite(running_max) or running_max <= 0:
        raise ValueError("Invalid running_max")

    if not math.isfinite(upper_bound) or upper_bound <= 0:
        raise ValueError("Invalid upper_bound")

    if not math.isfinite(k_first) or k_first <= 0:
        raise ValueError("Invalid k_first")

    if not math.isfinite(k_later) or k_later <= 0:
        raise ValueError("Invalid k_later")

    depth = plan["compare_depth"]
    binary_cost = 7

    total_cost = (
        1 + extra_pre_levels
        + depth * binary_cost
        + 1 + 1
    )

    # Reserve at least one output level for downstream work.
    if input_level - total_cost < 1:
        raise ValueError(
            f"Insufficient CKKS levels: "
            f"input={input_level}, required={total_cost}, "
            f"minimum_output_level=1"
        )

    scale = running_max / upper_bound
    level = input_level - 1 - extra_pre_levels
    rounds = []

    for r, round_plan in enumerate(plan["rounds"]):
        base_B = k_first if r == 0 else k_later
        effective_B = base_B * (2 ** r)

        rounds.append({
            "round_index": r,
            "comparison_count": len(round_plan["pairs"]),
            "carry_count": len(round_plan["carry"]),
            "effective_B": effective_B,
            "poly0_input_level": level,
            "poly1_input_level": level - 3,
            "binary_max_output_level": level - binary_cost,
        })

        level -= binary_cost

    repack_input_level = level
    restore_input_level = level - 1
    final_level = level - 2

    return {
        "rounds": rounds,
        "normalization": scale,
        "input_normalization_multiplier": 1.0 / scale,
        "output_restore_multiplier": scale / (2 ** depth),
        "repack_input_level": repack_input_level,
        "restore_input_level": restore_input_level,
        "final_level": final_level,
        "total_level_cost": total_cost,
    }


def build_maxpool_stage_plan(
    plan,
    *,
    normalize_cost=1,
    compare_cost=7,
    repack_cost=1,
    restore_cost=1,
    polynomial_costs=None,
):
    """Build executable logical MaxPool stages from a reduction plan.

    The stage boundaries are derived from the reduction tree and operation
    costs.  No Bootstrap limit is assumed here; the compiler/DP decides where
    to refresh the resulting stages later.
    """
    costs = {
        "normalize_taps": int(normalize_cost),
        "repack": int(repack_cost),
        "restore": int(restore_cost),
    }
    if any(value < 0 for value in costs.values()):
        raise ValueError("MaxPool stage costs must be nonnegative")
    compare_cost = int(compare_cost)
    if compare_cost < 0:
        raise ValueError("compare_cost must be nonnegative")
    if polynomial_costs is not None:
        if (not polynomial_costs or any(isinstance(c, bool) or not isinstance(c, int) or c <= 0
                                        for c in polynomial_costs)
                or sum(polynomial_costs) + 1 != compare_cost):
            raise ValueError('Invalid MaxPool split polynomial costs')

    stages = [
        {
            "name": "normalize_taps",
            "operations": ("normalize_taps",),
            "level_cost": costs["normalize_taps"],
            "live_tensors": plan["kernel_shape"][0] * plan["kernel_shape"][1],
        }
    ]
    for index, round_plan in enumerate(plan["rounds"], start=1):
        if polynomial_costs is not None:
            pairs, carries = len(round_plan['pairs']), len(round_plan['carry'])
            for poly_index, cost in enumerate(polynomial_costs):
                stages.append({'name': f'reduce_round_{index}_poly_{poly_index}',
                               'level_cost': cost, 'live_tensors': 3 * pairs + carries})
            stages.append({'name': f'reduce_round_{index}_finish',
                           'level_cost': 1, 'live_tensors': pairs + carries})
            continue
        stages.append({
            "name": f"reduce_round_{index}",
            "operations": (f"reduce_round_{index}",),
            "level_cost": compare_cost,
            "live_tensors": len(round_plan["pairs"]) + len(round_plan["carry"]),
        })
    stages.extend([
        {"name": "repack", "operations": ("repack",),
         "level_cost": costs["repack"], "live_tensors": 1},
        {"name": "restore", "operations": ("restore",),
         "level_cost": costs["restore"], "live_tensors": 1},
    ])
    return {
        "stages": tuple(stages),
        "total_level_cost": sum(stage["level_cost"] for stage in stages),
        "input_shape": plan["input_shape"],
        "output_shape": plan["output_shape"],
    }


def parse_maxpool_split_stage(stage):
    """Return (round index, polynomial index or 'finish'), or None."""
    import re
    match = re.fullmatch(r'reduce_round_([1-9][0-9]*)_(?:poly_([0-9]+)|(finish))', stage)
    if match is None:
        return None
    return int(match[1]) - 1, 'finish' if match[3] else int(match[2])
