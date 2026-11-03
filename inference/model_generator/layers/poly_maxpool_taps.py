"""CKKS MaxPool neighborhood access and fused normalization."""

import math
import numpy as np

from inference.lattisense.frontend.custom_task import (
    add, mult, rescale, rotate_cols,
)


def build_blocked_maxpool_access_plan(
    plan, block_shape, block_expansion=None,
):
    """Describe 3x3 tap sources for interleaved spatial blocks.

    The ordering follows ``Avgpool2DLayer.call_interleaved_avgpool``:
    ciphertexts are grouped by channel, then block-row, then block-column.
    Each output anchor records the source block and the rotation inside that
    block.  This is a layout planner only; it does not build FHE nodes.
    """
    h, w = map(int, plan['input_shape'])
    oh, ow = map(int, plan['output_shape'])
    sh, sw = map(int, plan['stride'])
    ph, pw = map(int, plan['padding'])
    bh, bw = map(int, block_shape)
    if min(h, w, oh, ow, bh, bw) <= 0 or (h % bh) or (w % bw):
        raise ValueError('Input shape must be divisible by block_shape')
    expected_expansion = [h // bh, w // bw]
    if block_expansion is not None and list(block_expansion) != expected_expansion:
        raise ValueError(
            f'block_expansion={block_expansion} does not match {expected_expansion}'
        )

    result = []
    for kr in range(int(plan['kernel_shape'][0])):
        for kc in range(int(plan['kernel_shape'][1])):
            dr, dc = kr - ph, kc - pw
            entries = []
            for out_r in range(oh):
                for out_c in range(ow):
                    anchor_r, anchor_c = out_r * sh, out_c * sw
                    src_r, src_c = anchor_r + dr, anchor_c + dc
                    valid = 0 <= src_r < h and 0 <= src_c < w
                    # Padding duplicates the legal anchor, matching the
                    # existing polynomial MaxPool (not zero padding).
                    if not valid:
                        src_r, src_c = anchor_r, anchor_c
                    local_r, block_r = divmod(src_r, expected_expansion[0])
                    local_c, block_c = divmod(src_c, expected_expansion[1])
                    entries.append({
                        'valid': valid,
                        'output': (out_r, out_c),
                        'block': (block_r, block_c),
                        'block_index': block_r * expected_expansion[1] + block_c,
                        'rotation': local_r * bw + local_c - (out_r * ow + out_c),
                    })
            result.append({
                'kernel_position': (kr, kc),
                'relative_offset': (dr, dc),
                'entries': entries,
            })
    return result


def build_interleaved_maxpool_tap_masks(plan, block_shape, channels, slots, inv_scale):
    """Build source/rotation masks for the Retina interleaved input layout.

    Outputs use one contiguous output plane per channel. Each branch is a
    rotate, plaintext multiply and rescale; branches are added at equal level.
    """
    if channels <= 0 or slots < math.prod(block_shape):
        raise ValueError('Invalid channel count or slot capacity')
    if not math.isfinite(inv_scale) or inv_scale <= 0:
        raise ValueError('Invalid normalization factor')
    expansion = [plan['input_shape'][i] // block_shape[i] for i in range(2)]
    if list(plan['stride']) != expansion or list(plan['output_shape']) != list(block_shape):
        raise NotImplementedError('Requires output_shape=block_shape and stride=block_expansion')
    taps = build_blocked_maxpool_access_plan(plan, block_shape, expansion)
    result = []
    for tap in taps:
        groups = {}
        for entry in tap['entries']:
            key = (entry['block_index'], entry['rotation'])
            mask = groups.setdefault(key, np.zeros(slots, dtype=np.float64))
            r, c = entry['output']
            mask[r * block_shape[1] + c] = inv_scale
        result.append([
            {'source_block': block, 'rotation_step': step, 'mask': mask}
            for (block, step), mask in sorted(groups.items())
        ])
    return result


def call_interleaved_normalized_maxpool_taps(x_cts, tap_specs, pt_nodes, channels, blocks_per_channel=4):
    """Return [tap][channel] DAG nodes using explicitly bound branch masks."""
    if len(x_cts) != channels * blocks_per_channel or len(pt_nodes) != len(tap_specs):
        raise ValueError('Interleaved MaxPool input or tap count mismatch')
    if len({ct.level for ct in x_cts}) != 1:
        raise ValueError('Interleaved MaxPool input levels differ')
    result = []
    for specs, weights in zip(tap_specs, pt_nodes):
        if len(weights) != channels or any(len(row) != len(specs) for row in weights):
            raise ValueError('Interleaved MaxPool mask count mismatch')
        row = []
        for channel in range(channels):
            value = None
            for spec, pt in zip(specs, weights[channel]):
                block = spec['source_block']
                if not 0 <= block < blocks_per_channel:
                    raise ValueError('Invalid source block')
                source = x_cts[channel * blocks_per_channel + block]
                step = spec['rotation_step']
                rotated = source if step == 0 else rotate_cols(source, [step])[0]
                branch = rescale(mult(rotated, pt))
                value = branch if value is None else add(value, branch)
            if value is None:
                raise ValueError('Empty tap branches')
            row.append(value)
        result.append(row)
    return result


def build_multiplexed_maxpool_tap_masks(
    plan, accesses, channels, slots, inv_scale
):
    """
    Build plaintext slot masks for MaxPool.

    Current supported layout:
      - multiplexed channel-contiguous packing
      - skip=(1,1)
      - each complete channel fits within one ciphertext

    The masks include the normalization factor inv_scale.
    """

    h, w = map(int, plan["input_shape"])
    oh, ow = map(int, plan["output_shape"])
    sh, sw = map(int, plan["stride"])
    kh, kw = map(int, plan["kernel_shape"])

    if any(v <= 0 for v in (
        h, w, oh, ow, sh, sw, kh, kw, channels, slots
    )):
        raise ValueError("Invalid MaxPool dimensions")

    if h & (h - 1) or w & (w - 1):
        raise NotImplementedError(
            "CKKS packing requires power-of-two input dimensions"
        )

    if slots % (h * w):
        raise NotImplementedError(
            "A complete channel must fit within one ciphertext"
        )

    if not math.isfinite(inv_scale) or inv_scale <= 0:
        raise ValueError("Invalid normalization factor")

    if len(accesses) != kh * kw:
        raise ValueError("Incorrect number of neighborhood accesses")

    if (oh - 1) * sh >= h or (ow - 1) * sw >= w:
        raise NotImplementedError("Output anchor outside input")

    block = h * w
    channels_per_ct = slots // block
    n_ct = (channels + channels_per_ct - 1) // channels_per_ct

    anchors = np.array([
        r * sh * w + c * sw
        for r in range(oh)
        for c in range(ow)
    ], dtype=np.int64)

    result = []

    for access in accesses:
        dr, dc = map(int, access["relative_offset"])
        step = int(access["rotation_step"])

        # The current implementation is intentionally limited
        # to the verified contiguous skip=(1,1) layout.
        if step != dr * w + dc:
            raise NotImplementedError(
                "Only skip=(1,1) packing is supported"
            )

        valid = np.array([
            0 <= r * sh + dr < h
            and 0 <= c * sw + dc < w
            for r in range(oh)
            for c in range(ow)
        ], dtype=bool)

        per_ct = []

        for ct_idx in range(n_ct):
            active = min(
                channels_per_ct,
                channels - ct_idx * channels_per_ct
            )

            valid_mask = np.zeros(slots, dtype=np.float64)
            invalid_mask = np.zeros(slots, dtype=np.float64)

            for ch in range(active):
                indices = ch * block + anchors

                valid_mask[indices[valid]] = inv_scale
                invalid_mask[indices[~valid]] = inv_scale

            per_ct.append({
                "valid": valid_mask,
                "invalid": invalid_mask,
            })

        result.append({
            "rotation_step": step,
            "per_ct": per_ct,
        })

    return result


def call_normalized_maxpool_taps(
    x_cts, tap_specs, pt_nodes
):
    """
    Build CKKS DAG nodes for normalized neighborhood taps.

    pt_nodes[tap][ct] contains:
        valid:   plaintext node or None
        invalid: plaintext node or None

    Each nonempty plaintext must be encoded at its
    corresponding input ciphertext's level.

    Returns:
        [tap_index][ciphertext_index]
    """

    if not x_cts:
        raise ValueError("Empty ciphertext input")

    if len(tap_specs) != len(pt_nodes):
        raise ValueError("Tap and plaintext counts differ")

    result = []

    for tap, provided in zip(tap_specs, pt_nodes):
        if (
            len(tap["per_ct"]) != len(x_cts)
            or len(provided) != len(x_cts)
        ):
            raise ValueError("Ciphertext count mismatch")

        step = tap["rotation_step"]
        row = []

        for ct, mask, pts in zip(
            x_cts, tap["per_ct"], provided
        ):
            rotated = (
                ct if step == 0
                else rotate_cols(ct, [step])[0]
            )

            branches = []

            for key, source in (
                ("valid", rotated),
                ("invalid", ct),
            ):
                needed = bool(np.any(mask[key]))
                pt = pts.get(key)

                if needed and pt is None:
                    raise ValueError(
                        f"Missing {key} mask plaintext"
                    )

                if not needed and pt is not None:
                    raise ValueError(
                        f"Unexpected {key} mask plaintext"
                    )

                if needed:
                    branches.append(
                        rescale(mult(source, pt))
                    )

            if not branches:
                raise ValueError("No valid output anchors")

            output = (
                branches[0]
                if len(branches) == 1
                else add(branches[0], branches[1])
            )

            row.append(output)

        result.append(row)

    return result
