
"""Range normalization for CKKS polynomial MaxPooling."""

from dataclasses import dataclass
import math

from inference.model_generator.layers.mult_scaler import MultScalarLayer




@dataclass(frozen=True)
class PolyMaxPoolConfig:
    """
    Parameters for polynomial MaxPooling.

    upper_bound: target normalized input range
    alpha:       Sign polynomial precision profile
    k1:          first-round B multiplier
    k2:          second-round B multiplier
    """

    upper_bound: float = 1.0
    alpha: int = 7
    k1: float = 1.0
    k2: float = 1.05

    def __post_init__(self):
        for name in ("upper_bound", "k1", "k2"):
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be positive and finite"
                )

        if self.k1 < 1.0:
            raise ValueError("k1 must be >= 1")

        # Only alpha=7 is implemented and validated.
        if self.alpha != 7:
            raise NotImplementedError(
                "Only alpha=7 is supported in v1"
            )

    @property
    def B_r1(self):
        return self.k1 * self.upper_bound

    @property
    def B_r2(self):
        return self.k2 * self.upper_bound

    def scale_from_calibration(self, running_max):
        """
        Compute a fixed scaling factor using plaintext
        calibration statistics.
        """
        if not math.isfinite(running_max) or running_max <= 0:
            raise ValueError(
                "running_max must be positive and finite"
            )

        return running_max / self.upper_bound


def scaling_plan(running_max, config=None):
    """
    Generate the constants needed by CKKS MaxPooling.

    running_max must be obtained during plaintext calibration.
    """

    if config is None:
        config = PolyMaxPoolConfig()

    scale = config.scale_from_calibration(running_max)

    return {
        "scale": scale,
        "inv_scale": 1.0 / scale,

        "B_r1": config.B_r1,
        "B_r2": config.B_r2,

        # Effective B values in original input coordinates.
        "reference_B_r1": scale * config.B_r1,
        "reference_B_r2": scale * config.B_r2,

        "alpha": config.alpha,
        "degree_p0": 7,
        "degree_p1": 7,

        # Current MultScalarLayer uses mult + rescale.
        "extra_ckks_levels": 2,
    }


def apply_range_normalization(input_ct, input_scale_pt):
    """
    CKKS input normalization:

        Enc(x) -> Enc(x / scale)

    input_scale_pt contains the encoded public constant
    1 / scale.

    Consumes one CKKS level with current MultScalarLayer.
    """

    if len(input_ct) != len(input_scale_pt):
        raise ValueError(
            "Input ciphertext and plaintext counts differ"
        )

    return MultScalarLayer().call(
        input_ct,
        input_scale_pt
    )


def restore_range_after_repack(output_ct, output_scale_pt):
    """
    CKKS output scale restoration:

        Enc(y / scale) -> Enc(y)

    output_scale_pt contains the encoded public constant
    scale.

    Called after channel repacking.
    """

    if len(output_ct) != len(output_scale_pt):
        raise ValueError(
            "Output ciphertext and plaintext counts differ"
        )

    return MultScalarLayer().call(
        output_ct,
        output_scale_pt
    )
