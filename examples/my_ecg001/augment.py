import numpy as np


class ECGAugment:
    def __init__(
        self,
        enable=True,
        noise_std=0.01,
        scale_range=(0.95, 1.05),
        shift_range=(-0.05, 0.05),
        clip_range=(-4.0, 4.0),
        p_noise=0.5,
        p_scale=0.5,
        p_shift=0.5,
        p_clip=1.0,
        seed=42
    ):
        self.enable = enable
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.clip_range = clip_range
        self.p_noise = p_noise
        self.p_scale = p_scale
        self.p_shift = p_shift
        self.p_clip = p_clip
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray):
        if not self.enable:
            return x.astype(np.float32)

        out = x.copy().astype(np.float32)

        if self.rng.random() < self.p_scale:
            scale = self.rng.uniform(self.scale_range[0], self.scale_range[1])
            out = out * scale

        if self.rng.random() < self.p_shift:
            shift = self.rng.uniform(self.shift_range[0], self.shift_range[1])
            out = out + shift

        if self.rng.random() < self.p_noise:
            noise = self.rng.normal(0.0, self.noise_std, size=out.shape).astype(np.float32)
            out = out + noise

        if self.rng.random() < self.p_clip:
            out = np.clip(out, self.clip_range[0], self.clip_range[1])

        return out.astype(np.float32)