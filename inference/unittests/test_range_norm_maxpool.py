
import unittest

from inference.model_generator.layers.range_norm_maxpool import (
    PolyMaxPoolConfig,
    scaling_plan,
)


class TestRangeNormMaxPool(unittest.TestCase):

    def test_default_config(self):
        config = PolyMaxPoolConfig()

        self.assertEqual(config.upper_bound, 1.0)
        self.assertEqual(config.alpha, 7)
        self.assertEqual(config.k1, 1.0)
        self.assertEqual(config.k2, 1.05)

        self.assertAlmostEqual(config.B_r1, 1.0)
        self.assertAlmostEqual(config.B_r2, 1.05)

    def test_scaling_plan(self):
        config = PolyMaxPoolConfig()

        plan = scaling_plan(
            running_max=10.0,
            config=config
        )

        self.assertAlmostEqual(plan["scale"], 10.0)
        self.assertAlmostEqual(plan["inv_scale"], 0.1)

        self.assertAlmostEqual(plan["B_r1"], 1.0)
        self.assertAlmostEqual(plan["B_r2"], 1.05)

        self.assertAlmostEqual(
            plan["reference_B_r1"], 10.0
        )
        self.assertAlmostEqual(
            plan["reference_B_r2"], 10.5
        )

    def test_custom_parameters(self):
        config = PolyMaxPoolConfig(
            upper_bound=0.8,
            alpha=7,
            k1=1.0,
            k2=1.1
        )

        self.assertAlmostEqual(config.B_r1, 0.8)
        self.assertAlmostEqual(config.B_r2, 0.88)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            PolyMaxPoolConfig(k1=0.9)

        with self.assertRaises(NotImplementedError):
            PolyMaxPoolConfig(alpha=8)


if __name__ == "__main__":
    unittest.main()
