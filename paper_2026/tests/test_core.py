from __future__ import annotations

import math
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from evaluate_external_validation_lowest_fraction import (  # noqa: E402
    CMBParameters,
    calibrate_sigma_f_prime,
    lowest_fraction_mean,
    solve_life,
)
from train_parameterized_ppinn import (  # noqa: E402
    ParameterRanges,
    ParameterizedLifePINN,
    sample_parameters,
    solve_bisection,
)
from unet import UNetPowerToFatigue  # noqa: E402


class CoreTests(unittest.TestCase):
    def test_unet_shape(self) -> None:
        model = UNetPowerToFatigue(
            in_channels=7, out_channels=2, base_ch=4, use_attention=False
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 7, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 2, 32, 32))

    def test_parameterized_bisection_residual(self) -> None:
        ranges = ParameterRanges()
        values = sample_parameters(32, ranges, seed=7)
        life = solve_bisection(values, ranges.b)
        epsilon_a, sigma_mean, epsilon_f, sigma_f, c, modulus = values.T
        modeled = epsilon_f * np.power(2.0 * life, c)
        modeled += ((sigma_f - sigma_mean) / modulus) * np.power(2.0 * life, ranges.b)
        relative = np.abs(epsilon_a - modeled) / epsilon_a
        self.assertTrue(np.all(np.isfinite(life)))
        self.assertLess(float(relative.max()), 1.0e-6)

    def test_parameterized_model_shape(self) -> None:
        model = ParameterizedLifePINN()
        self.assertEqual(tuple(model(torch.zeros(5, 6)).shape), (5, 1))

    def test_lowest_fraction_uses_ceil(self) -> None:
        values = np.arange(1.0, 1001.0).reshape(20, 50)
        result, count = lowest_fraction_mean(values, np.ones_like(values, bool), 0.0011)
        self.assertEqual(count, 2)
        self.assertEqual(result, 1.5)

    def test_external_validation_numbers(self) -> None:
        data_dir = ROOT / "sample_data" / "external_validation"
        base = np.load(data_dir / "no_vibration_fields.npz", allow_pickle=False)
        mask = base["valid_mask"].astype(bool)
        params = CMBParameters()
        sigma_f, achieved, count = calibrate_sigma_f_prime(
            base["target_fields"][0],
            base["target_fields"][1],
            mask,
            20000.0,
            0.001,
            params,
        )
        self.assertEqual(count, 13)
        self.assertAlmostEqual(achieved, 20000.0, places=5)
        self.assertAlmostEqual(sigma_f, 212.3853468, places=5)

        expected = {10: 17920.80, 30: 13366.01}
        for frequency, reference in expected.items():
            case = np.load(
                data_dir / f"vertical_{frequency}Hz_2mm_fields.npz",
                allow_pickle=False,
            )
            life = solve_life(
                case["pred_eps_total"],
                case["pred_sigma_mean"],
                case["valid_mask"],
                sigma_f,
                params,
            )
            value, _ = lowest_fraction_mean(life, case["valid_mask"], 0.001)
            self.assertTrue(math.isclose(value, reference, abs_tol=0.02))


if __name__ == "__main__":
    unittest.main()
