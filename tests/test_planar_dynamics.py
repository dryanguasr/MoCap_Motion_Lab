"""Analytic checks, not empirical validation against force plates."""
import unittest
from pathlib import Path
import sys
import numpy as np
from seima_mocap.planar_dynamics import (
    G, cross2, local_polynomial, whole_body_wrench, double_support, segment_proximal_wrench,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from estimate_planar_kinetics import CLIPS, Assumptions, estimate, make_segments


class PlanarDynamicsTests(unittest.TestCase):
    def test_static_and_vertical_acceleration(self):
        for az in (0.0, 2.0):
            center, acc, force, hdot = whole_body_wrench(
                np.array([45., 45.]), np.array([[-.1, 1.], [.1, 1.]]),
                np.array([[0., az], [0., az]]), np.zeros(2), np.zeros(2))
            np.testing.assert_allclose(force, [0., 90 * (G + az)])
            self.assertAlmostEqual(float(hdot), 0.)
            fl, fr, fraction, valid, residual, lever = double_support(
                center, force, hdot, np.array([-.2, 0.]), np.array([.2, 0.]))
            self.assertTrue(valid)
            self.assertAlmostEqual(float(fraction), .5)
            np.testing.assert_allclose(fl + fr, force)
            self.assertAlmostEqual(float(residual), 0.)

    def test_unidentifiable_and_negative_loads_rejected(self):
        for left, right, com in (([0, 0], [0, 0], [0, 1]), ([-.2, 0], [.2, 0], [2, 1])):
            _, _, _, valid, _, _ = double_support(
                np.array(com), np.array([0., 900.]), np.array(0.), np.array(left), np.array(right))
            self.assertFalse(valid)

    def test_static_foot_moment_sign_and_balance(self):
        force, moment = segment_proximal_wrench(
            1., np.array([.05, 0.]), np.zeros(2), 0., 0.,
            np.array([.1, 0.]), np.array([0., 450.]), 0., np.array([0., 0.]))
        np.testing.assert_allclose(force, [0, G - 450])
        self.assertAlmostEqual(float(moment), .05 * G - .1 * 450)

    def test_dynamic_segment_newton_euler_closure(self):
        mass, inertia, alpha = 4., .3, 2.1
        com, acc = np.array([.3, .6]), np.array([1.2, -.7])
        dp, df, dm, pp = np.array([.5, .1]), np.array([100., 400.]), 8., np.array([.2, 1.])
        pf, pm = segment_proximal_wrench(mass, com, acc, inertia, alpha, dp, df, dm, pp)
        np.testing.assert_allclose(pf + df + [0, -mass * G], mass * acc)
        self.assertAlmostEqual(float(pm + dm + cross2(dp - com, df) + cross2(pp - com, pf)), inertia * alpha)

    def test_irregular_timestamps_and_no_gap_interpolation(self):
        t = np.arange(61) / 30 + .001 * np.sin(np.arange(61))
        values = (3 * t**2 + 2 * t + 1)[:, None]
        valid = np.ones(len(t), dtype=bool)
        pos, vel, acc, accepted = local_polynomial(t, values, valid)
        np.testing.assert_allclose(acc[accepted], 6., atol=1e-8)
        np.testing.assert_allclose(vel[accepted, 0], 6 * t[accepted] + 2, atol=1e-8)
        valid[30] = False
        pos, vel, acc, accepted = local_polynomial(t, values, valid)
        self.assertFalse(np.any(accepted[24:37]))
        self.assertTrue(np.all(np.isnan(acc[24:37])))
        self.assertFalse(accepted[0])

    def test_full_static_pipeline_mass_conservation_and_scope(self):
        points = np.zeros((100, 33, 2))
        locations = {0: (0, 1.75), 11: (-.2, 1.5), 12: (.2, 1.5),
                     13: (-.3, 1.25), 14: (.3, 1.25), 15: (-.4, 1.), 16: (.4, 1.),
                     23: (-.15, .95), 24: (.15, .95), 25: (-.18, .5), 26: (.18, .5),
                     27: (-.2, .1), 28: (.2, .1), 29: (-.25, .02), 30: (.15, .02),
                     31: (-.1, .02), 32: (.3, .02)}
        for index, xy in locations.items():
            points[:, index] = xy
        flags = {"raw_pose_valid": np.ones(100, dtype=bool), "near_heuristic_stroke": np.zeros(100, dtype=bool)}
        t = np.arange(100) / 30
        spec = Assumptions()
        _, masses, _, _, _ = make_segments(points, spec)
        self.assertAlmostEqual(float(masses.sum()), 90.18)
        result, _ = estimate(points, t, 1., flags, spec)
        accepted = result["torque_valid"]
        self.assertEqual(int(accepted.sum()), 88)
        np.testing.assert_allclose(result["total_Fz_N"][accepted], 90.18 * G, atol=1e-7)
        np.testing.assert_allclose(result["left_Fz_N"][accepted] + result["right_Fz_N"][accepted], 90.18 * G, atol=1e-7)
        self.assertTrue(all(name.endswith("_1") for name in CLIPS))
        self.assertEqual(len(CLIPS), 6)
        self.assertNotIn("20251212_132025", CLIPS)


if __name__ == "__main__":
    unittest.main()
