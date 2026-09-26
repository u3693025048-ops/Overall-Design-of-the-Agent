#!/usr/bin/env python3
"""Tests for physics bounds probe and judge hard-constraint guard."""

import unittest

from multi_agent.simulation.physics_bounds import (
    build_probe_script,
    extract_gf_function,
    parse_probe_stdout,
)
from multi_agent.tools.judge_requirements_tool import (
    _hard_constraint_failed,
    _rule_check,
)


_GF_SNIPPET = """
function [N1,N2,G1,G2,gc]=gf(Tx,Ty,Tz,XK,TVx,TVy,TVz,Np)
g0=9.8;
N1=100; N2=100;
N1=max(-20,min(20,N1));
N2=max(-20,min(20,N2));
end
"""

_UNCLAMPED_GF = """
function [N1,N2,G1,G2,gc]=gf(Tx,Ty,Tz,XK,TVx,TVy,TVz,Np)
N1=100; N2=50; gc=0; G1=1; G2=0;
end
"""


class TestPhysicsBounds(unittest.TestCase):
    def test_extract_gf(self):
        block = extract_gf_function(_GF_SNIPPET)
        self.assertIsNotNone(block)
        self.assertIn("function", block)

    def test_build_probe(self):
        src, err = build_probe_script(_GF_SNIPPET, ny_limit_g=20.0)
        self.assertIsNotNone(src)
        self.assertEqual(err, "")
        self.assertIn("physics_bounds_probe_main", src)

    def test_parse_probe_pass(self):
        ok, peak, msg = parse_probe_stdout("PHYS_BOUNDS peak_cmd_g=18.5000 pass=1\n")
        self.assertTrue(ok)
        self.assertAlmostEqual(peak, 18.5)

    def test_parse_probe_fail(self):
        ok, peak, msg = parse_probe_stdout("PHYS_BOUNDS peak_cmd_g=25.0000 pass=0\n")
        self.assertFalse(ok)
        self.assertGreater(peak, 20.0)

    def test_unclamped_still_builds_probe(self):
        src, err = build_probe_script(_UNCLAMPED_GF, ny_limit_g=20.0)
        self.assertIsNotNone(src)


class TestJudgeHardConstraint(unittest.TestCase):
    def test_peak_ny_uses_mean_metric(self):
        metrics = {"peak_ny": 21.0, "peak_ny_max": 25.0, "hit_rate": 100.0, "SEP": 5.0}
        reqs = {"peak_ny_max": 20.0, "hit_rate_min": 92.0, "sep_max": 7.0}
        ok, reasons = _rule_check(metrics, reqs)
        self.assertFalse(ok)
        self.assertTrue(_hard_constraint_failed(reasons))

    def test_peak_ny_max_ignored_when_mean_ok(self):
        metrics = {"peak_ny": 17.0, "peak_ny_max": 25.0, "hit_rate": 100.0, "SEP": 5.0}
        reqs = {"peak_ny_max": 20.0, "hit_rate_min": 92.0, "sep_max": 7.0}
        ok, reasons = _rule_check(metrics, reqs)
        self.assertTrue(ok)

    def test_hard_constraint_failed_detects_sep(self):
        reasons = ["SEP 8.00m > 要求 7.00m [NG]"]
        self.assertTrue(_hard_constraint_failed(reasons))


if __name__ == "__main__":
    unittest.main()
