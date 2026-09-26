#!/usr/bin/env python3
"""Tests for MATLAB timeout scaling."""

import unittest

from multi_agent.simulation.sim_timeout import get_matlab_timeout_sec


class TestSimTimeout(unittest.TestCase):
    def test_nmc_scales_timeout(self):
        t1 = get_matlab_timeout_sec(1)
        t25 = get_matlab_timeout_sec(25)
        self.assertGreater(t25, t1)
        self.assertGreaterEqual(t1, 120)

    def test_zero_nmc_uses_ceiling(self):
        t = get_matlab_timeout_sec(None)
        self.assertGreaterEqual(t, 120)


if __name__ == "__main__":
    unittest.main()
