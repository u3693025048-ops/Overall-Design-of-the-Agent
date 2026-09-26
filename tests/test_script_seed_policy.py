#!/usr/bin/env python3
"""Tests for iterative MATLAB seed policy."""

import os
import tempfile
import unittest

from multi_agent.integration.script_seed_policy import (
    is_kb_seed_source,
    pick_cli_seed_path,
    script_content_rl_ready,
    should_apply_deterministic_apn_on_seed,
)

_MIN_RL_SCRIPT = """
function test_seed_policy()
clc; close all;
Nmc = 10;
RUN_CASE='T'; SUB_IDX=4;
%% RL_PARAMS_BEGIN
rl_w1 = 40;
rl_zeta1 = 0.75;
rl_tao1 = 0.20;
rl_w2 = 35;
rl_zeta2 = 0.75;
rl_tao2 = 0.15;
rl_w3 = 35;
rl_zeta3 = 0.75;
rl_N_pn = 4;
%% RL_PARAMS_END
fprintf('命中率(miss<10m): 96.0%%  SEP: 4.00m\\n');
fprintf('峰值法向过载: 19.0±1.0g\\n');
fprintf('俯仰PM: 60.0±1.0°  BW: 50.0±1.0 rad/s\\n');
end
"""


class TestScriptSeedPolicy(unittest.TestCase):
    def test_is_kb_seed_source(self):
        self.assertTrue(is_kb_seed_source("local:monte_carlo_single.m"))
        self.assertTrue(is_kb_seed_source("kb:monte_carlo_single.m"))
        self.assertFalse(is_kb_seed_source("file:guidance_T4_v8.m"))
        self.assertFalse(is_kb_seed_source("scripts:guidance_T4_v8.m"))

    def test_script_content_rl_ready(self):
        ok, _ = script_content_rl_ready(_MIN_RL_SCRIPT)
        self.assertTrue(ok)
        ok_bad, _ = script_content_rl_ready("function x()\nend\n")
        self.assertFalse(ok_bad)

    def test_pick_cli_seed_prefers_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = os.path.join(tmp, "gate_ok.m")
            layer = os.path.join(tmp, "layer3.m")
            with open(gate, "w", encoding="utf-8") as fh:
                fh.write(_MIN_RL_SCRIPT)
            with open(layer, "w", encoding="utf-8") as fh:
                fh.write(_MIN_RL_SCRIPT.replace("96.0", "94.0"))

            picked = pick_cli_seed_path(
                task_mode="MODIFY_LAW",
                gate_passed_script=gate,
                layer3_script=layer,
                seed_policy="iterative",
            )
            self.assertEqual(picked, gate)

    def test_pick_cli_seed_safe_kb_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "ok.m")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(_MIN_RL_SCRIPT)
            self.assertEqual(
                pick_cli_seed_path(
                    task_mode="MODIFY_LAW",
                    gate_passed_script=p,
                    seed_policy="safe_kb",
                ),
                "",
            )

    def test_deterministic_apn_kb_only(self):
        self.assertTrue(
            should_apply_deterministic_apn_on_seed(
                mission_conditions="RUN_CASE='T'; SUB_IDX=4;",
                task_prompt="T4",
                seed_source="local:monte_carlo_single.m",
                mode="MODIFY_LAW",
            )
        )
        self.assertFalse(
            should_apply_deterministic_apn_on_seed(
                mission_conditions="RUN_CASE='T'; SUB_IDX=4;",
                task_prompt="T4",
                seed_source="file:guidance_v8.m",
                mode="MODIFY_LAW",
            )
        )


if __name__ == "__main__":
    unittest.main()
