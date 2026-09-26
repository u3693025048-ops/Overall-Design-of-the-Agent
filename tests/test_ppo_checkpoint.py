#!/usr/bin/env python3
"""Unit tests for PPO checkpoint save/restore."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from multi_agent.rl.matlab_rl_optimizer import MatlabRLOptimizer
from multi_agent.rl.ppo_checkpoint import (
    CHECKPOINT_VERSION,
    PPOCheckpointManager,
    checkpoint_stem,
)


class TestCheckpointStem(unittest.TestCase):
    def test_stem_format(self):
        self.assertEqual(checkpoint_stem("20260608_120000", 10), "ppo_20260608_120000_ep00010")


class TestResolveCheckpointPath(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_pair(self, run_id: str, episode: int) -> str:
        stem = checkpoint_stem(run_id, episode)
        json_path = os.path.join(self.tmp, stem + ".json")
        npz_path = os.path.join(self.tmp, stem + ".npz")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({"version": CHECKPOINT_VERSION, "npz_path": stem + ".npz"}, fh)
        np.savez_compressed(npz_path, actor_W1=np.zeros((1, 1)))
        latest = {
            "run_id": run_id,
            "stem": stem,
            "json_path": stem + ".json",
            "npz_path": stem + ".npz",
            "episode_completed": episode,
        }
        with open(os.path.join(self.tmp, "latest.json"), "w", encoding="utf-8") as fh:
            json.dump(latest, fh)
        return json_path

    def test_resolve_none(self):
        self.assertIsNone(PPOCheckpointManager.resolve_checkpoint_path(None))
        self.assertIsNone(PPOCheckpointManager.resolve_checkpoint_path(""))

    def test_resolve_json_file(self):
        path = self._write_pair("run1", 5)
        resolved = PPOCheckpointManager.resolve_checkpoint_path(path)
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(path))

    def test_resolve_directory_via_latest(self):
        self._write_pair("run1", 5)
        resolved = PPOCheckpointManager.resolve_checkpoint_path(self.tmp)
        self.assertTrue(resolved.endswith(".json"))
        self.assertTrue(os.path.isfile(resolved))

    def test_resolve_stem_without_extension(self):
        stem_path = os.path.join(self.tmp, checkpoint_stem("run1", 5))
        self._write_pair("run1", 5)
        resolved = PPOCheckpointManager.resolve_checkpoint_path(stem_path)
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(stem_path + ".json"))

    def test_resolve_latest_json_file(self):
        json_path = self._write_pair("run1", 5)
        latest_path = os.path.join(self.tmp, "latest.json")
        resolved = PPOCheckpointManager.resolve_checkpoint_path(latest_path)
        self.assertEqual(os.path.abspath(resolved), os.path.abspath(json_path))

    def test_resolve_missing_returns_none(self):
        self.assertIsNone(
            PPOCheckpointManager.resolve_checkpoint_path(os.path.join(self.tmp, "nope"))
        )


class TestCheckpointSaveLoad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.opt = MatlabRLOptimizer(
            checkpoint_dir=self.tmp,
            checkpoint_enabled=True,
            checkpoint_save_every=1,
            checkpoint_keep_last_n=2,
        )
        self.opt._init_networks()
        self.opt._best_reward = 1.25
        self.opt._best_params = {"w1": 42.0}
        self.opt._best_metrics = {"hit_rate": 88.0, "SEP": 5.5}
        self.opt._best_miss_ever = 5.5
        self.opt._best_peak_n_ever = 18.0
        self.opt._current_params = dict(self.opt._base_auto)
        self.opt._last_metrics = {"hit_rate": 80.0}

    def test_save_creates_json_npz_and_latest(self):
        mgr = PPOCheckpointManager(
            checkpoint_dir=self.tmp,
            save_every_episodes=1,
            keep_last_n=5,
            enabled=True,
        )
        mgr.set_run_id("testrun")
        saved = mgr.save(self.opt, episode=10, max_episodes=50, script_path="/tmp/a.m", nmc=10)
        self.assertIsNotNone(saved)
        self.assertTrue(os.path.isfile(saved))
        stem = checkpoint_stem("testrun", 10)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, stem + ".npz")))
        with open(os.path.join(self.tmp, "latest.json"), encoding="utf-8") as fh:
            latest = json.load(fh)
        self.assertEqual(latest["episode_completed"], 10)
        self.assertEqual(latest["stem"], stem)

    def test_load_restores_weights_and_meta(self):
        mgr = PPOCheckpointManager(checkpoint_dir=self.tmp, enabled=True)
        mgr.set_run_id("testrun")
        self.opt._actor.W1.fill(3.14)
        self.opt._critic.W1.fill(2.71)
        mgr.save(self.opt, episode=5, max_episodes=50)

        fresh = MatlabRLOptimizer(checkpoint_dir=self.tmp)
        fresh._init_networks()
        fresh._actor.W1.fill(0.0)
        fresh._critic.W1.fill(0.0)

        meta = mgr.load(fresh, self.tmp)
        self.assertEqual(meta["episode_completed"], 5)
        self.assertAlmostEqual(fresh._best_reward, 1.25)
        self.assertAlmostEqual(fresh._best_params["w1"], 42.0)
        self.assertAlmostEqual(float(fresh._actor.W1.flat[0]), 3.14, places=4)
        self.assertAlmostEqual(float(fresh._critic.W1.flat[0]), 2.71, places=4)

    def test_prune_keeps_last_n(self):
        mgr = PPOCheckpointManager(
            checkpoint_dir=self.tmp,
            save_every_episodes=1,
            keep_last_n=2,
            enabled=True,
        )
        mgr.set_run_id("prune")
        for ep in (1, 2, 3):
            mgr.save(self.opt, episode=ep, max_episodes=50)
        remaining = [
            f for f in os.listdir(self.tmp)
            if f.startswith("ppo_prune_ep") and f.endswith(".json")
        ]
        self.assertEqual(len(remaining), 2)
        self.assertTrue(any("ep00002" in f for f in remaining))
        self.assertTrue(any("ep00003" in f for f in remaining))

    def test_save_disabled_returns_none(self):
        mgr = PPOCheckpointManager(checkpoint_dir=self.tmp, enabled=False)
        self.assertIsNone(mgr.save(self.opt, episode=1, max_episodes=10))

    def test_load_missing_raises(self):
        mgr = PPOCheckpointManager(checkpoint_dir=self.tmp)
        with self.assertRaises(FileNotFoundError):
            mgr.load(self.opt, os.path.join(self.tmp, "missing.json"))

    def test_maybe_save_checkpoint_accepts_nmc_kwarg(self):
        """Regression: callers pass nmc=; must not raise TypeError."""
        self.opt._init_training_loggers()
        self.opt._maybe_save_checkpoint(
            ep=0,
            max_ep=50,
            script_path="/tmp/a.m",
            nmc=10,
            force=True,
        )
        latest = os.path.join(self.tmp, "latest.json")
        self.assertTrue(os.path.isfile(latest))


if __name__ == "__main__":
    unittest.main()
