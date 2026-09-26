#!/usr/bin/env python3
"""Unit tests for startup preflight checks."""

import unittest

from multi_agent.preflight.preflight_check import run_preflight_checks


class TestPreflightCheck(unittest.TestCase):
    def test_run_preflight_returns_report(self):
        report = run_preflight_checks()
        self.assertTrue(report.items)
        names = {i.name for i in report.items}
        self.assertIn("python_version", names)
        self.assertIn("parser", names)
        self.assertIn("kb_template", names)

    def test_parser_check_passes(self):
        report = run_preflight_checks()
        parser_items = [i for i in report.items if i.name == "parser"]
        self.assertEqual(len(parser_items), 1)
        self.assertEqual(parser_items[0].status, "ok")


if __name__ == "__main__":
    unittest.main()
