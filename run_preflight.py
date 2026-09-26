#!/usr/bin/env python3
"""Standalone entry: run startup preflight checks without full cli_agent."""

from multi_agent.preflight import ensure_preflight_or_exit

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="GSMultiAgent preflight checks")
    p.add_argument("--strict", action="store_true", help="警告也视为失败")
    p.add_argument("--config", type=str, default=None, help="config.yaml 路径")
    args = p.parse_args()
    ensure_preflight_or_exit(
        config_path=args.config,
        strict=args.strict,
        preflight_only=True,
    )
