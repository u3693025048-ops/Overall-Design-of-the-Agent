"""
Startup preflight checks for GSMultiAgent13.

Validates environment, config, simulation engine, KB template, parser, and
common failure modes discussed in production logs before Layer 1~3 run.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

_STATUS_OK = "ok"
_STATUS_WARN = "warn"
_STATUS_FAIL = "fail"


@dataclass
class CheckItem:
    name: str
    status: str
    message: str
    hint: str = ""


@dataclass
class PreflightReport:
    items: List[CheckItem] = field(default_factory=list)
    project_root: str = "."

    @property
    def ok(self) -> bool:
        return not any(i.status == _STATUS_FAIL for i in self.items)

    @property
    def has_warnings(self) -> bool:
        return any(i.status == _STATUS_WARN for i in self.items)

    def add(self, name: str, status: str, message: str, hint: str = "") -> None:
        self.items.append(CheckItem(name=name, status=status, message=message, hint=hint))


def _add_ok(report: PreflightReport, name: str, message: str) -> None:
    report.add(name, _STATUS_OK, message)


def _add_warn(report: PreflightReport, name: str, message: str, hint: str = "") -> None:
    report.add(name, _STATUS_WARN, message, hint)


def _add_fail(report: PreflightReport, name: str, message: str, hint: str = "") -> None:
    report.add(name, _STATUS_FAIL, message, hint)


def _resolve_project_root(project_root: Optional[str]) -> Path:
    if project_root:
        return Path(project_root).resolve()
    # cli_agent.py lives at repo root; this module is multi_agent/preflight/
    return Path(__file__).resolve().parents[2]


def _check_python_version(report: PreflightReport) -> None:
    ver = sys.version_info
    text = f"Python {ver.major}.{ver.minor}.{ver.micro}"
    if ver >= (3, 13):
        _add_warn(
            report,
            "python_version",
            f"{text} — in-process matlab.engine 可能不可用，建议 simulation.engine=matlab",
            "config.yaml: engine: matlab",
        )
    else:
        _add_ok(report, "python_version", text)


def _check_imports(report: PreflightReport) -> None:
    required = ("numpy", "yaml", "torch")
    optional = ("openai", "langchain_openai", "chromadb")
    for mod in required:
        if importlib.util.find_spec(mod) is None:
            _add_fail(
                report,
                "imports",
                f"缺少必需依赖: {mod}",
                "pip install -r requirements.txt",
            )
            return
    missing_opt = [m for m in optional if importlib.util.find_spec(m) is None]
    if missing_opt:
        _add_warn(
            report,
            "imports_optional",
            f"可选依赖未安装: {', '.join(missing_opt)}（LLM/RAG 功能可能受限）",
        )
    else:
        _add_ok(report, "imports", "核心依赖 numpy / yaml / torch 可用")


def _check_config_files(report: PreflightReport, root: Path) -> Optional[str]:
    cfg = root / "config.yaml"
    if not cfg.is_file():
        _add_warn(
            report,
            "config_file",
            "未找到 config.yaml，将使用 config_loader 内置默认值",
            "建议复制 config.copy.yaml 并填入 API Key",
        )
        return None
    text = cfg.read_text(encoding="utf-8", errors="ignore")
    if "<<<<<<<" in text or ">>>>>>>" in text:
        _add_fail(
            report,
            "config_merge_conflict",
            "config.yaml 含未解决的 git 合并冲突标记",
            "手动编辑 config.yaml 删除 <<<<<<< / ======= / >>>>>>> 行",
        )
    else:
        _add_ok(report, "config_file", str(cfg))
    return str(cfg)


def _check_llm_config(report: PreflightReport) -> None:
    from multi_agent.config_loader import get_config

    cfg = get_config().llm
    key = (cfg.api_key or "").strip()
    env_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or ""
    ).strip()
    if not key and not env_key:
        _add_fail(
            report,
            "llm_api_key",
            "未配置 LLM API Key（config.yaml llm.api_key 与环境变量均为空）",
            "在 config.yaml 或 .env 中设置 api_key",
        )
        return
    model = cfg.model or "(default)"
    _add_ok(report, "llm_api_key", f"LLM 已配置 model={model}")


def _check_ablation_and_simulation(report: PreflightReport) -> None:
    from multi_agent.config_loader import get_config

    cfg = get_config()
    ab = cfg.ablation
    sim = cfg.simulation

    if not ab.optimization_workflow:
        _add_warn(
            report,
            "ablation_workflow",
            "ablation.optimization_workflow=false — Layer3 / Gate 不会运行",
        )
    elif not ab.rl_optimization:
        _add_warn(
            report,
            "ablation_rl",
            "ablation.rl_optimization=false — PPO 优化已关闭",
        )
    else:
        _add_ok(report, "ablation_workflow", "optimization_workflow / rl_optimization 已启用")

    if not sim.layer2_script_gate:
        _add_warn(
            report,
            "layer2_gate_config",
            "simulation.layer2_script_gate=false — 坏脚本可能直接进入 RL",
            "建议设为 true",
        )
    else:
        _add_ok(report, "layer2_gate_config", "layer2_script_gate=true")

    seed_policy = getattr(sim, "seed_policy", "iterative")
    if seed_policy == "iterative" and not sim.prefer_kb_template:
        _add_ok(
            report,
            "seed_policy",
            "seed_policy=iterative — Gate/Layer3 通过脚本可作为 MODIFY 种子",
        )
    elif not sim.prefer_kb_template:
        _add_warn(
            report,
            "prefer_kb_template",
            "prefer_kb_template=false 但 seed_policy 非 iterative",
        )
    elif sim.prefer_kb_template and seed_policy == "safe_kb":
        _add_ok(report, "seed_policy", "legacy safe_kb：每轮 KB 模板种子")


def _check_simulation_engine(report: PreflightReport) -> None:
    from multi_agent.config_loader import get_config
    from multi_agent.simulation.engine_resolver import (
        ENGINE_MATLAB,
        ENGINE_MATLAB_ENGINE,
        ENGINE_PYTHON,
        resolve_engine_config,
    )

    cfg = get_config().simulation
    resolved = resolve_engine_config(
        engine=cfg.engine,
        matlab_path=cfg.matlab_path,
        octave_path=cfg.octave_path,
    )
    eng = resolved.engine
    detail = (
        f"engine={eng}  matlab_exe={resolved.matlab_executable_available}  "
        f"matlab_engine={resolved.matlab_engine_available}  "
        f"octave={resolved.octave_executable_available}"
    )

    if eng == ENGINE_PYTHON:
        _add_warn(
            report,
            "simulation_engine",
            f"仿真引擎解析为 python fallback — {detail}",
            "安装/配置 MATLAB 或 Octave，并设置 simulation.engine=matlab",
        )
        return

    if eng in (ENGINE_MATLAB, ENGINE_MATLAB_ENGINE):
        if eng == ENGINE_MATLAB and not resolved.matlab_executable_available:
            _add_fail(
                report,
                "simulation_engine",
                f"engine=matlab 但未找到 MATLAB 可执行文件 ({cfg.matlab_path})",
                "config.yaml 中设置 matlab_path 为 matlab.exe 绝对路径",
            )
            return
        if eng == ENGINE_MATLAB_ENGINE and not resolved.matlab_engine_available:
            _add_warn(
                report,
                "simulation_engine",
                "engine=matlab_engine 但 matlab.engine 包不可用，运行时会降级 subprocess",
            )
            return

    _add_ok(report, "simulation_engine", detail)


def _check_kb_template(report: PreflightReport) -> None:
    from multi_agent.rl.matlab_rl_optimizer import validate_rl_script_content
    from multi_agent.tools.simulation_tool import RunSimulationTool

    kb_path = Path(RunSimulationTool.KB_TEMPLATE_PATH)
    if not kb_path.is_file():
        _add_fail(
            report,
            "kb_template",
            f"KB 模板不存在: {kb_path}",
            "确保 knowledge_base/matlab/guidance/monte_carlo_single.m 已打包",
        )
        return
    content = kb_path.read_text(encoding="utf-8", errors="ignore")
    ok, reason = validate_rl_script_content(content)
    if not ok:
        _add_fail(report, "kb_template", f"KB 模板校验失败: {reason}")
    else:
        _add_ok(report, "kb_template", f"monte_carlo_single.m 可用 ({len(content.splitlines())} 行)")


def _check_layer2_gate_code(report: PreflightReport) -> None:
    try:
        from multi_agent.integration import layer2_script_gate  # noqa: F401
    except ImportError as exc:
        _add_fail(
            report,
            "layer2_gate_code",
            f"无法导入 layer2_script_gate: {exc}",
            "同步 multi_agent/integration/layer2_script_gate.py",
        )
        return

    from multi_agent.tools.simulation_tool import GenerateMATLABTool

    import inspect

    sig = inspect.signature(GenerateMATLABTool.execute)
    if "force_kb_template" not in sig.parameters:
        _add_warn(
            report,
            "force_kb_template",
            "GenerateMATLABTool.execute 缺少 force_kb_template — Gate 烟雾失败后无法自动重生 KB 脚本",
            "同步 multi_agent/tools/simulation_tool.py",
        )
    else:
        _add_ok(report, "layer2_gate_code", "Layer2 Gate 模块与 force_kb_template 可用")


def _check_parser(report: PreflightReport) -> None:
    from multi_agent.rl.matlab_rl_optimizer import (
        metrics_look_like_parse_defaults,
        parse_sim_stdout,
    )

    good = """
  [  1] HIT    5.20    10.50    55.0    40.00     8.00
  命中率(miss<10m): 50.0%  SEP: 5.20m
"""
    bad = """
  Mean pitch PM: 30.5 deg, BW: 21.12 rad/s, GM: 6.09 dB
Case        Hit %   MeanMiss MeanNy  MeanPM MeanBW MeanGM
"""
    m_good = parse_sim_stdout(good)
    m_bad = parse_sim_stdout(bad)

    if m_good.get("hit_rate", 0) <= 0 or m_good.get("_parse_incomplete"):
        _add_fail(report, "parser", "解析器自检失败：标准 stdout 未解析出 hit_rate")
        return
    if not m_bad.get("_parse_incomplete") or not metrics_look_like_parse_defaults(m_bad, bad):
        _add_fail(
            report,
            "parser",
            "解析器自检失败：无 MC 证据的 stdout 应标记 _parse_incomplete",
        )
        return
    _add_ok(report, "parser", "parse_sim_stdout 自检通过（含占位值检测）")


def _check_guidance_scripts(report: PreflightReport, root: Path) -> None:
    from multi_agent.rl.matlab_rl_optimizer import validate_rl_script_content

    script_dirs = [
        root / "guidance_output" / "scripts",
        root / "guidance_output" / "matlab_scripts",
    ]
    bad: List[str] = []
    checked = 0
    for d in script_dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.m"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
            checked += 1
            content = path.read_text(encoding="utf-8", errors="ignore")
            ok, reason = validate_rl_script_content(content)
            if not ok:
                bad.append(f"{path.name}: {reason}")

    if not checked:
        _add_ok(report, "guidance_scripts", "无历史 guidance_output 脚本（Gate 将使用 KB 模板）")
        return
    if bad:
        preview = "; ".join(bad[:3])
        extra = f" 等共 {len(bad)} 个" if len(bad) > 3 else ""
        _add_warn(
            report,
            "guidance_scripts",
            f"发现 {len(bad)}/{checked} 个可能不可用的 .m 脚本: {preview}{extra}",
            "删除 guidance_output/scripts 下旧脚本，或开启 layer2_script_gate 自动重生",
        )
    else:
        _add_ok(report, "guidance_scripts", f"已扫描 {checked} 个历史脚本，RL 校验均通过")


def _check_hermes(report: PreflightReport) -> None:
    if importlib.util.find_spec("run_agent") is None:
        _add_warn(
            report,
            "hermes",
            "Hermes (run_agent) 不可用 — 将跳过 Layer2 任务规划（Layer3 RL 仍可运行）",
        )
    else:
        _add_ok(report, "hermes", "Hermes run_agent 可用")


def _check_knowledge_base(report: PreflightReport, root: Path) -> None:
    kb = root / "knowledge_base"
    if not kb.is_dir():
        _add_fail(report, "knowledge_base", f"缺少 knowledge_base 目录: {kb}")
        return
    md_count = len(list(kb.rglob("*.md")))
    m_count = len(list((kb / "matlab").rglob("*.m"))) if (kb / "matlab").is_dir() else 0
    _add_ok(report, "knowledge_base", f"knowledge_base 存在 ({md_count} md, {m_count} m)")


def _check_cli_agent_gate_hook(report: PreflightReport, root: Path) -> None:
    cli = root / "cli_agent.py"
    if not cli.is_file():
        _add_warn(report, "cli_agent", "未找到 cli_agent.py")
        return
    text = cli.read_text(encoding="utf-8", errors="ignore")
    if "ensure_script_ready_for_layer3" not in text:
        _add_warn(
            report,
            "cli_agent_gate",
            "cli_agent.py 未集成 Layer2 Gate — 坏脚本可能绕过烟雾测试",
            "同步最新 cli_agent.py",
        )
    elif "ensure_preflight_or_exit" not in text:
        _add_warn(
            report,
            "cli_agent_preflight",
            "cli_agent 未集成启动前检查",
            "同步最新 cli_agent.py",
        )
    else:
        _add_ok(report, "cli_agent_gate", "cli_agent 已集成 Gate + Preflight")


def run_preflight_checks(
    *,
    project_root: Optional[str] = None,
    config_path: Optional[str] = None,
) -> PreflightReport:
    """Run all preflight checks and return a structured report."""
    root = _resolve_project_root(project_root)
    os.chdir(root)
    report = PreflightReport(project_root=str(root))

    if config_path and Path(config_path).is_file():
        from multi_agent.config_loader import reload_config
        reload_config(config_path)

    _check_python_version(report)
    _check_imports(report)
    _check_config_files(report, root)
    _check_llm_config(report)
    _check_ablation_and_simulation(report)
    _check_simulation_engine(report)
    _check_knowledge_base(report, root)
    _check_kb_template(report)
    _check_layer2_gate_code(report)
    _check_parser(report)
    _check_guidance_scripts(report, root)
    _check_hermes(report)
    _check_cli_agent_gate_hook(report, root)

    return report


def print_preflight_report(
    report: PreflightReport,
    *,
    stream=None,
) -> None:
    """Print human-readable preflight report."""
    out = stream or sys.stdout
    icons = {_STATUS_OK: "[OK]", _STATUS_WARN: "[WARN]", _STATUS_FAIL: "[FAIL]"}
    print("\n" + "=" * 72, file=out)
    print("  GSMultiAgent 启动前检查 (Preflight)", file=out)
    print("=" * 72, file=out)
    for item in report.items:
        icon = icons.get(item.status, "[??]")
        print(f"  {icon} {item.name}: {item.message}", file=out)
        if item.hint and item.status != _STATUS_OK:
            print(f"         → {item.hint}", file=out)
    print("-" * 72, file=out)
    if report.ok and not report.has_warnings:
        print("  结论: 全部通过，可以启动主流程。", file=out)
    elif report.ok:
        print(f"  结论: 无阻断项，有 {sum(1 for i in report.items if i.status == _STATUS_WARN)} 条警告，可继续运行。", file=out)
    else:
        n_fail = sum(1 for i in report.items if i.status == _STATUS_FAIL)
        print(f"  结论: {n_fail} 项失败，请修复后再运行 cli_agent。", file=out)
    print("=" * 72 + "\n", file=out)


def ensure_preflight_or_exit(
    *,
    project_root: Optional[str] = None,
    config_path: Optional[str] = None,
    strict: bool = False,
    preflight_only: bool = False,
) -> PreflightReport:
    """
    Run preflight checks; exit the process on failure (or strict warnings).

    Returns the report when checks pass (or preflight_only mode).
    """
    report = run_preflight_checks(project_root=project_root, config_path=config_path)
    print_preflight_report(report)

    block = not report.ok or (strict and report.has_warnings)
    if block:
        code = 1
        logger.error("Preflight failed — aborting startup")
        sys.exit(code)
    if preflight_only:
        sys.exit(0)
    return report


__all__ = [
    "CheckItem",
    "PreflightReport",
    "ensure_preflight_or_exit",
    "print_preflight_report",
    "run_preflight_checks",
]
