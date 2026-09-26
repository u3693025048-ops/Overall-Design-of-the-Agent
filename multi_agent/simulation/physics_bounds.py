"""
Physics bounds verification for MODIFY_LAW scripts.

Runs a lightweight MATLAB/Octave probe that calls gf() with extreme
line-of-sight rate geometries and asserts N1/N2 command outputs stay
within the physical limit (default 20g).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_GF_BLOCK_RE = re.compile(
    r"(?m)^function\s+(?:\[[^\]]+\]|[^\n=]+=\s*)?gf\s*\([^\n]*\n[\s\S]*?(?=^function\b|\Z)",
)

_PROBE_RESULT_RE = re.compile(
    r"PHYS_BOUNDS\s+peak_cmd_g=([\d.]+)\s+pass=(\d)",
    re.IGNORECASE,
)

_PROBE_TEMPLATE = """\
%% Auto-generated physics bounds probe — do not edit manually
{gf_block}

function physics_bounds_probe_main()
    limit_g = {limit_g};
    peak = 0;
    Np = 6;
    cases = {{
        struct('Tx',100,'Ty',0,'Tz',0,'V',900,'th',0.35,'pv',0.25,'up',0,'ga',0,'ph',0,'x',5000,'y',0,'z',-500,'TVx',0,'TVy',0,'TVz',0),
        struct('Tx',200,'Ty',50,'Tz',-30,'V',750,'th',-0.2,'pv',0.4,'up',0.1,'ga',0.05,'ph',0.2,'x',8000,'y',200,'z',-800,'TVx',-50,'TVy',10,'TVz',5),
        struct('Tx',50,'Ty',-20,'Tz',10,'V',650,'th',0.5,'pv',-0.35,'up',-0.05,'ga',-0.1,'ph',-0.15,'x',3000,'y',-100,'z',-200,'TVx',20,'TVy',-5,'TVz',0)
    }};
    for i = 1:numel(cases)
        c = cases{{i}};
        XK = zeros(12,1);
        XK(1)=c.V; XK(2)=c.up; XK(3)=c.ga; XK(4)=c.ph; XK(5)=c.th; XK(6)=c.pv;
        XK(10)=c.x; XK(11)=c.y; XK(12)=c.z;
        try
            [N1,N2,~,~,~] = gf(c.Tx,c.Ty,c.Tz,XK,c.TVx,c.TVy,c.TVz,Np);
            peak = max([peak, abs(N1), abs(N2)]);
        catch ME
            fprintf('PHYS_BOUNDS_ERROR case%%d: %%s\\n', i, ME.message);
            fprintf('PHYS_BOUNDS peak_cmd_g=Inf pass=0\\n');
            return;
        end
    end
    pass = peak <= limit_g + 1e-3;
    fprintf('PHYS_BOUNDS peak_cmd_g=%.4f pass=%d\\n', peak, pass);
end

physics_bounds_probe_main();
"""


def extract_gf_function(script_text: str) -> Optional[str]:
    """Return the gf() function block from a multi-function .m script."""
    m = _GF_BLOCK_RE.search(script_text or "")
    if not m:
        return None
    return m.group(0).rstrip()


def build_probe_script(script_text: str, ny_limit_g: float = 20.0) -> Tuple[Optional[str], str]:
    """
    Build standalone probe .m source from *script_text*.

    Returns (probe_source, error_message).  probe_source is None on failure.
    """
    gf_block = extract_gf_function(script_text)
    if not gf_block:
        return None, "未找到 gf() 函数，无法进行物理限幅探测"
    if "function" not in gf_block.lower():
        return None, "gf() 块格式无效"
    source = _PROBE_TEMPLATE.format(gf_block=gf_block, limit_g=float(ny_limit_g))
    return source, ""


def parse_probe_stdout(stdout: str) -> Tuple[bool, float, str]:
    """Parse probe stdout → (passed, peak_cmd_g, detail)."""
    if not stdout:
        return False, float("inf"), "探测无 stdout"
    m = _PROBE_RESULT_RE.search(stdout)
    if not m:
        if "PHYS_BOUNDS_ERROR" in stdout:
            err = stdout.strip().splitlines()[-1]
            return False, float("inf"), err[:300]
        return False, float("inf"), "未找到 PHYS_BOUNDS 标记（gf 探测未执行）"
    peak = float(m.group(1))
    passed = m.group(2) == "1"
    detail = f"peak_cmd_g={peak:.2f}g pass={passed}"
    return passed, peak, detail


def _run_matlab_probe(
    probe_path: str,
    *,
    engine: str = "matlab",
    octave_path: str = "octave",
    matlab_path: str = "matlab",
    timeout_sec: float = 60.0,
) -> Tuple[str, str, int]:
    """Run probe script synchronously; returns (stdout, stderr, returncode)."""
    abs_path = os.path.abspath(probe_path).replace("\\", "/")
    script_dir = os.path.dirname(abs_path).replace("\\", "/")
    base = os.path.splitext(os.path.basename(abs_path))[0]

    if engine == "octave":
        cmd = [
            octave_path,
            "--no-gui",
            "--no-window-system",
            "--quiet",
            "--eval",
            f"addpath('{script_dir}'); {base};",
        ]
    else:
        cmd = [
            matlab_path,
            "-batch",
            f"addpath('{script_dir}'); {base};",
        ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout or "", proc.stderr or "", proc.returncode


async def verify_physics_bounds(
    script_path: str,
    *,
    ny_limit_g: float = 20.0,
    work_dir: Optional[str] = None,
    engine: str = "matlab",
    octave_path: str = "octave",
    matlab_path: str = "matlab",
    timeout_sec: float = 60.0,
) -> Tuple[bool, str]:
    """
    Verify gf() output stays within *ny_limit_g* under extreme LOS stubs.

    Returns (ok, message).
    """
    if not script_path or not os.path.isfile(script_path):
        return False, f"脚本不存在: {script_path}"

    try:
        with open(script_path, "r", encoding="utf-8", errors="ignore") as fh:
            script_text = fh.read()
    except OSError as exc:
        return False, f"无法读取脚本: {exc}"

    probe_src, err = build_probe_script(script_text, ny_limit_g=ny_limit_g)
    if not probe_src:
        return False, err

    out_dir = work_dir or tempfile.mkdtemp(prefix="phys_bounds_")
    os.makedirs(out_dir, exist_ok=True)
    probe_path = os.path.join(out_dir, "physics_bounds_probe.m")
    with open(probe_path, "w", encoding="utf-8") as fh:
        fh.write(probe_src)

    loop = asyncio.get_event_loop()
    try:
        stdout, stderr, rc = await loop.run_in_executor(
            None,
            lambda: _run_matlab_probe(
                probe_path,
                engine=engine,
                octave_path=octave_path,
                matlab_path=matlab_path,
                timeout_sec=timeout_sec,
            ),
        )
    except subprocess.TimeoutExpired:
        return False, f"物理限幅探测超时 ({timeout_sec:.0f}s)"
    except FileNotFoundError as exc:
        return False, f"仿真引擎不可用 ({engine}): {exc}"

    passed, peak, detail = parse_probe_stdout(stdout)
    if passed:
        logger.info("[PhysicsBounds] OK %s (script=%s)", detail, script_path)
        return True, f"物理限幅探测通过: {detail}"

    msg = f"物理限幅探测失败: {detail} (limit={ny_limit_g:.0f}g)"
    if rc != 0 and stderr:
        msg += f"; stderr={stderr[:200]}"
    logger.warning("[PhysicsBounds] FAIL %s — %s", script_path, msg)
    return False, msg


async def verify_physics_bounds_text(
    script_text: str,
    *,
    ny_limit_g: float = 20.0,
    engine: str = "matlab",
    octave_path: str = "octave",
    matlab_path: str = "matlab",
    timeout_sec: float = 60.0,
) -> Tuple[bool, str]:
    """Verify gf() bounds from in-memory script text (before file is committed)."""
    probe_src, err = build_probe_script(script_text, ny_limit_g=ny_limit_g)
    if not probe_src:
        return False, err

    out_dir = tempfile.mkdtemp(prefix="phys_bounds_")
    probe_path = os.path.join(out_dir, "physics_bounds_probe.m")
    with open(probe_path, "w", encoding="utf-8") as fh:
        fh.write(probe_src)

    loop = asyncio.get_event_loop()
    try:
        stdout, stderr, rc = await loop.run_in_executor(
            None,
            lambda: _run_matlab_probe(
                probe_path,
                engine=engine,
                octave_path=octave_path,
                matlab_path=matlab_path,
                timeout_sec=timeout_sec,
            ),
        )
    except subprocess.TimeoutExpired:
        return False, f"物理限幅探测超时 ({timeout_sec:.0f}s)"
    except FileNotFoundError as exc:
        return False, f"仿真引擎不可用 ({engine}): {exc}"

    passed, peak, detail = parse_probe_stdout(stdout)
    if passed:
        return True, f"物理限幅探测通过: {detail}"
    msg = f"物理限幅探测失败: {detail} (limit={ny_limit_g:.0f}g)"
    if rc != 0 and stderr:
        msg += f"; stderr={stderr[:200]}"
    return False, msg
