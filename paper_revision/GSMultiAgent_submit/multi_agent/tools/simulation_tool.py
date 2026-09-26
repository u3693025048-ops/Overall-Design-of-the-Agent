#!/usr/bin/env python3
"""
Simulation Tools for Hermes Agent
Provides SysML model and MATLAB script generation for guidance system simulation.

Both GenerateSysMLTool and GenerateMATLABTool auto-retrieve templates from the
RAG knowledge base before generating output.  If a RAG knowledge base is not
injected the tools fall back to the built-in default templates.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from multi_agent.metric_contract import constraint_violations
from multi_agent.tools.tool_arg_aliases import (
    coerce_script_path as _coerce_script_path,
    coerce_task_prompt as _coerce_task_prompt,
)

logger = logging.getLogger(__name__)


# ── Unicode annotation chars that LLMs insert as "verification" lines ─────────
_ANNOTATION_FIRST_CHARS = frozenset((
    '\u2705', '\u274c', '\u26a0', '\u2714', '\u2716',
    '\U0001f50d', '\U0001f4dd',
))

# Default values for guidance parameters that LLMs add to dp struct but may
# forget to initialise in dp=struct(...)
_DP_GUIDANCE_DEFAULTS: Dict[str, float] = {
    "N_guidance":   3.0,
    "R_switch":   300.0,
    "gama_max_deg": 45.0,
}


def _coerce_mc_seed_list(raw: Any) -> List[int]:
    """Normalize a seed list accepted from config or tool kwargs."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [s.strip() for s in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = [raw]

    seeds: List[int] = []
    for item in items:
        if item is None or str(item).strip() == "":
            continue
        try:
            seeds.append(int(item))
        except (TypeError, ValueError):
            continue
    return seeds


def _mc_validation_auto_runs(mode: Any) -> bool:
    """Whether configured multi-seed MC should apply to every tool call."""
    return str(mode or "").strip().lower() in {
        "all",
        "all_simulations",
        "always",
    }


def _inject_matlab_rng_seed(content: str, random_seed: int) -> str:
    """Patch or insert a MATLAB rng(seed,'twister') call for this run."""
    seed_line = f"rng({int(random_seed)}, 'twister');"
    rng_re = re.compile(r"\brng\s*\([^;\n]*\)\s*;")
    if rng_re.search(content):
        return rng_re.sub(seed_line, content, count=1)

    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if re.match(r"^\s*function\b", line):
            lines.insert(idx + 1, seed_line)
            return "\n".join(lines)

    insert_at = 0
    while insert_at < len(lines):
        stripped = lines[insert_at].strip()
        if stripped and not stripped.startswith("%"):
            break
        insert_at += 1
    lines.insert(insert_at, seed_line)
    return "\n".join(lines)


def _finite_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    try:
        val = float(metrics[key])
    except (KeyError, TypeError, ValueError):
        return None
    if val != val or val in (float("inf"), float("-inf")):
        return None
    return val


def _aggregate_mc_seed_metrics(seed_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build conservative metrics: every scalar reflects the worst seed."""
    from multi_agent.rl.metric_utils import normalize_peak_ny_aliases

    normalized: List[Dict[str, Any]] = []
    for item in seed_metrics:
        m = normalize_peak_ny_aliases(item)
        normalized.append(dict(m))

    def vals(key: str) -> List[float]:
        out: List[float] = []
        for m in normalized:
            v = _finite_metric(m, key)
            if v is not None:
                out.append(v)
        return out

    agg: Dict[str, Any] = {
        "mc_seed_results": normalized,
        "mc_seed_count": len(normalized),
        "mc_seed_values": [
            int(m["mc_seed"]) for m in normalized if _finite_metric(m, "mc_seed") is not None
        ],
        "mc_seed_all_pass": None,
    }

    hit_vals = vals("hit_rate")
    sep_vals = vals("SEP") or vals("miss_distance")
    pny_vals = vals("peak_ny") or vals("peak_n")
    pny_max_vals = vals("peak_ny_max") or vals("peak_n_max")
    pm_vals = vals("pitch_PM")
    bw_vals = vals("pitch_BW")

    if hit_vals:
        agg["hit_rate"] = min(hit_vals)
    if sep_vals:
        agg["SEP"] = max(sep_vals)
        agg["miss_distance"] = agg["SEP"]
    if pny_vals:
        agg["peak_ny"] = max(pny_vals)
        agg["peak_n"] = agg["peak_ny"]
    if pny_max_vals:
        agg["peak_ny_max"] = max(pny_max_vals)
        agg["peak_n_max"] = agg["peak_ny_max"]
    elif pny_vals:
        agg["peak_ny_max"] = agg["peak_ny"]
        agg["peak_n_max"] = agg["peak_ny"]
    if pm_vals:
        agg["pitch_PM"] = min(pm_vals)
        agg["pitch_PM_min"] = min(pm_vals)
        agg["pitch_PM_max"] = max(pm_vals)
    if bw_vals:
        agg["pitch_BW"] = min(bw_vals)
        agg["pitch_BW_min"] = min(bw_vals)
        agg["pitch_BW_max"] = max(bw_vals)
    if any(bool(m.get("_parse_incomplete")) for m in normalized):
        agg["_parse_incomplete"] = True
    return agg


def _sanitize_matlab_guidance_script(content: str) -> Tuple[str, List[str]]:
    """Post-process a MATLAB guidance simulation script before writing to disk.

    Fixes applied (in order):

    1.  Annotation stripping       – ✅❌⚠️ lines → sidecar
    2.  Markdown fence removal     – stray ```matlab / ``` lines
    3.  Unicode math operators     – × ÷ ≥ ≤ ≠ ± → MATLAB equivalents
    4.  Full-width punctuation     – Chinese ；（）：，" " → ASCII
    5.  Prose line comment-out     – pure Chinese / natural-language lines inside
                                     function bodies → prepend %
    6.  dp struct field injection  – add N_guidance/R_switch/gama_max_deg defaults
                                     to dp=struct(...) if missing
    7.  global dp before struct    – inject ``global dp`` before dp=struct(...)
                                     if missing within 5 preceding lines
    8.  run_single_case_general    – inject ``global dp`` + ``dp=sample.dp``
    9.  dp name collision in       – inside guidance_local: if ``dp = <scalar>;``
        guidance_local               conflicts with ``global dp``, rename to
                                     ``dp_rate``

    Returns (cleaned_content, stripped_annotation_lines).
    """
    notes: List[str] = []

    # ── 1. Strip annotation/verification lines ────────────────────────────────
    clean: List[str] = []
    for ln in content.splitlines():
        s = ln.lstrip()
        if s and s[0] in _ANNOTATION_FIRST_CHARS:
            notes.append(ln)
        else:
            clean.append(ln)
    content = "\n".join(clean)

    # ── 2. Remove stray markdown fences ──────────────────────────────────────
    content = re.sub(r"^```(?:matlab)?\s*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^```\s*$",            "", content, flags=re.MULTILINE)

    # ── 2b. Octave endfunction → MATLAB end ─────────────────────────────────
    if re.search(r"\bendfunction\b", content, re.IGNORECASE):
        content = re.sub(r"\bendfunction\b", "end", content, flags=re.IGNORECASE)
        notes.append("[fix] endfunction → end (Octave → MATLAB)")

    # ── 3. Unicode math operator substitution ────────────────────────────────
    _UNICODE_OPS: List[Tuple[str, str]] = [
        ('\u00d7', '*'),    # ×  multiplication sign
        ('\u00f7', '/'),    # ÷  division sign
        ('\u2265', '>='),   # ≥
        ('\u2264', '<='),   # ≤
        ('\u2260', '~='),   # ≠
        ('\u00b1', ''),     # ±  (context-dependent; remove so parser doesn't choke)
        ('\u2192', '%->'),  # →  (likely prose; comment it out)
        ('\u2190', '%<-'),  # ←
        ('\u03bb', 'lambda'),  # λ
        ('\u03c9', 'omega'),   # ω
        ('\u03b6', 'zeta_'),   # ζ (already named zeta in code)
        ('\u03b1', 'alpha_'),  # α (avoid shadowing MATLAB alpha var)
        ('\u03b2', 'beta_'),   # β
        ('\u0307', ''),        # combining dot above (λ̇ → lambda)
        ('\u00b2', '.^2'),     # ²  superscript 2
        ('\u00b3', '.^3'),     # ³
        ('\u221a', 'sqrt'),    # √  (partial – need to manually wrap argument)
    ]
    for uni, asc in _UNICODE_OPS:
        content = content.replace(uni, asc)

    # ── 4. Full-width Chinese punctuation → ASCII ─────────────────────────────
    _FW_MAP: List[Tuple[str, str]] = [
        ('\uff1b', ';'),   # ；
        ('\uff08', '('),   # （
        ('\uff09', ')'),   # ）
        ('\uff1a', ':'),   # ：
        ('\uff0c', ','),   # ，
        ('\u201c', "'"),   # "
        ('\u201d', "'"),   # "
        ('\u2018', "'"),   # '
        ('\u2019', "'"),   # '
        ('\uff05', '%'),   # ％
        ('\uff3b', '['),   # ［
        ('\uff3d', ']'),   # ］
        ('\uff5b', '{'),   # ｛
        ('\uff5d', '}'),   # ｝
        ('\u3002', '.'),   # 。
        ('\uff01', '!'),   # ！
        ('\uff1d', '='),   # ＝
        ('\uff0b', '+'),   # ＋
        ('\uff0d', '-'),   # －
        ('\uff0a', '*'),   # ＊
        ('\uff0f', '/'),   # ／
    ]
    for fw, asc in _FW_MAP:
        content = content.replace(fw, asc)

    # ── 5. Comment-out pure Chinese prose lines inside function bodies ─────────
    # A line is "Chinese prose" if it contains ≥4 CJK characters, has no
    # MATLAB operator (= ; ( [ %) and does NOT start with % already.
    _CJK = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
    _MATLAB_PUNCT = re.compile(r'[=;()\[\]%]')
    prose_fixed = []
    for ln in content.splitlines():
        stripped = ln.lstrip()
        if (stripped and not stripped.startswith('%')
                and len(_CJK.findall(ln)) >= 4
                and not _MATLAB_PUNCT.search(ln)):
            prose_fixed.append('%% [sanitised] ' + ln)
            notes.append('[prose→comment] ' + ln)
        else:
            prose_fixed.append(ln)
    content = "\n".join(prose_fixed)

    # ── 6. dp struct field injection ─────────────────────────────────────────
    def _fix_dp_struct(m: re.Match) -> str:
        struct_line = m.group(0)
        missing = {
            k: v for k, v in _DP_GUIDANCE_DEFAULTS.items()
            if k not in struct_line
        }
        if missing:
            new_fields = ",".join(f"'{k}',{v}" for k, v in missing.items())
            # Check whether the struct already has fields (non-empty interior)
            inner = re.search(r"struct\s*\(([^)]*)\)", struct_line)
            has_existing = inner and inner.group(1).strip()
            sep = "," if has_existing else ""
            struct_line = re.sub(r"\)\s*;", sep + new_fields + ");", struct_line, count=1)
        return struct_line

    content = re.sub(
        r"dp\s*=\s*struct\s*\([^)]*\)\s*;",
        _fix_dp_struct,
        content,
    )

    # ── 7. Inject ``global dp`` before dp=struct(...) if missing ─────────────
    lines = content.splitlines()
    out: List[str] = []
    for i, ln in enumerate(lines):
        if re.match(r"\s*dp\s*=\s*struct\s*\(", ln):
            preceding = "\n".join(lines[max(0, i - 5):i])
            if "global dp" not in preceding:
                out.append("global dp")
        out.append(ln)
    content = "\n".join(out)

    # ── 8. global dp + dp=sample.dp in run_single_case_general ───────────────
    if "function res=run_single_case_general" in content:
        pat = re.compile(
            r"(function\s+res\s*=\s*run_single_case_general\s*\(sample\)[^\n]*\n)"
            r"((?:global\s+\w[^\n]*\n)*)",
            re.DOTALL,
        )
        def _inject_global_dp(m: re.Match) -> str:
            sig, globs = m.group(1), m.group(2)
            if "global dp" not in globs and "global dp" not in sig:
                globs += "global dp\n"
                globs += "if isfield(sample,'dp') && isstruct(sample.dp), dp=sample.dp; end\n"
            return sig + globs
        content = pat.sub(_inject_global_dp, content, count=1)

    # ── 9. dp name collision in guidance_local ────────────────────────────────
    # If guidance_local uses "global dp" AND also assigns dp = <scalar_expr>;
    # the scalar assignment shadows the struct.  Rename scalar uses to dp_rate.
    if "function" in content and "global dp" in content:
        # Find each function body that has BOTH "global dp" AND a bare scalar
        # assignment "dp = <expr>;" where expr does not look like struct().
        def _fix_dp_collision(func_body: str) -> str:
            if "global dp" not in func_body:
                return func_body
            # scalar assignment: dp = <something that is NOT struct(...)>
            scalar_pat = re.compile(
                r"\bdp\s*=\s*(?!struct\s*\()([^;]{1,120});",
            )
            # check if any scalar assignment exists
            if not scalar_pat.search(func_body):
                return func_body
            # rename all bare dp assignments and uses that are NOT dp.field
            # Replace "dp = expr;" → "dp_rate = expr;"
            func_body = scalar_pat.sub(lambda m: m.group(0).replace("dp =", "dp_rate =", 1), func_body)
            # Replace standalone dp in expressions (not dp. or dp_)
            func_body = re.sub(r"\bdp\b(?![._])", "dp_rate", func_body)
            # Restore "global dp" (we just renamed it to global dp_rate)
            func_body = func_body.replace("global dp_rate", "global dp")
            # Restore dp.field references
            func_body = re.sub(r"dp_rate\.([\w]+)", r"dp.\1", func_body)
            return func_body

        # Apply per function body (split on "^function " boundaries)
        func_split = re.split(r"(?=\nfunction\s)", content)
        content = "".join(_fix_dp_collision(fb) for fb in func_split)

    # ── 10. Fix mixed function-end style ──────────────────────────────────────
    # MATLAB requires ALL functions in a file to use the same style:
    # either ALL closed with 'end', or NONE.
    # Strategy: count control-flow openers vs 'end' per function span.
    #   ends == openers     → function has NO closing 'end'
    #   ends == openers + 1 → function IS closed with 'end'
    # If the file is mixed, add a closing 'end' to every function that lacks one.
    _OPENERS = re.compile(
        r"^\s*(?:if|for|while|switch|try|parfor)\b", re.IGNORECASE
    )
    _END_KW  = re.compile(r"^\s*end\s*(?:%.*)?$", re.IGNORECASE)

    def _count_openers_ends(span_lines: List[str]) -> Tuple[int, int]:
        """Count control-flow openers and 'end' keywords (skip comments/strings)."""
        opens, ends = 0, 0
        for sl in span_lines:
            s = sl.strip()
            if not s or s.startswith('%'):
                continue
            if _OPENERS.match(sl):
                opens += 1
            if _END_KW.match(sl):
                ends += 1
        return opens, ends

    lines10 = content.splitlines()
    n10     = len(lines10)

    # Find all top-level function definition line indices
    func_indices10 = [
        i for i, ln in enumerate(lines10)
        if re.match(r'^\s*function\b', ln)
    ]

    if len(func_indices10) > 1:
        # Build spans: (func_start, next_func_start_or_eof)
        spans10 = [
            (func_indices10[k], func_indices10[k + 1] if k + 1 < len(func_indices10) else n10)
            for k in range(len(func_indices10))
        ]

        # Classify each function span
        has_end10: List[bool] = []
        for fi, ni in spans10:
            body_lines = lines10[fi + 1: ni]   # body = everything after function declaration
            opens, ends = _count_openers_ends(body_lines)
            # If ends > opens → extra 'end' closes the function itself
            has_end10.append(ends > opens)

        any_with    = any(has_end10)
        any_without = not all(has_end10)

        if any_with and any_without:
            # Mixed style: insert 'end' after the last code line of each no-end function.
            # Process in reverse order so inserted lines don't shift earlier indices.
            new_lines10 = list(lines10)
            offset = 0   # accumulated line shift from previous insertions
            for k, (fi, ni) in enumerate(spans10):
                if has_end10[k]:
                    continue
                # Find last non-blank, non-comment line in the span (before next func)
                adj_fi = fi + offset
                adj_ni = ni + offset
                insert_at = adj_ni   # default: just before next function
                for j in range(adj_ni - 1, adj_fi, -1):
                    s = new_lines10[j].strip()
                    if s and not s.startswith('%'):
                        insert_at = j + 1
                        break
                new_lines10.insert(insert_at, 'end')
                offset += 1
                notes.append(
                    f'[fix_mixed_end] Inserted closing end after line {insert_at} '
                    f'(function at original line {fi + 1})'
                )
            content = "\n".join(new_lines10)

    # ── 11. Remove LLM-fabricated functions not in the template ──────────────
    # The template (monte_carlo_single style) has a fixed set of subfunctions.
    # The LLM sometimes appends new functions (e.g. dump_metrics, print_result,
    # output_data) that do not exist in the template and break the parser.
    # We detect any function whose name is NOT in the known-good template set
    # and strip its entire body.
    _TEMPLATE_FUNC_NAMES: frozenset = frozenset({
        # monte_carlo_single family
        "monte_carlo_single", "run_category", "run_category_mc", "run_case_mc",
        "print_table", "print_summary", "tf2str", "plot_results",
        "mst", "anoise",
        "sim_s", "cg", "af", "df", "rk4f", "gf", "cf",
        # guidance_local / alternate naming conventions
        "guidance_local", "guidance_func", "guidance_calc",
        "run_single_case_general", "run_all_cases",
        # external MC helpers injected by _patch_external_mc_script
        "apply_external_mc_sample", "field_or",
    })
    lines11 = content.splitlines()
    n11 = len(lines11)
    fi11 = [
        (i, re.match(r'^\s*function\b', lines11[i]).group(0))  # (line_idx, _)
        for i in range(n11)
        if re.match(r'^\s*function\b', lines11[i])
    ]
    if fi11:
        # Extract function names from declarations
        _fname_re11 = re.compile(
            r'^\s*function\s+(?:[^=\s]+\s*=\s*)?([a-zA-Z_]\w*)\s*[\(\s]'
        )
        spans11w = []
        for idx, (li, _) in enumerate(fi11):
            m11 = _fname_re11.match(lines11[li])
            fname11 = m11.group(1) if m11 else ""
            end11   = fi11[idx + 1][0] if idx + 1 < len(fi11) else n11
            spans11w.append((li, end11, fname11))

        # Collect line ranges to delete (fabricated functions only)
        delete_ranges: List[Tuple[int, int]] = []
        for li, end_li, fname11 in spans11w:
            if fname11 and fname11 not in _TEMPLATE_FUNC_NAMES:
                # Also exclude: the first function (main) always kept, and
                # any function that is a renamed version of the main function
                # (generated by _out_basename renaming — detected by it having
                # the same line range as the very first function span).
                if li == spans11w[0][0]:
                    continue  # never delete the main (first) function
                delete_ranges.append((li, end_li))
                notes.append(
                    f'[strip_fabricated] Removed non-template function '
                    f'"{fname11}" (lines {li+1}-{end_li})'
                )

        if delete_ranges:
            # Delete in reverse order to preserve indices
            keep = list(range(n11))
            for start_r, end_r in delete_ranges:
                for ri in range(start_r, end_r):
                    keep[ri] = -1
            lines11 = [lines11[i] for i in range(n11) if keep[i] != -1]
            content = "\n".join(lines11)

    def _func_closing_ends_depth(span_lines: List[str]) -> int:
        """Count function-closing ``end`` keywords using nesting-depth tracking.

        Unlike ``_count_openers_ends`` (which counts openers/ends separately and
        takes their difference), this function tracks the *current nesting depth*
        as it scans each statement.  It therefore handles inline one-liners such
        as ``if cond; stmts; end`` correctly: the opener and its closing ``end``
        are both on the same physical line and balance each other out.

        A bare ``end`` encountered when depth == 0 is a function-closing end.
        """
        depth = 0
        func_ends = 0
        _OPEN_RE  = re.compile(r'^\s*(if|for|while|switch|try|parfor)\b', re.IGNORECASE)
        _END_RE   = re.compile(r'^\s*end\b', re.IGNORECASE)
        _SKIP_RE  = re.compile(r'^\s*(else|elseif|case|otherwise|catch)\b', re.IGNORECASE)
        # Detect one-liner functions whose body AND closing end are on the declaration
        # line itself, e.g.:  function s=tf2str(b); if b; ...; end; end
        # In such cases the closing end is on span_lines[0] and span_lines[1:] skips it.
        # We pre-count it here so any extra `end` in the body is correctly flagged.
        if span_lines:
            decl = span_lines[0]
            # Count semicolon-delimited tokens on the declaration line
            _decl_parts = [p.strip() for p in re.split(r';', decl) if p.strip() and not p.strip().startswith('%')]
            _d_opens = sum(1 for p in _decl_parts if _OPEN_RE.match(p))
            _d_ends  = sum(1 for p in _decl_parts if _END_RE.match(p))
            # If ends > opens on the declaration line, the extra end(s) close the function
            if _d_ends > _d_opens:
                func_ends += _d_ends - _d_opens
        for ln in span_lines[1:]:   # skip the function-declaration line itself
            s = ln.strip()
            if not s or s.startswith('%'):
                continue
            # Split by semicolons to handle inline patterns like `if ..; ..; end`
            parts = re.split(r';', ln)
            for part in parts:
                p = part.strip()
                if not p or p.startswith('%'):
                    continue
                if _SKIP_RE.match(p):
                    continue   # else/elseif/case/catch don't change depth
                if _OPEN_RE.match(p):
                    depth += 1
                elif _END_RE.match(p):
                    if depth == 0:
                        func_ends += 1
                    else:
                        depth -= 1
        return func_ends

    # ── 12. Precise function-end whitelist correction ─────────────────────────
    # Template function-end convention (monte_carlo_single.m):
    #   ALL local functions use explicit 'end': monte_carlo_single, run_category,
    #   print_table, print_summary, tf2str, mst, anoise, sim_s, cg, af, df, rk4f,
    #   gf, cf, plot_results — verified against template (grep '^end').
    #
    # Strategy: all functions must have EXACTLY ONE function-closing end.
    # Strip extras.  _NO_END_FUNCS is intentionally empty.
    _NO_END_FUNCS: frozenset = frozenset()
    lines12 = content.splitlines()
    n12 = len(lines12)
    fi12_all = [i for i in range(n12) if re.match(r'^\s*function\b', lines12[i])]
    if len(fi12_all) >= 2:
        _fname_re12 = re.compile(
            r'^\s*function\s+(?:[^=\s]+\s*=\s*)?([a-zA-Z_]\w*)\s*[\(\s]'
        )
        spans12 = []
        for idx, li in enumerate(fi12_all):
            m12 = _fname_re12.match(lines12[li])
            fname12 = m12.group(1) if m12 else ""
            end12   = fi12_all[idx + 1] if idx + 1 < len(fi12_all) else n12
            spans12.append((li, end12, fname12))

        new_lines12 = list(lines12)
        offset12 = 0
        for li, end_li, fname12 in spans12:
            adj_li  = li  + offset12
            adj_end = end_li + offset12
            body12  = new_lines12[adj_li: adj_end]  # include declaration for _func_closing_ends_depth
            excess12 = _func_closing_ends_depth(body12)  # number of function-closing ends

            if fname12 in _NO_END_FUNCS:
                # These functions must have ZERO function-closing ends → strip all
                n_to_strip = max(excess12, 0)
            else:
                # These functions may have AT MOST ONE function-closing end → strip extras
                n_to_strip = max(excess12 - 1, 0)

            if n_to_strip <= 0:
                continue

            # Strip n_to_strip bare `end` lines, scanning backwards from end of span
            stripped12 = 0
            for j in range(adj_end - 1, adj_li, -1):
                if stripped12 >= n_to_strip:
                    break
                if _END_KW.match(new_lines12[j]):
                    del new_lines12[j]
                    offset12 -= 1
                    adj_end  -= 1
                    stripped12 += 1
                    reason = "no-end template style" if fname12 in _NO_END_FUNCS else "excess function-closing end"
                    notes.append(
                        f'[strip_spurious_end] Removed function-closing end at line {j+1} '
                        f'for function "{fname12}" ({reason})'
                    )
                elif new_lines12[j].strip() and not new_lines12[j].strip().startswith('%'):
                    break  # hit real code before finding another end
        content = "\n".join(new_lines12)

    # ── 13. Ensure gf() overload limiter & output completeness ────────────────
    # If the gf() function assigns N1 and N2, ensure hard limiter lines exist.
    # Also verify all 5 outputs (N1,N2,G1,G2,gc) are assigned; inject defaults
    # for any missing outputs.  This step is DETERMINISTIC — it cannot be broken
    # by LLM hallucinations and guarantees the overload constraint is enforced.
    #
    # Strategy: find gf() function span using function-boundary detection (not
    # greedy/non-greedy end matching which fails when gf() contains if/for/end).
    lines13 = content.splitlines()
    _gf_start_idx = None
    _gf_end_idx = None  # index of the function-closing end line

    # (i) Locate gf() function declaration
    for i, ln in enumerate(lines13):
        if re.match(r'\s*function\s+\[N1,N2,G1,G2,gc\]\s*=\s*gf\b', ln):
            _gf_start_idx = i
            break

    if _gf_start_idx is not None:
        # (ii) Find function span end: next 'function' line or EOF
        _gf_span_end = len(lines13)
        for i in range(_gf_start_idx + 1, len(lines13)):
            if re.match(r'\s*function\b', lines13[i]):
                _gf_span_end = i
                break

        # (iii) Find the function-closing 'end' — last bare 'end' in span
        # Use nesting depth to identify the function-closing end
        _depth = 0
        _func_end_line = None
        _OPEN_KW = re.compile(r'^\s*(if|for|while|switch|try|parfor)\b', re.IGNORECASE)
        _END_KW13 = re.compile(r'^\s*end\b', re.IGNORECASE)
        for i in range(_gf_start_idx + 1, _gf_span_end):
            s = lines13[i].strip()
            if not s or s.startswith('%'):
                continue
            # Handle semicolon-separated statements on one line
            parts = re.split(r';', lines13[i])
            for part in parts:
                p = part.strip()
                if not p or p.startswith('%'):
                    continue
                if _OPEN_KW.match(p):
                    _depth += 1
                elif _END_KW13.match(p):
                    if _depth == 0:
                        _func_end_line = i
                    else:
                        _depth -= 1

        # (iv) Extract body and analyze
        if _func_end_line is not None:
            _gf_end_idx = _func_end_line
        else:
            # No function-closing end found; use span end (will inject end too)
            _gf_end_idx = _gf_span_end

        gf_body = "\n".join(lines13[_gf_start_idx + 1: _gf_end_idx])

        _needs_fix = False

        # (a0) Remove INCORRECT limiter patterns that use division by g0
        # e.g. N1=max(-20/g0, min(20/g0, N1)) — limits to ±2.04 instead of ±20g
        # These are unit-mismatch bugs introduced by LLM (N1 is already in g-units)
        _BAD_LIMIT_N1 = re.compile(
            r"N1\s*=\s*(?:max|min)\s*\([^;]*(?:20\s*/\s*g0|20\s*\*\s*g0|lim_g\s*/\s*g0)[^;]*;[^\n]*",
        )
        _BAD_LIMIT_N2 = re.compile(
            r"N2\s*=\s*(?:max|min)\s*\([^;]*(?:20\s*/\s*g0|20\s*\*\s*g0|lim_g\s*/\s*g0)[^;]*;[^\n]*",
        )
        # Also catch: if abs(N1) > 20 * max(1, abs(g0/9.8))... patterns
        _BAD_IF_LIMIT = re.compile(
            r"if\s+abs\(N[12]\)\s*>[^;]*20\s*\*\s*max\([^)]*g0[^;]*\n"
            r"[^\n]*\n"
            r"\s*end[^\n]*",
            re.MULTILINE,
        )
        for bad_pat in [_BAD_LIMIT_N1, _BAD_LIMIT_N2, _BAD_IF_LIMIT]:
            if bad_pat.search(gf_body):
                gf_body = bad_pat.sub("", gf_body)
                _needs_fix = True
                notes.append("[gf_limiter] Removed incorrect limiter with unit mismatch (20/g0)")

        # (a) Ensure N1/N2 overload hard limiter (±20g) exists
        # ONLY accept correct forms where threshold is plain 20 (not 20/g0, 20*g0, etc.)
        # N1 is in g-units in the template, so ±20 means ±20g.
        # Correct patterns: max(-20,min(20,N1)) or max(min(N1,20),-20) or
        #                   max(-lim_g,min(lim_g,N1)) where lim_g=20
        _has_n1_limit = bool(re.search(
            r"N1\s*=\s*max\s*\(\s*(?:-20|min)\s*,\s*(?:min|max)\s*\(\s*(?:20|N1)\s*,\s*(?:N1|20)\s*\)\s*\)",
            gf_body
        )) or bool(re.search(
            r"N1\s*=\s*max\s*\(\s*min\s*\(\s*N1(?:_raw)?\s*,\s*(?:20|lim_g)\s*\)\s*,\s*(?:-20|-lim_g)\s*\)",
            gf_body
        ))
        _has_n2_limit = bool(re.search(
            r"N2\s*=\s*max\s*\(\s*(?:-20|min)\s*,\s*(?:min|max)\s*\(\s*(?:20|N2)\s*,\s*(?:N2|20)\s*\)\s*\)",
            gf_body
        )) or bool(re.search(
            r"N2\s*=\s*max\s*\(\s*min\s*\(\s*N2(?:_raw)?\s*,\s*(?:20|lim_g)\s*\)\s*,\s*(?:-20|-lim_g)\s*\)",
            gf_body
        ))

        # (b) Ensure G1, G2, gc are assigned somewhere in the body
        _has_G1 = bool(re.search(r"\bG1\s*=", gf_body))
        _has_G2 = bool(re.search(r"\bG2\s*=", gf_body))
        _has_gc = bool(re.search(r"\bgc\s*=", gf_body))

        # Build injection block (inserted just before function-closing end)
        inject_lines = []
        if not _has_n1_limit or not _has_n2_limit:
            inject_lines.append("% ── 过载指令硬限幅 (±20g, N1/N2单位为g) ──")
            if not _has_n1_limit:
                inject_lines.append("N1=max(-20,min(20,N1));")
                notes.append("[gf_limiter] Injected N1 hard limiter ±20g")
            if not _has_n2_limit:
                inject_lines.append("N2=max(-20,min(20,N2));")
                notes.append("[gf_limiter] Injected N2 hard limiter ±20g")
            _needs_fix = True

        if not _has_gc:
            inject_lines.append("gc=0;")
            notes.append("[gf_output] Injected missing gc=0")
            _needs_fix = True
        if not _has_G1:
            inject_lines.append("G1=cos(XK(2))*cos(XK(3));")
            notes.append("[gf_output] Injected missing G1")
            _needs_fix = True
        if not _has_G2:
            inject_lines.append("G2=-cos(XK(2))*sin(XK(3));")
            notes.append("[gf_output] Injected missing G2")
            _needs_fix = True

        if _needs_fix:
            # Re-sync lines13 if gf_body was modified (bad limiters removed)
            new_body_lines = gf_body.splitlines()
            lines13[_gf_start_idx + 1: _gf_end_idx] = new_body_lines
            # Recalculate insertion point
            _gf_end_idx = _gf_start_idx + 1 + len(new_body_lines)

            # Insert injection lines just before the function-closing end
            insert_at = _gf_end_idx
            for j, il in enumerate(inject_lines):
                lines13.insert(insert_at + j, il)
            # If no function-closing end existed, also add 'end'
            if _func_end_line is None:
                lines13.insert(insert_at + len(inject_lines), "end")
                notes.append("[gf_output] Injected missing function-closing end")
            content = "\n".join(lines13)

    return content, notes


@dataclass
class SimulationToolConfig:
    """仿真工具配置"""

    name: str
    description: str
    input_schema: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge-base paths (resolved relative to this file's package root)
# ─────────────────────────────────────────────────────────────────────────────
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
_KB_SYSML  = _PKG_ROOT / "knowledge_base" / "sysml"
_KB_MATLAB = _PKG_ROOT / "knowledge_base" / "matlab"


def _read_kb_file(path: Path) -> str:
    """Read a knowledge-base file, returning empty string on error."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.warning(f"Could not read KB file {path}: {exc}")
        return ""


def check_constraints(
    hit_rate: float,
    SEP: float,
    peak_nz: float,
    PM: float,
    BW: float,
) -> Tuple[bool, Dict[str, float]]:
    """Evaluate paper constraints for one MC result.

    Args:
        hit_rate: Hit rate in percent.
        SEP: Spherical error probable in meters.
        peak_nz: Peak normal acceleration proxy in g.
        PM: Pitch phase margin in degrees.
        BW: Pitch bandwidth in rad/s.

    Returns:
        Tuple of ``(feasible, violations)``.
    """
    return constraint_violations(hit_rate, SEP, peak_nz, PM, BW)


def _matlab_quote_path(path: Path) -> str:
    """Return a MATLAB-safe single-quoted path payload."""
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _scenario_to_t_subcase(mc_data_path: Path) -> Tuple[str, int]:
    """Infer ``RUN_CASE`` and ``SUB_IDX`` from a paper MC dataset path."""
    scenario = mc_data_path.parent.name
    if scenario.startswith("T1"):
        return "T", 1
    if scenario in {"T2", "T3", "T4"}:
        return "T", int(scenario[1])
    logger.warning(
        "Could not infer T-subcase from MC data path '%s'; defaulting to T1",
        mc_data_path,
    )
    return "T", 1


def _infer_external_n_runs(mc_data_path: Path, requested: int) -> int:
    """Infer available runs from the ``.mat`` file without making SciPy mandatory."""
    try:
        from scipy.io import loadmat

        mat = loadmat(str(mc_data_path))
        target_init = mat.get("target_init")
        if target_init is not None and hasattr(target_init, "size"):
            return max(1, min(int(requested), int(target_init.size)))
    except Exception as exc:
        logger.debug("Could not infer n_runs from %s: %s", mc_data_path, exc)
    return int(requested)


def _patch_external_mc_script(
    content: str,
    mc_data_path: Path,
    n_runs: int,
    kill_radius: float,
) -> Tuple[str, str]:
    """Patch a monte_carlo_single-style script to consume external MC data."""
    run_case, sub_idx = _scenario_to_t_subcase(mc_data_path)
    mc_path = _matlab_quote_path(mc_data_path)
    header_re = re.compile(
        r"RUN_CASE\s*=\s*'[^']*'\s*;\s*SUB_IDX\s*=\s*\d+\s*;\s*"
        r"N_MC\s*=\s*\d+\s*;\s*KILL_R\s*=\s*[-+]?\d+(?:\.\d*)?",
    )
    header = (
        f"RUN_CASE='{run_case}'; SUB_IDX={sub_idx}; "
        f"N_MC={int(n_runs)}; KILL_R={float(kill_radius):.12g}"
    )
    content, n_header = header_re.subn(header, content, count=1)
    if n_header == 0:
        content = content.replace(
            "clc; close all;",
            f"clc; close all;\n{header};",
            1,
        )

    external_block = f"""
% EXTERNAL_MC_BEGIN
global EXT_MC_ACTIVE EXT_MC_MISSILE_INIT EXT_MC_TARGET_INIT EXT_MC_SIM_CONFIG;
EXT_MC_ACTIVE = false;
MC_DATA_FILE = '{mc_path}';
if exist(MC_DATA_FILE, 'file')
    mc_ext = load(MC_DATA_FILE);
    EXT_MC_MISSILE_INIT = mc_ext.missile_init;
    EXT_MC_TARGET_INIT = mc_ext.target_init;
    if isfield(mc_ext, 'sim_config')
        EXT_MC_SIM_CONFIG = mc_ext.sim_config;
    else
        EXT_MC_SIM_CONFIG = [];
    end
    EXT_MC_ACTIVE = true;
    N_MC = min(N_MC, numel(EXT_MC_TARGET_INIT));
else
    error('External MC data file not found: %s', MC_DATA_FILE);
end
% EXTERNAL_MC_END
"""
    content = re.sub(
        r"(RUN_CASE\s*=\s*'[^']*'\s*;\s*SUB_IDX\s*=\s*\d+\s*;\s*N_MC\s*=\s*\d+\s*;\s*KILL_R\s*=\s*[-+]?\d+(?:\.\d*)?\s*;?)",
        r"\1" + external_block,
        content,
        count=1,
    )

    content = re.sub(
        r"'T_Vz_amp'\s*,\s*80\s*,\s*'T_Vz_freq'\s*,\s*0\.2\s*,\s*'N_pn'\s*,\s*rl_N_pn\s*,\s*\.\.\.",
        "'T_Vz_amp',80,'T_Vz_freq',0.2,'N_pn',rl_N_pn,'ny_lim',rl_ny_lim,...",
        content,
        count=1,
    )
    content = content.replace(
        "p=anoise(p0,sf);",
        "if EXT_MC_ACTIVE\n"
        "            p=apply_external_mc_sample(p0,k,EXT_MC_MISSILE_INIT,EXT_MC_TARGET_INIT,EXT_MC_SIM_CONFIG);\n"
        "        else\n"
        "            p=anoise(p0,sf);\n"
        "        end",
        1,
    )
    # Prefer serial for-loop by default. Nested MATLAB parfor + outer job-pool
    # concurrency previously spawned dozens of parallelserver workers → OOM/R6025.
    use_parfor = False
    try:
        from multi_agent.simulation.memory_guard import mc_use_parfor

        use_parfor = bool(mc_use_parfor(default=False))
    except Exception:
        use_parfor = False
    loop_kw = "parfor" if use_parfor else "for"
    old_loop = (
        "    res=struct('miss',nan(N_MC,1),'peak_ny',nan(N_MC,1),...\n"
        "               'pitch_pm',nan(N_MC,1),'pitch_bw',nan(N_MC,1),'pitch_gm',nan(N_MC,1));\n"
        "    p0=mst(base,SC{sc});\n"
        "    for k=1:N_MC\n"
    )
    new_loop = (
        "    miss_v=nan(N_MC,1); peak_ny_v=nan(N_MC,1); pitch_pm_v=nan(N_MC,1); "
        "pitch_bw_v=nan(N_MC,1); pitch_gm_v=nan(N_MC,1);\n"
        "    p0=mst(base,SC{sc});\n"
        "    EXT_MC_MI_LOCAL=EXT_MC_MISSILE_INIT; EXT_MC_TI_LOCAL=EXT_MC_TARGET_INIT; "
        "EXT_MC_SC_LOCAL=EXT_MC_SIM_CONFIG; EXT_MC_ON_LOCAL=EXT_MC_ACTIVE;\n"
        f"    {loop_kw} k=1:N_MC\n"
    )
    if old_loop in content:
        content = content.replace(old_loop, new_loop, 1)
        content = content.replace(
            "if EXT_MC_ACTIVE\n"
            "            p=apply_external_mc_sample(p0,k,EXT_MC_MISSILE_INIT,EXT_MC_TARGET_INIT,EXT_MC_SIM_CONFIG);\n"
            "        else\n"
            "            p=anoise(p0,sf);\n"
            "        end",
            "if EXT_MC_ON_LOCAL\n"
            "            p=apply_external_mc_sample(p0,k,EXT_MC_MI_LOCAL,EXT_MC_TI_LOCAL,EXT_MC_SC_LOCAL);\n"
            "        else\n"
            "            p=anoise(p0,sf);\n"
            "        end",
            1,
        )
        content = content.replace(
            "            r=sim_s(p);\n"
            "            res.miss(k)=r.miss; res.peak_ny(k)=r.peak_ny;\n"
            "            res.pitch_pm(k)=r.pitch_pm; res.pitch_bw(k)=r.bw_pitch; res.pitch_gm(k)=r.gm_pitch;\n"
            "            hit=res.miss(k)<KILL_R;\n"
            "            fprintf('  [%3d] %-4s %7.2f   %6.2f  %7.1f  %8.2f  %7.2f\\n',...\n"
            "                k,tf2str(hit),res.miss(k),res.peak_ny(k),res.pitch_pm(k),res.pitch_bw(k),res.pitch_gm(k));\n",
            "            r=sim_s(p);\n"
            "            miss_v(k)=r.miss; peak_ny_v(k)=r.peak_ny;\n"
            "            pitch_pm_v(k)=r.pitch_pm; pitch_bw_v(k)=r.bw_pitch; pitch_gm_v(k)=r.gm_pitch;\n"
            "            hit=miss_v(k)<KILL_R;\n"
            "            fprintf('  [%3d] %-4s %7.2f   %6.2f  %7.1f  %8.2f  %7.2f\\n',...\n"
            "                k,tf2str(hit),miss_v(k),peak_ny_v(k),pitch_pm_v(k),pitch_bw_v(k),pitch_gm_v(k));\n",
            1,
        )
        content = content.replace(
            "    RES{sc}=res;\n",
            "    res=struct('miss',miss_v,'peak_ny',peak_ny_v,'pitch_pm',pitch_pm_v,"
            "'pitch_bw',pitch_bw_v,'pitch_gm',pitch_gm_v);\n"
            "    RES{sc}=res;\n",
            1,
        )
    content = re.sub(
        r"dt\s*=\s*0\.002\s*;\s*tt\s*=\s*70\s*;\s*t\s*=\s*0\s*;",
        "if isfield(p,'dt'); dt=p.dt; else; dt=0.002; end; "
        "if isfield(p,'t_max'); tt=p.t_max; else; tt=70; end; t=0;",
        content,
        count=1,
    )
    content = content.replace(
        "plot_results(RES,SC,KILL_R,RUN_CASE);",
        "if ~EXT_MC_ACTIVE; plot_results(RES,SC,KILL_R,RUN_CASE); end",
        1,
    )
    # run_category is a subfunction: it must re-declare the globals it uses.
    # Without this, MATLAB raises "EXT_MC_ACTIVE cannot be identified".
    content = re.sub(
        r"(function\s+SUMM\s*=\s*run_category\s*\([^\n]*\))",
        r"\1\nglobal EXT_MC_ACTIVE EXT_MC_MISSILE_INIT EXT_MC_TARGET_INIT EXT_MC_SIM_CONFIG;",
        content,
        count=1,
    )
    content = content.replace(
        "[N1,N2,G1,G2,gc]=gf(T(1),T(2),T(3),XK,p.T_Vx,p.T_Vy,TVz,p.N_pn);",
        "[N1,N2,G1,G2,gc]=gf(T(1),T(2),T(3),XK,p.T_Vx,p.T_Vy,TVz,p.N_pn,p.ny_lim);",
        1,
    )
    content = content.replace(
        "function [N1,N2,G1,G2,gc]=gf(Tx,Ty,Tz,XK,TVx,TVy,TVz,Np)",
        "function [N1,N2,G1,G2,gc]=gf(Tx,Ty,Tz,XK,TVx,TVy,TVz,Np,ny_lim)",
        1,
    )
    content = content.replace(
        "N1=max(-20,min(20,N1));\nN2=max(-20,min(20,N2));",
        "N1=max(-ny_lim,min(ny_lim,N1));\nN2=max(-ny_lim,min(ny_lim,N2));",
        1,
    )

    helper = r"""
function p=apply_external_mc_sample(p0,k,missile_init,target_init,sim_config)
p=p0;
m=missile_init(k);
t=target_init(k);
p.x0=field_or(m,'x',p.x0);
p.y0=field_or(m,'alt',p.y0);
p.z0=field_or(m,'z',p.z0);
p.V0=field_or(m,'V',p.V0);
p.theta0=field_or(m,'gamma',p.theta0);
p.phiv0=field_or(m,'psi',p.phiv0);
p.upsilon0=field_or(m,'theta',p.upsilon0);
p.phi0=field_or(m,'phi',p.phi0);
p.gama0=field_or(m,'psi_body',p.gama0);
p.T_x0=field_or(t,'x',p.T_x0);
p.T_y0=field_or(t,'alt',p.T_y0);
p.T_z0=field_or(t,'z',p.T_z0);
p.T_Vx=field_or(t,'Vx',p.T_Vx);
p.T_Vy=field_or(t,'Vy',p.T_Vy);
p.T_Vz_amp=field_or(t,'Az_realized',p.T_Vz_amp);
p.T_Vz_freq=field_or(t,'omega_z_realized',p.T_Vz_freq);
if ~isempty(sim_config)
    sc=sim_config(k);
    p.dt=field_or(sc,'dt',0.001);
    p.t_max=field_or(sc,'t_max',70.0);
else
    p.dt=0.001;
    p.t_max=70.0;
end
end

function v=field_or(s,name,default)
v=default;
if isstruct(s) && isfield(s,name)
    raw=s.(name);
    if iscell(raw); raw=raw{1}; end
    if ~isempty(raw)
        v=double(raw);
        if numel(v)>1; v=v(1); end
    end
end
end
"""
    content = content.replace("\nfunction print_table", "\n" + helper + "\nfunction print_table", 1)
    return content, f"{run_case}{sub_idx}"


def _extract_mc_vectors(stdout: str) -> Tuple[List[float], List[float]]:
    """Extract per-run miss distances and peak normal loads from MATLAB stdout."""
    miss_distances: List[float] = []
    nz_peaks: List[float] = []
    pat = re.compile(
        r"^\s*\[\s*\d+\]\s+(?:HIT|MISS)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)",
        re.MULTILINE,
    )
    for match in pat.finditer(stdout or ""):
        try:
            miss_distances.append(float(match.group(1)))
            nz_peaks.append(float(match.group(2)))
        except ValueError:
            continue
    return miss_distances, nz_peaks


def _external_mc_error(message: str, sim_time_sec: float = 0.0) -> Dict[str, Any]:
    """Build a non-throwing external MC failure result."""
    feasible, violations = check_constraints(0.0, 1e9, 1e9, 0.0, 0.0)
    return {
        "hit_rate": 0.0,
        "SEP": float("inf"),
        "peak_nz": float("inf"),
        "PM": 0.0,
        "BW": 0.0,
        "miss_distances": [],
        "nz_peaks": [],
        "feasible": feasible,
        "constraint_violations": violations,
        "sim_time_sec": float(sim_time_sec),
        "success": False,
        "error": message,
        "error_msg": message,
    }


def run_mc_with_external_data(
    matlab_script_path: str,
    mc_data_path: str,
    n_runs: int = 100,
    kill_radius: float = 10.0,
    dp_params: Optional[Dict[str, float]] = None,
    timeout_sec: float = 300.0,
) -> Dict[str, Any]:
    """Run MATLAB/Octave Monte Carlo using pre-generated external MC data.

    Args:
        matlab_script_path: Path to ``monte_carlo_single.m`` or a generated variant.
        mc_data_path: Path to the external ``.mat`` file containing MC conditions.
        n_runs: Number of MC samples from the external file to run.
        kill_radius: Hit threshold in meters.
        dp_params: Optional 10-D tunable parameter overrides.
        timeout_sec: MATLAB/Octave subprocess timeout in seconds.

    Returns:
        Dictionary with paper metrics, feasibility, timing, and error fields.
    """
    import subprocess
    import uuid

    started = time.perf_counter()
    script_path = Path(matlab_script_path)
    data_path = Path(mc_data_path)
    if not script_path.exists():
        return _external_mc_error(f"MATLAB script not found: {matlab_script_path}")
    if not data_path.exists():
        return _external_mc_error(f"MC data file not found: {mc_data_path}")

    content = _read_kb_file(script_path)
    if not content:
        return _external_mc_error(f"Could not read MATLAB script: {matlab_script_path}")

    n_runs = _infer_external_n_runs(data_path, int(n_runs))
    if dp_params:
        content = GenerateMATLABTool._patch_autopilot_params(content, dp_params)
    content, scenario_label = _patch_external_mc_script(
        content, data_path, n_runs, kill_radius
    )
    try:
        _san_content, _san_notes = _sanitize_matlab_guidance_script(content)
        if _san_notes:
            logger.debug(
                "[run_mc_with_external_data] sanitizer applied %d fix(es)",
                len(_san_notes),
            )
        content = _san_content
    except Exception as exc:
        logger.warning("[run_mc_with_external_data] sanitizer failed: %s", exc)

    func_match = re.match(r"^\s*function\s+(?:[^=\s]+\s*=\s*)?([a-zA-Z_]\w*)", content)
    func_name = func_match.group(1) if func_match else None
    unique_tag = uuid.uuid4().hex[:8]
    tmp_dir = script_path.parent
    if func_name:
        call_name = f"{func_name}_extmc_{unique_tag}"
        tmp_name = f"{call_name}.m"
        content = re.sub(
            r"^function\s+" + re.escape(func_name),
            f"function {call_name}",
            content,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        call_name = None
        tmp_name = f"extmc_{scenario_label}_{unique_tag}.m"
    tmp_path = tmp_dir / tmp_name

    proc = None
    try:
        tmp_path.write_text(content, encoding="utf-8")
        try:
            from multi_agent.config_loader import get_config
            from multi_agent.simulation.engine_resolver import resolve_engine_config

            sim_cfg = get_config().simulation
            engine_cfg = resolve_engine_config(
                engine=sim_cfg.engine,
                octave_path=sim_cfg.octave_path,
                matlab_path=sim_cfg.matlab_path,
            )
            engine = engine_cfg.engine
            octave_path = engine_cfg.octave_path
            matlab_path = engine_cfg.matlab_path
        except Exception:
            engine = "matlab"
            octave_path = "octave"
            matlab_path = "matlab"

        if engine == "matlab_engine":
            engine = "matlab"
        if engine == "auto":
            engine = "matlab"

        abs_path = str(tmp_path.resolve()).replace("\\", "/")
        script_dir = str(tmp_dir.resolve()).replace("\\", "/")
        if engine == "octave":
            from multi_agent.simulation.guidance_simulator import (
                OCTAVE_BATCH_FLAGS,
                build_octave_eval_string,
            )

            eval_str = build_octave_eval_string(abs_path, call_name)
            cmd = [octave_path, *OCTAVE_BATCH_FLAGS, "--eval", eval_str]
            eval_body = eval_str
        elif engine == "matlab":
            from multi_agent.simulation.memory_guard import matlab_eval_preamble

            preamble = matlab_eval_preamble()
            if call_name:
                eval_body = f"{preamble}addpath('{script_dir}'); {call_name};"
            else:
                eval_body = f"{preamble}run('{abs_path}');"
            # One-shot -batch must exit; persistent workers must NOT.
            eval_str = eval_body + " exit"
            cmd = [
                matlab_path,
                "-nosplash",
                "-nodesktop",
                "-nodisplay",
                "-batch",
                eval_str,
            ]
        else:
            return _external_mc_error(f"Unsupported engine for external MC: {engine}")

        logger.info(
            "[run_mc_with_external_data] %s n_runs=%d script=%s data=%s",
            engine,
            n_runs,
            script_path.name,
            data_path,
        )

        # Prefer persistent MATLAB workers (skip cold start). Fall back to -batch.
        used_pool = False
        stdout = ""
        stderr = ""
        returncode = 1
        if engine == "matlab":
            try:
                from multi_agent.simulation.matlab_job_pool import get_job_pool

                pool = get_job_pool()
                if pool is not None:
                    from multi_agent.simulation.matlab_parallel import matlab_process_slot

                    with matlab_process_slot():
                        pooled = pool.run_eval(
                            eval_body, timeout_sec=float(timeout_sec)
                        )
                    used_pool = True
                    stdout = str(pooled.get("stdout") or "")
                    stderr = ""
                    returncode = 0 if pooled.get("ok") else 1
            except Exception as exc:
                logger.warning(
                    "[run_mc_with_external_data] job pool failed (%s); falling back to -batch",
                    exc,
                )
                used_pool = False

        if not used_pool:
            from multi_agent.simulation.matlab_parallel import matlab_process_slot

            with matlab_process_slot():
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                while proc.poll() is None:
                    if time.perf_counter() - started > float(timeout_sec):
                        proc.kill()
                        proc.wait(timeout=5)
                        return _external_mc_error(
                            "MATLAB timeout",
                            sim_time_sec=time.perf_counter() - started,
                        )
                    time.sleep(1.0)

                import locale as _loc

                enc = _loc.getpreferredencoding(False) or "utf-8"
                stdout = (proc.stdout.read() or b"").decode(enc, errors="replace")
                stderr = (proc.stderr.read() or b"").decode(enc, errors="replace")
                returncode = int(proc.returncode or 0)

        sim_time = time.perf_counter() - started
        if returncode != 0:
            msg = (stderr or stdout or f"returncode={returncode}")[:2000]
            logger.warning("[run_mc_with_external_data] MATLAB failed: %s", msg)
            err = _external_mc_error(msg, sim_time_sec=sim_time)
            try:
                from multi_agent.simulation.memory_guard import looks_like_oom

                if looks_like_oom(msg):
                    err["oom"] = True
                    err["error"] = "MATLAB_OOM: " + str(err.get("error") or msg)[:500]
                    err["error_msg"] = err["error"]
            except Exception:
                pass
            return err

        from multi_agent.rl.matlab_rl_optimizer import parse_sim_stdout

        metrics = parse_sim_stdout(stdout)
        miss_distances, nz_peaks = _extract_mc_vectors(stdout)
        if miss_distances:
            hit_rate = (
                sum(v < float(kill_radius) for v in miss_distances)
                / len(miss_distances)
                * 100.0
            )
            ordered = sorted(float(v) for v in miss_distances)
            mid = len(ordered) // 2
            sep = (
                ordered[mid]
                if len(ordered) % 2
                else 0.5 * (ordered[mid - 1] + ordered[mid])
            )
        else:
            hit_rate = float(metrics.get("hit_rate", 0.0))
            sep = float(
                metrics.get("SEP", metrics.get("miss_distance", float("inf")))
            )
        if nz_peaks:
            peak_nz = sum(float(v) for v in nz_peaks) / len(nz_peaks)
        else:
            peak_nz = float(metrics.get("peak_ny", metrics.get("peak_n", 0.0)))
        pm = float(metrics.get("pitch_PM", 0.0))
        bw = float(metrics.get("pitch_BW", 0.0))
        feasible, violations = check_constraints(hit_rate, sep, peak_nz, pm, bw)
        return {
            "hit_rate": hit_rate,
            "SEP": sep,
            "peak_nz": peak_nz,
            "PM": pm,
            "BW": bw,
            "miss_distances": miss_distances,
            "nz_peaks": nz_peaks,
            "feasible": feasible,
            "constraint_violations": violations,
            "sim_time_sec": sim_time,
            "success": True,
            "error": None,
            "error_msg": None,
            "matlab_job_pool": bool(used_pool),
        }
    except FileNotFoundError as exc:
        return _external_mc_error(f"Engine executable not found: {exc}")
    except Exception as exc:
        logger.exception("[run_mc_with_external_data] failed")
        return _external_mc_error(str(exc), sim_time_sec=time.perf_counter() - started)
    finally:
        try:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        except Exception:
            pass
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def run_mc_and_across_seeds(
    matlab_script_path: str,
    *,
    mc_root: str,
    scenario: str,
    seeds: List[int],
    split: str = "dev",
    n_runs: int = 100,
    kill_radius: float = 10.0,
    dp_params: Optional[Dict[str, float]] = None,
    timeout_sec: float = 300.0,
    early_abort: Optional[bool] = None,
    bootstrap_fill: Optional[bool] = None,
) -> Dict[str, Any]:
    """Evaluate params on every external-MC seed and apply AND feasibility.

    Paper §4.1: a candidate passes only when *all* listed seeds simultaneously
    satisfy the five hard constraints. Aggregated metrics use conservative
    pooling (min hit_rate, max SEP/peak, min PM/BW).

    When ``mc_and_bootstrap_fill`` / ``GSM_MC_AND_BOOTSTRAP`` is enabled, only
    the first seed is executed in MATLAB; remaining seed records are filled by
    bootstrap sampling from that seed's trajectory vectors. Pass
    ``bootstrap_fill=False`` to force a real MATLAB run on every listed seed.
    """
    if not seeds:
        return _external_mc_error("run_mc_and_across_seeds: empty seeds list")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from multi_agent.simulation.matlab_parallel import resolve_mc_and_workers
    from multi_agent.simulation.mc_eval_cache import (
        cache_get,
        cache_put,
        make_cache_key,
    )
    from multi_agent.simulation.mc_seed_bootstrap import (
        bootstrap_fill_enabled,
        bootstrap_seed_result,
    )

    root = Path(mc_root)
    seed_list = [int(s) for s in seeds]
    if bootstrap_fill is None:
        bootstrap_fill = bootstrap_fill_enabled(default=True)
    else:
        bootstrap_fill = bool(bootstrap_fill)
    eval_seeds = seed_list[:1] if bootstrap_fill and len(seed_list) > 1 else seed_list
    workers = resolve_mc_and_workers(len(eval_seeds))
    if early_abort is None:
        env_ab = os.environ.get("GSM_MC_AND_EARLY_ABORT", "").strip().lower()
        if env_ab in ("0", "false", "no", "off"):
            early_abort = False
        elif env_ab in ("1", "true", "yes", "on"):
            early_abort = True
        else:
            try:
                from multi_agent.config_loader import get_config

                early_abort = bool(
                    getattr(get_config().simulation, "mc_and_early_abort", True)
                )
            except Exception:
                early_abort = True

    cache_mode = "and_bootstrap_first" if bootstrap_fill else "and_full"
    cache_key = make_cache_key(
        matlab_script_path=matlab_script_path,
        mc_root=str(root),
        scenario=scenario,
        seeds=seed_list,
        split=split,
        n_runs=int(n_runs),
        kill_radius=float(kill_radius),
        dp_params=dp_params,
        mode=cache_mode,
    )
    cached = cache_get(cache_key)
    if cached is not None:
        out = dict(cached)
        out["from_cache"] = True
        return out

    def _eval_one(seed: int) -> Dict[str, Any]:
        data_path = root / scenario / f"seed_{seed}_{split}.mat"
        result = run_mc_with_external_data(
            matlab_script_path,
            str(data_path),
            n_runs=n_runs,
            kill_radius=kill_radius,
            dp_params=dp_params,
            timeout_sec=timeout_sec,
        )
        return dict(result, mc_seed=seed, split=split, synthetic=False)

    def _skipped(seed: int, reason: str) -> Dict[str, Any]:
        return {
            "success": False,
            "feasible": False,
            "hit_rate": 0.0,
            "SEP": 1e9,
            "peak_nz": 1e9,
            "PM": 0.0,
            "BW": 0.0,
            "sim_time_sec": 0.0,
            "error": reason,
            "error_msg": reason,
            "mc_seed": seed,
            "split": split,
            "skipped": True,
            "synthetic": False,
        }

    wall0 = time.perf_counter()
    seed_results: List[Dict[str, Any]] = []
    if workers <= 1 or len(eval_seeds) <= 1:
        for seed in eval_seeds:
            item = _eval_one(seed)
            seed_results.append(item)
            if early_abort and (
                not bool(item.get("success", False))
                or not bool(item.get("feasible", False))
            ):
                for rest in eval_seeds[len(seed_results) :]:
                    seed_results.append(
                        _skipped(rest, "early_abort_after_failed_seed")
                    )
                break
    else:
        logger.info(
            "[run_mc_and_across_seeds] parallel seeds=%s workers=%d early_abort=%s",
            eval_seeds,
            workers,
            early_abort,
        )
        by_seed: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_one, seed): seed for seed in eval_seeds}
            abort = False
            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    by_seed[seed] = fut.result()
                except Exception as exc:
                    by_seed[seed] = dict(
                        _external_mc_error(str(exc)),
                        mc_seed=seed,
                        split=split,
                        synthetic=False,
                    )
                item = by_seed[seed]
                if early_abort and (
                    not bool(item.get("success", False))
                    or not bool(item.get("feasible", False))
                ):
                    abort = True
                    for other_fut, other_seed in futures.items():
                        if other_seed not in by_seed:
                            other_fut.cancel()
                    break
            if abort:
                for seed in eval_seeds:
                    if seed not in by_seed:
                        by_seed[seed] = _skipped(
                            seed, "early_abort_after_failed_seed"
                        )
        seed_results = [by_seed[s] for s in eval_seeds]

    if bootstrap_fill and len(seed_list) > 1:
        source = seed_results[0] if seed_results else _skipped(
            seed_list[0], "bootstrap_missing_source"
        )
        if not seed_results:
            seed_results = [source]
        filled: List[Dict[str, Any]] = [dict(source, synthetic=False)]
        for target in seed_list[1:]:
            filled.append(
                bootstrap_seed_result(
                    source,
                    target_seed=int(target),
                    source_seed=int(seed_list[0]),
                    split=split,
                    kill_radius=float(kill_radius),
                    n_runs=int(n_runs),
                )
            )
        seed_results = filled
        logger.info(
            "[run_mc_and_across_seeds] bootstrap_fill source=%s synthetic=%s",
            seed_list[0],
            seed_list[1:],
        )

    sim_time = sum(float(r.get("sim_time_sec", 0.0) or 0.0) for r in seed_results)
    wall_time = time.perf_counter() - wall0

    all_pass = True
    sim_ok = True
    hit_vals: List[float] = []
    sep_vals: List[float] = []
    peak_vals: List[float] = []
    pm_vals: List[float] = []
    bw_vals: List[float] = []
    errors: List[str] = []
    for item in seed_results:
        ran = bool(item.get("success", False))
        feasible = bool(item.get("feasible", False))
        sim_ok = sim_ok and ran
        all_pass = all_pass and ran and feasible
        if not ran:
            errors.append(
                str(
                    item.get("error")
                    or item.get("error_msg")
                    or f"seed {item.get('mc_seed')} failed"
                )
            )
        try:
            hit_vals.append(float(item.get("hit_rate", 0.0)))
        except (TypeError, ValueError):
            hit_vals.append(0.0)
        try:
            sep_vals.append(float(item.get("SEP", 1e9)))
        except (TypeError, ValueError):
            sep_vals.append(1e9)
        try:
            peak_vals.append(
                float(
                    item.get(
                        "peak_nz",
                        item.get("peak_ny", item.get("peak_n", 1e9)),
                    )
                )
            )
        except (TypeError, ValueError):
            peak_vals.append(1e9)
        try:
            pm_vals.append(float(item.get("PM", item.get("pitch_PM", 0.0))))
        except (TypeError, ValueError):
            pm_vals.append(0.0)
        try:
            bw_vals.append(float(item.get("BW", item.get("pitch_BW", 0.0))))
        except (TypeError, ValueError):
            bw_vals.append(0.0)

    matlab_calls = 1 if bootstrap_fill and len(seed_list) > 1 else len(seed_list)
    # Count only non-synthetic / non-skipped real evals that were attempted.
    real_runs = [
        r
        for r in seed_results
        if not r.get("synthetic") and not r.get("skipped")
    ]
    if real_runs:
        matlab_calls = len(real_runs)
    elif bootstrap_fill:
        matlab_calls = 1

    out = {
        "hit_rate": min(hit_vals) if hit_vals else 0.0,
        "SEP": max(sep_vals) if sep_vals else 1e9,
        "peak_nz": max(peak_vals) if peak_vals else 1e9,
        "peak_ny": max(peak_vals) if peak_vals else 1e9,
        "PM": min(pm_vals) if pm_vals else 0.0,
        "BW": min(bw_vals) if bw_vals else 0.0,
        "miss_distances": [],
        "nz_peaks": [],
        "feasible": bool(all_pass),
        "constraint_violations": {} if all_pass else {"and_logic": 1.0},
        "sim_time_sec": float(sim_time),
        "wall_time_sec": float(wall_time),
        "success": bool(sim_ok),
        "error": None if sim_ok else "; ".join(errors[:5]),
        "error_msg": None if sim_ok else "; ".join(errors[:5]),
        "mc_calls": int(matlab_calls),
        "mc_seeds": [int(s) for s in seeds],
        "seed_results": seed_results,
        "and_logic": True,
        "mc_and_workers": int(workers),
        "bootstrap_fill": bool(bootstrap_fill and len(seed_list) > 1),
        "from_cache": False,
    }
    if out.get("success") or out.get("feasible") is not None:
        cache_put(cache_key, out)
    if len(seed_list) > 1 and seed_results:
        try:
            from multi_agent.simulation.bootstrap_export import write_bootstrap_export

            source_row = next(
                (r for r in seed_results if not r.get("synthetic") and not r.get("skipped")),
                seed_results[0],
            )
            write_bootstrap_export(
                scenario=scenario,
                split=split,
                source_seed=int(seed_list[0]),
                target_seeds=seed_list[1:],
                n_runs=int(n_runs),
                kill_radius=float(kill_radius),
                base_result=dict(source_row),
                matlab_script_path=matlab_script_path,
                dp_params=dp_params,
                local_aggregate={
                    "hit_rate": out.get("hit_rate"),
                    "SEP": out.get("SEP"),
                    "peak_nz": out.get("peak_nz"),
                    "PM": out.get("PM"),
                    "BW": out.get("BW"),
                    "feasible": out.get("feasible"),
                    "success": out.get("success"),
                    "bootstrap_fill": out.get("bootstrap_fill"),
                },
            )
        except Exception as exc:
            logger.debug("[run_mc_and_across_seeds] bootstrap_export skip: %s", exc)
    return out


class GenerateSysMLTool:
    """
    Generate SysML models for guidance system architecture.

    Templates are retrieved **automatically** from the RAG knowledge base
    (``knowledge_base/sysml/``).  Three diagram types are produced:
      - BDD  (Block Definition Diagram)  → ``*_bdd.xml``
      - Parametric Diagram               → ``*_parametric.xml``
      - IBD  (Internal Block Diagram)    → ``*_ibd.xml``

    All files are saved to the configured ``sysml_output_dir``.
    """

    name = "generate_sysml"
    description = """
    Generate SysML models (BDD, Parametric, IBD) for the guidance system.
    Automatically retrieves templates from the knowledge base via RAG.
    Saves generated .xml files to the SysML output directory.
    Use this for system architecture design or documentation tasks.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Model/project name used in file names and XML",
                "default": "GuidanceSystem",
            },
            "navigation_coefficient": {
                "type": "number",
                "description": "Proportional navigation coefficient N (e.g. 3.5)",
            },
            "damping_ratio": {
                "type": "number",
                "description": "Autopilot damping ratio ζ (e.g. 0.75)",
            },
            "control_gain": {
                "type": "number",
                "description": "Control gain (default 1.0)",
                "default": 1.0,
            },
            "target_position": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Target position [x, y, z] in metres (default [20000,2000,5000])",
            },
            "task_description": {
                "type": "string",
                "description": "Free-text description of the task/scenario (used for RAG retrieval and LLM customisation)",
            },
            "mode": {
                "type": "string",
                "description": "Generation mode: REUSE_HISTORY | TUNE_PARAMS | MODIFY_LAW",
                "enum": ["REUSE_HISTORY", "TUNE_PARAMS", "MODIFY_LAW"],
                "default": "TUNE_PARAMS",
            },
        },
        "required": [],
    }

    def __init__(self, *args, **kwargs):
        self.simulator = None
        self.rag_kb = None
        self.parameter_experience = None
        self._model_agent = None
        super().__init__(*args, **kwargs)

    def set_simulator(self, simulator) -> None:
        self.simulator = simulator

    def set_rag_kb(self, rag_kb) -> None:
        self.rag_kb = rag_kb

    def set_parameter_experience(self, parameter_experience) -> None:
        self.parameter_experience = parameter_experience

    def _get_model_agent(self):
        """Lazy-instantiate ModelGenerationAgent (avoids import cost on every run)."""
        if self._model_agent is None:
            try:
                from ..integration.model_generation_agent import ModelGenerationAgent
                self._model_agent = ModelGenerationAgent()
            except Exception as exc:
                logger.warning(f"[SysML] Could not create ModelGenerationAgent: {exc}")
        return self._model_agent

    async def _load_pe_sysml(self, diagram_type: str, query_hint: str = "") -> Optional[str]:
        """
        Return the best-matching PE SysML XML for *diagram_type* from
        ``experience_base_dir/models/``, or ``None`` if none found.

        Retrieval priority:
        1. ``ParameterExperience.retrieve_best()`` when PE is injected —
           structural similarity × fitness × reliability (Q-score).
        2. JSON file scan with fitness + keyword scoring (fallback when PE
           is not loaded).
        3. Most-recent file by mtime (last resort).
        """
        import json as _json
        try:
            from ..config_loader import get_config
            pe_base   = Path(get_config().model_output.experience_base_dir)
            pe_models = pe_base / "models"
            pe_params = pe_base / "params"
            if not pe_models.exists():
                return None
            candidates = sorted(
                pe_models.glob(f"*_{diagram_type}_*.xml"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                return None

            # ── Priority 1: ParameterExperience structural retrieval ──────────
            if self.parameter_experience is not None:
                try:
                    task_ctx: Dict[str, Any] = {"task": "guidance_rl_optimization"}
                    if query_hint:
                        task_ctx["prompt"] = query_hint
                    pe_results = await self.parameter_experience.retrieve_best(
                        task_ctx, top_k=3
                    )
                    for pe_rec in pe_results:
                        mem_id = pe_rec.get("memory_id", "")
                        if not mem_id:
                            continue
                        matched = next(
                            (c for c in candidates if c.name.startswith(mem_id)), None
                        )
                        if matched:
                            content = _read_kb_file(matched)
                            if content:
                                logger.info(
                                    f"[SysML] PE structural retrieval: '{matched.name}' "
                                    f"(Q={pe_rec.get('quality_score', 0):.3f}, "
                                    f"fitness={pe_rec.get('fitness', 0):.3f}, "
                                    f"sim={pe_rec.get('similarity', 0):.3f})"
                                )
                                return content
                except Exception as exc:
                    logger.warning(f"[SysML] PE retrieve_best failed: {exc}")

            # ── Priority 2: JSON file scan with fitness + keyword scoring ─────
            best_hash, best_score = None, -1.0
            if query_hint and pe_params.exists():
                kw = [w for w in query_hint.lower().split() if len(w) > 3]
                for jf in pe_params.glob("*.json"):
                    try:
                        rec      = _json.loads(jf.read_text(encoding="utf-8"))
                        fitness  = float(rec.get("fitness", 0.0))
                        kw_bonus = 0.1 if kw and any(
                            w in rec.get("task_context", {}).get("prompt", "").lower()
                            for w in kw
                        ) else 0.0
                        score = fitness + kw_bonus
                        if score > best_score:
                            best_score = score
                            best_hash  = rec.get("memory_id", "")
                    except Exception:
                        pass

            matched = None
            if best_hash:
                matched = next((c for c in candidates if c.name.startswith(best_hash)), None)
            target = matched or candidates[0]

            content = _read_kb_file(target)
            if content:
                logger.info(
                    f"[SysML] Loaded PE {diagram_type} model '{target.name}' "
                    f"(matched={'yes' if matched else 'latest'}, score={best_score:.3f})"
                )
                return content
        except Exception as exc:
            logger.warning(f"[SysML] _load_pe_sysml failed: {exc}")
        return None

    # ── template retrieval ────────────────────────────────────────────────────

    async def _retrieve_template(self, diagram_type: str, query_hint: str) -> str:
        """
        Retrieve a SysML template.  Priority:
        1. RAG knowledge base (full-text search)
        2. Direct file read from ``knowledge_base/sysml/``
        """
        filename_map = {
            "bdd":        "sysml_bdd_reference.xml",
            "parametric": "sysml_parametric_reference.xml",
            "ibd":        "sysml_ibd_reference.xml",
        }
        fname = filename_map.get(diagram_type, "sysml_bdd_reference.xml")

        # Try RAG first
        if self.rag_kb:
            try:
                query = f"SysML {diagram_type.upper()} guidance system {query_hint}"
                results = await self.rag_kb.retrieve(query=query, top_k=3)
                for r in results:
                    content = r.get("content", "")
                    src = r.get("metadata", {}).get("filename", "")
                    # Prefer results that look like XML SysML diagrams
                    if ("<SysML" in content or "SysML" in content) and fname.split(".")[0].split("_")[-1] in src.lower():
                        logger.info(f"[SysML] Retrieved {diagram_type} template via RAG (src={src})")
                        return content
                # Fallback: return best-scored result if it contains XML
                for r in results:
                    if "<SysML" in r.get("content", ""):
                        return r["content"]
            except Exception as exc:
                logger.warning(f"RAG retrieval for SysML {diagram_type} failed: {exc}")

        # Direct KB file read
        content = _read_kb_file(_KB_SYSML / fname)
        if content:
            logger.info(f"[SysML] Using local KB template: {fname}")
        return content

    # ── template parametrization ──────────────────────────────────────────────

    @staticmethod
    def _fill_template(template: str, substitutions: Dict[str, str]) -> str:
        """Replace ``{key}`` placeholders in template with values."""
        result = template
        for key, val in substitutions.items():
            result = result.replace(f"{{{key}}}", str(val))
        return result

    # ── main execute ──────────────────────────────────────────────────────────

    async def execute(
        self,
        model_name: str = "GuidanceSystem",
        navigation_coefficient: float = 3.5,
        damping_ratio: float = 0.75,
        control_gain: float = 1.0,
        target_position: Optional[List[float]] = None,
        task_description: str = "",
        mode: str = "TUNE_PARAMS",
    ) -> Dict[str, Any]:
        """Retrieve SysML templates from knowledge base, generate via ModelGenerationAgent, and save."""
        try:
            from ..config_loader import get_config
            cfg = get_config()
            out_dir = Path(cfg.model_output.sysml_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            tpos = target_position or [20000.0, 2000.0, 5000.0]
            subs = {
                "model_name":             model_name,
                "navigation_coefficient": str(navigation_coefficient),
                "damping_ratio":          str(damping_ratio),
                "control_gain":           str(control_gain),
                "target_position_0":      str(tpos[0]),
                "target_position_1":      str(tpos[1]),
                "target_position_2":      str(tpos[2]),
            }

            hint = task_description or f"nav={navigation_coefficient} damping={damping_ratio}"
            ts   = int(time.time())
            saved: Dict[str, str] = {}

            diagram_configs = [
                ("bdd",        f"{model_name}_bdd_{ts}.xml"),
                ("parametric", f"{model_name}_parametric_{ts}.xml"),
                ("ibd",        f"{model_name}_ibd_{ts}.xml"),
            ]

            agent = self._get_model_agent() if mode == "MODIFY_LAW" else None

            for diag_type, fname in diagram_configs:
                if mode == "REUSE_HISTORY":
                    # Prefer best-matching PE XML; fall back to KB template
                    content = await self._load_pe_sysml(diag_type, query_hint=hint)
                    if not content:
                        template = await self._retrieve_template(diag_type, hint)
                        content = self._fill_template(template, subs) if template else ""
                        logger.info(
                            f"[SysML] REUSE_HISTORY: no PE {diag_type} found, "
                            "using KB template."
                        )
                elif mode == "TUNE_PARAMS":
                    # Check PE first (richer starting point); fall back to KB template
                    pe_base_content = await self._load_pe_sysml(diag_type, query_hint=hint)
                    if pe_base_content:
                        template = pe_base_content
                        logger.info(f"[SysML] TUNE_PARAMS: seeding from PE model for {diag_type}")
                    else:
                        template = await self._retrieve_template(diag_type, hint)
                    if not template:
                        logger.warning(f"[SysML] No template for {diag_type}, skipping.")
                        continue
                    content = self._fill_template(template, subs)
                else:  # MODIFY_LAW
                    # Use best-matching PE model as LLM base if available;
                    # fall back to KB template.  PE experience gives the LLM
                    # a better-tuned starting architecture to adapt from.
                    pe_base_content = await self._load_pe_sysml(diag_type, query_hint=hint)
                    kb_template     = await self._retrieve_template(diag_type, hint)
                    template = pe_base_content or kb_template
                    if pe_base_content:
                        logger.info(f"[SysML] MODIFY_LAW: seeding LLM from PE model for {diag_type}")
                    if not template:
                        logger.warning(f"[SysML] No template for {diag_type}, skipping.")
                        continue
                    if agent:
                        content = await agent.generate_sysml(
                            task_description=task_description,
                            template=template,
                            diagram_type=diag_type,
                            substitutions=subs,
                        )
                    else:
                        content = self._fill_template(template, subs)

                if not content:
                    logger.warning(f"[SysML] Empty content for {diag_type}, skipping.")
                    continue
                out_path = out_dir / fname
                out_path.write_text(content, encoding="utf-8")
                saved[diag_type] = str(out_path)
                logger.info(f"[SysML] Saved {diag_type} → {out_path}  (mode={mode})")

            return {
                "status":          "success",
                "model_name":      model_name,
                "output_dir":      str(out_dir),
                "generated_files": saved,
                "mode":            mode,
                "template_source": "pe_model" if mode == "REUSE_HISTORY" else "rag_knowledge_base",
            }
        except Exception as exc:
            logger.error(f"SysML generation failed: {exc}")
            return {"status": "error", "message": str(exc)}


class GenerateMATLABTool:
    """
    Generate MATLAB simulation scripts for the guidance system.

    The template is retrieved **automatically** from the RAG knowledge base
    (``knowledge_base/matlab/``).  The tool then applies the requested
    autopilot design parameters (``dp.*``) and mission condition flags
    (``run_A``, ``sub_A``, etc.) to produce a ready-to-run script.

    The generated script is saved to the configured ``matlab_scripts_dir``.
    """

    name = "generate_matlab"
    description = """
    Generate a MATLAB simulation script from the knowledge-base template.
    Automatically retrieves the best-matching MATLAB template via RAG
    (monte_carlo_single.m — primary template; legacy fallbacks also supported),
    then patches it with the specified autopilot design parameters and
    mission conditions (RUN_CASE / SUB_IDX for T/G/AP/R categories).
    Saves the result to the MATLAB scripts directory.
    Use this when you need to create or refresh the simulation script.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "script_name": {
                "type": "string",
                "description": "Output file name (without .m extension)",
                "default": "guidance_simulation",
            },
            "task_description": {
                "type": "string",
                "description": "Free-text description used for RAG template retrieval and LLM generation",
            },
            "mode": {
                "type": "string",
                "description": (
                    "Generation mode: "
                    "REUSE_HISTORY (minimal patching), "
                    "TUNE_PARAMS (patch dp.* params + conditions), "
                    "MODIFY_LAW (LLM rewrites guidance-law function body)."
                ),
                "enum": ["REUSE_HISTORY", "TUNE_PARAMS", "MODIFY_LAW"],
                "default": "TUNE_PARAMS",
            },
            "autopilot_params": {
                "type": "object",
                "description": (
                    "Autopilot design parameter overrides (11 params).  "
                    "Keys: w1, zeta1, tao1, w2, zeta2, tao2, w3, zeta3, tao3, N_pn.  "
                    "These map to rl_w1/rl_zeta1/rl_tao1/... in the RL_PARAMS block.  "
                    "Missing keys keep the template default."
                ),
                "additionalProperties": {"type": "number"},
            },
            "mission_conditions": {
                "type": "string",
                "description": (
                    "Mission condition overrides, e.g. "
                    "'run_A=true;sub_A=[];run_B=false;'  "
                    "Overwrites the corresponding flags in the template.  "
                    "Leave empty for non-standard conditions and set "
                    "non_standard_task instead."
                ),
            },
            "non_standard_task": {
                "type": "string",
                "description": (
                    "Free-text description of working conditions that do NOT map "
                    "to standard A-F categories.  The ModelGenerationAgent will "
                    "infer the closest A-F flag mapping via LLM."
                ),
            },
            "nmc": {
                "type": "integer",
                "description": "Number of Monte Carlo runs (Nmc in the script)",
                "default": 100,
            },
        },
        "required": [],
    }

    # Priority-ordered list of MATLAB template file names to search for
    _TEMPLATE_PRIORITY = [
        "monte_carlo_single.m",                # current primary template
        "MC_gongkuang_simulation_robust_all.m",  # legacy fallback
        "chengxu_robust_analysis_singlefile.m",  # legacy fallback
    ]

    def __init__(self, *args, **kwargs):
        self.simulator = None
        self.rag_kb = None
        self.parameter_experience = None
        self.last_script_path: str = ""
        self.last_modification_desc: str = ""  # task description for MODIFY_LAW
        self._model_agent = None
        self._analysis_context: str = ""  # accumulated analysis/reflection suggestions
        self._iterative_seed_path: str = ""  # gate-passed / Layer3 script for next MODIFY
        super().__init__(*args, **kwargs)

    def set_iterative_seed(self, path: str) -> None:
        """Set preferred on-disk seed for the next execute() (cli / gate-passed)."""
        self._iterative_seed_path = (path or "").strip()

    def set_gate_passed_seed(self, path: str) -> None:
        """Alias for set_iterative_seed — script that passed Layer2 Gate."""
        self.set_iterative_seed(path)

    def set_analysis_context(self, context: str) -> None:
        """Inject analysis suggestions from previous iterations.

        These are prepended to ``task_description`` in every ``execute()``
        call so the LLM always sees them — regardless of what Hermes decides
        to put in its ``task_description`` argument.
        """
        self._analysis_context = (context or "").strip()

    def set_simulator(self, simulator) -> None:
        self.simulator = simulator

    def set_rag_kb(self, rag_kb) -> None:
        self.rag_kb = rag_kb

    def set_parameter_experience(self, parameter_experience) -> None:
        self.parameter_experience = parameter_experience

    def _get_model_agent(self):
        """Lazy-instantiate ModelGenerationAgent."""
        if self._model_agent is None:
            try:
                from ..integration.model_generation_agent import ModelGenerationAgent
                self._model_agent = ModelGenerationAgent()
            except Exception as exc:
                logger.warning(f"[MATLAB] Could not create ModelGenerationAgent: {exc}")
        return self._model_agent

    async def _load_pe_model(self, query: str = "") -> Optional[tuple]:
        """
        Return ``(content, source_label)`` for the best-matching PE model
        script in ``experience_base_dir/models/``, or ``None`` if none exist.

        Retrieval priority:
        1. ``ParameterExperience.retrieve_best()`` when PE is injected —
           structural similarity × fitness × reliability (Q-score).
        2. JSON file scan with fitness + keyword scoring (fallback when PE
           is not loaded).
        3. Most-recent file by mtime (last resort).
        """
        import json as _json
        try:
            from ..config_loader import get_config
            cfg       = get_config()
            pe_base   = Path(cfg.model_output.experience_base_dir)
            pe_models = pe_base / "models"
            pe_params = pe_base / "params"
            if not pe_models.exists():
                return None
            candidates = sorted(
                pe_models.glob("*.m"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                return None

            # ── Priority 1: ParameterExperience structural retrieval ──────────
            if self.parameter_experience is not None:
                try:
                    task_ctx: Dict[str, Any] = {"task": "guidance_rl_optimization"}
                    if query:
                        task_ctx["prompt"] = query
                    pe_results = await self.parameter_experience.retrieve_best(
                        task_ctx, top_k=3
                    )
                    for pe_rec in pe_results:
                        mem_id = pe_rec.get("memory_id", "")
                        if not mem_id:
                            continue
                        matched = next(
                            (c for c in candidates if c.name.startswith(mem_id)), None
                        )
                        if matched:
                            content = _read_kb_file(matched)
                            if content:
                                logger.info(
                                    f"[MATLAB] PE structural retrieval: '{matched.name}' "
                                    f"(Q={pe_rec.get('quality_score', 0):.3f}, "
                                    f"fitness={pe_rec.get('fitness', 0):.3f}, "
                                    f"sim={pe_rec.get('similarity', 0):.3f})"
                                )
                                return content, (
                                    f"pe_model:{matched.name}"
                                    f"(Q={pe_rec.get('quality_score', 0):.3f})"
                                )
                except Exception as exc:
                    logger.warning(f"[MATLAB] PE retrieve_best failed: {exc}")

            # ── Priority 2: JSON file scan with fitness + keyword scoring ─────
            best_hash, best_score = None, -1.0
            if query and pe_params.exists():
                kw = [w for w in query.lower().split() if len(w) > 3]
                for jf in pe_params.glob("*.json"):
                    try:
                        rec      = _json.loads(jf.read_text(encoding="utf-8"))
                        fitness  = float(rec.get("fitness", 0.0))
                        kw_bonus = 0.1 if kw and any(
                            w in rec.get("task_context", {}).get("prompt", "").lower()
                            for w in kw
                        ) else 0.0
                        score = fitness + kw_bonus
                        if score > best_score:
                            best_score = score
                            best_hash  = rec.get("memory_id", "")
                    except Exception:
                        pass

            matched = None
            if best_hash:
                matched = next((c for c in candidates if c.name.startswith(best_hash)), None)
            target = matched or candidates[0]

            content = _read_kb_file(target)
            if content:
                logger.info(
                    f"[MATLAB] Loaded PE model '{target.name}' "
                    f"(matched={'yes' if matched else 'latest'}, score={best_score:.3f})"
                )
                return content, (
                    f"pe_model:{target.name}(score={best_score:.3f})"
                    if matched else
                    f"pe_model:{target.name}(latest)"
                )
        except Exception as exc:
            logger.warning(f"[MATLAB] _load_pe_model failed: {exc}")
        return None

    # ── script code analysis ────────────────────────────────────────────────

    @staticmethod
    def _analyze_retrieved_script(content: str, source_label: str = "") -> str:
        """Extract key function blocks from a retrieved MATLAB script and
        return a structured code summary that can be injected into the LLM
        prompt for informed code generation.

        Extracts: RL_PARAMS block + gf/cf/sim_s/cg/df/rk4f/af functions.
        """
        if not content:
            return ""
        try:
            parts: list = []
            lines = content.splitlines()

            # 1. RL_PARAMS block
            _rl_m = re.search(
                r"(%% ── RL_PARAMS_BEGIN.*?%% ── RL_PARAMS_END[^\n]*)",
                content, re.DOTALL,
            )
            if _rl_m:
                parts.append(f"% === RL参数块 ===\n{_rl_m.group(1).strip()}")

            # 2. Extract target function blocks
            func_starts = [i for i, ln in enumerate(lines)
                           if re.match(r"^function\s+", ln)]
            _TARGET = {"gf", "cf", "sim_s", "cg", "df", "rk4f", "af"}

            for j, start in enumerate(func_starts):
                end = func_starts[j + 1] - 1 if j + 1 < len(func_starts) else len(lines) - 1
                header = lines[start]
                fm = re.search(r"(?:=\s*|\s)(\w+)\s*\(", header)
                if fm and fm.group(1) in _TARGET:
                    block = "\n".join(lines[start:end + 1]).strip()
                    parts.append(f"% === {fm.group(1)}() ===\n{block}")

            if not parts:
                return ""

            code_summary = "\n\n".join(parts)
            # Cap to ~6000 chars
            if len(code_summary) > 6000:
                code_summary = code_summary[:6000] + "\n... (truncated)"

            label = f" (来源: {source_label})" if source_label else ""
            return (
                f"[检索到的参考模型代码{label}]\n"
                f"以下是检索到的最匹配脚本的核心函数，请在此基础上修改，"
                f"保留其中合理的部分，改进不足之处：\n\n{code_summary}"
            )
        except Exception as exc:
            logger.warning(f"[MATLAB] _analyze_retrieved_script failed: {exc}")
            return ""

    # ── scripts folder retrieval ─────────────────────────────────────────────

    def _load_scripts_folder_model(self, query: str = "") -> Optional[tuple]:
        """Search ``guidance_output/scripts/`` for a usable MATLAB script.

        Returns ``(content, source_label)`` or ``None``.

        Selection logic:
        1. All ``*.m`` files are scored by keyword overlap with *query* and
           validated with ``_is_real_matlab_template``.
        2. Among valid candidates, the highest-scoring (keyword + recency) wins.
        3. If no keyword match, the most recent valid file is returned.
        """
        try:
            from ..config_loader import get_config
            scripts_dir = Path(get_config().model_output.generated_dir).parent / "scripts"
            if not scripts_dir.exists():
                return None

            candidates = sorted(
                scripts_dir.glob("*.m"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                return None

            kw = [w.lower() for w in (query or "").split() if len(w) > 2]

            best_file: Optional[Path] = None
            best_score = -1.0
            for cand in candidates[:30]:  # cap scan to 30 most recent
                content = _read_kb_file(cand)
                if not content or not self._is_real_matlab_template(content):
                    continue
                from multi_agent.integration.script_seed_policy import script_content_rl_ready
                _rl_ok, _ = script_content_rl_ready(content)
                if not _rl_ok:
                    continue
                # Score: keyword bonus + recency bonus (newer = higher)
                name_low = cand.stem.lower()
                kw_score = sum(1.0 for w in kw if w in name_low)
                # Normalise mtime to [0,1] within the candidate set
                score = kw_score + 0.01  # +0.01 so any valid file beats -1
                if score > best_score:
                    best_score = score
                    best_file = cand

            if best_file is not None:
                content = _read_kb_file(best_file)
                if content:
                    logger.info(
                        f"[MATLAB] Loaded template from scripts folder: "
                        f"'{best_file.name}' (kw_score={best_score:.1f})"
                    )
                    return content, f"scripts:{best_file.name}"
        except Exception as exc:
            logger.warning(f"[MATLAB] _load_scripts_folder_model failed: {exc}")
        return None

    # ── template retrieval ────────────────────────────────────────────────────

    async def _retrieve_template(self, query: str) -> tuple:
        """
        Return ``(template_content, source_label)``.

        Priority:
        1. Direct file read from ``knowledge_base/matlab/``
           (preferred: monte_carlo_single.m).
           Local files are the **source of truth** and always up-to-date.
        2. RAG knowledge base — fallback when no local file matches.
           (RAG may contain stale indexed content from before bug fixes.)
        """
        # 1. Direct file read (always prefer the on-disk source of truth)
        for fname in self._TEMPLATE_PRIORITY:
            for subdir in ["guidance", "robust_analysis", ""]:
                candidate = _KB_MATLAB / subdir / fname if subdir else _KB_MATLAB / fname
                content = _read_kb_file(candidate)
                if content:
                    logger.info(f"[MATLAB] Using local KB template: {candidate}")
                    return content, f"local:{fname}"

        # 2. RAG retrieval (fallback only) — strict content gate so a
        #    markdown documentation chunk that mentions "function" and "dp."
        #    can't be mistaken for a real MATLAB template.
        if self.rag_kb:
            try:
                rag_query = f"MATLAB guidance simulation robust {query}"
                results = await self.rag_kb.retrieve(query=rag_query, top_k=5)
                for r in results:
                    content = r.get("content", "")
                    src = r.get("metadata", {}).get("filename", "")
                    if self._is_real_matlab_template(content):
                        logger.info(
                            f"[MATLAB] Retrieved template via RAG "
                            f"(src={src}, score={r.get('score',0):.2f})"
                        )
                        return content, f"rag:{src}"
                    else:
                        logger.debug(
                            f"[MATLAB] RAG result rejected (src={src}): "
                            f"does not look like a real MATLAB template"
                        )
            except Exception as exc:
                logger.warning(f"RAG retrieval for MATLAB template failed: {exc}")

        return "", "not_found"

    @staticmethod
    def _is_real_matlab_template(content: str) -> bool:
        """Strict gate: only accept content that is unambiguously a
        MATLAB template (not a markdown doc, not LLM prose).

        Required positive markers:
          • Source file name suffix `.m` is implied by the text features
            below (markdown docs would not pass them).
          • Multiple ``dp.<name> = <number>;`` numeric assignments
            (the autopilot parameter block — load-bearing for RL patching).
          • A function declaration OR a ``clc;``/``clear`` script header.

        Hard rejections:
          • Markdown fenced code markers (triple-backtick blocks).
          • Two or more markdown-style headings (``# `` at line start).
        """
        if not content or not content.strip():
            return False
        # Hard rejections
        if "```" in content:
            return False
        if len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+\S", content)) >= 2:
            return False
        # Must have at least 3 parameter assignments: either dp.* (legacy) or
        # rl_* RL_PARAMS block variables (new monte_carlo_single template style).
        n_dp_assigns = len(re.findall(
            r"dp\.[a-zA-Z_]\w*\s*=\s*[-+]?\d", content
        ))
        n_rl_assigns = len(re.findall(
            r"rl_[a-zA-Z_]\w*\s*=\s*[-+]?\d", content
        ))
        if n_dp_assigns < 3 and n_rl_assigns < 3:
            return False
        # Must have either a function header or a clc/clear script header
        has_function = bool(re.search(r"(?m)^\s*function\s+\w", content))
        has_script_header = bool(re.search(r"(?m)^\s*(clc|clear)\b", content))
        if not (has_function or has_script_header):
            return False
        # Minimum line count: header-only stubs (< 80 lines) are incomplete
        if len(content.splitlines()) < 80:
            return False
        # Must contain at least one MC-body marker (MC loop or guidance function)
        _body_re = re.compile(
            r"run_category_mc|run_case_mc|for.*N_MC|function\s+gf\b"
            r"|function\s+guidance_local\b|function\s+cg\b|mont.*carlo",
            re.IGNORECASE | re.MULTILINE,
        )
        if not _body_re.search(content):
            return False
        return True

    # ── parameter patching ────────────────────────────────────────────────────

    # RL_PARAMS block variable name mapping (Python key → MATLAB rl_* variable)
    _RL_PARAM_MAP: Dict[str, str] = {
        "w1":    "rl_w1",    "zeta1": "rl_zeta1", "tao1": "rl_tao1",
        "w2":    "rl_w2",    "zeta2": "rl_zeta2", "tao2": "rl_tao2",
        "w3":    "rl_w3",    "zeta3": "rl_zeta3", "tao3": "rl_tao3",
        "N_pn":  "rl_N_pn",  "ny_lim": "rl_ny_lim",
        # aliases
        "N_guidance": "rl_N_pn",
    }

    @classmethod
    def _patch_autopilot_params(cls, script: str, params: Dict[str, float]) -> str:
        """Patch parameters. Strategy A: RL_PARAMS block (monte_carlo_single).
        Strategy B: dp.key = value (legacy templates)."""
        # Strategy A: RL_PARAMS block (match %% or % prefix, flexible spacing)
        rl_block_pat = re.compile(
            r"(%+.*?RL_PARAMS_BEGIN.*?\n)(.*?)(%+.*?RL_PARAMS_END)",
            re.DOTALL,
        )
        rl_match = rl_block_pat.search(script)
        if rl_match:
            block = rl_match.group(2)
            matched_any = False
            for py_key, ml_var in cls._RL_PARAM_MAP.items():
                val = params.get(py_key)
                if val is None:
                    continue
                new_block, n = re.subn(
                    rf"({re.escape(ml_var)}\s*=\s*)[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?(\s*;)",
                    rf"\g<1>{val:.6f}\2",
                    block,
                )
                if n:
                    block = new_block
                    matched_any = True
            if matched_any:
                script = rl_block_pat.sub(
                    lambda m: m.group(1) + block + m.group(3), script, count=1
                )
                return script
        # Strategy B: bare rl_key = value anywhere in script (outside block or block missing)
        patched_b = False
        for py_key, ml_var in cls._RL_PARAM_MAP.items():
            val = params.get(py_key)
            if val is None:
                continue
            new_script, n = re.subn(
                rf"({re.escape(ml_var)}\s*=\s*)[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?(\s*;)",
                rf"\g<1>{val:.6f}\2",
                script,
            )
            if n:
                script = new_script
                patched_b = True
        if patched_b:
            return script
        # Strategy C: dp.key = value (legacy templates)
        for key, val in params.items():
            script = re.sub(
                rf"(dp\.{key}\s*=\s*)[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?\s*;",
                f"dp.{key} = {val:.6f};",
                script,
            )
        return script

    @staticmethod
    def _patch_mission_conditions(script: str, conditions_str: str) -> str:
        """
        Apply mission conditions to a MATLAB script.

        Supports two formats:
          New: "RUN_CASE='T';SUB_IDX=0;" (monte_carlo_single style)
          Legacy: "run_A=true;sub_A=[1,2];run_B=false;" (old template style)
        """
        # ── New format: RUN_CASE / SUB_IDX ───────────────────────────────────
        rc_m = re.search(r"RUN_CASE\s*=\s*'([A-Za-z]+)'", conditions_str)
        si_m = re.search(r"SUB_IDX\s*=\s*(\d+)", conditions_str)
        if rc_m:
            rc_val = rc_m.group(1)
            si_val = si_m.group(1) if si_m else "0"
            script = re.sub(
                r"RUN_CASE\s*=\s*'[A-Za-z]+'", f"RUN_CASE='{rc_val}'", script
            )
            script = re.sub(r"SUB_IDX\s*=\s*\d+", f"SUB_IDX={si_val}", script)
            return script

        # ── Legacy format: run_X / sub_X ────────────────────────────────────
        _ALL_CATS = list("ABCDEF")
        enabled: set = set()
        for m in re.finditer(r"run_([A-F])\s*=\s*(true|false)", conditions_str, re.IGNORECASE):
            cat, val = m.group(1).upper(), m.group(2).lower()
            if val == "true":
                enabled.add(cat)
        for cat in _ALL_CATS:
            val = "true" if cat in enabled else "false"
            script = re.sub(rf"run_{cat}\s*=\s*(true|false)\s*;", f"run_{cat} = {val};", script)
        for m in re.finditer(r"sub_([A-F])\s*=\s*\[(.*?)\]", conditions_str, re.IGNORECASE):
            cat, content = m.group(1).upper(), m.group(2).strip()
            script = re.sub(rf"sub_{cat}\s*=\s*\[.*?\]\s*;", f"sub_{cat} = [{content}];", script)
        return script

    # ── main execute ──────────────────────────────────────────────────────────

    async def execute(
        self,
        script_name: str = "guidance_simulation",
        task_description: str = "",
        mode: str = "TUNE_PARAMS",
        autopilot_params: Optional[Dict[str, float]] = None,
        mission_conditions: Optional[str] = None,
        non_standard_task: str = "",
        nmc: Optional[int] = None,
        force_kb_template: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Retrieve MATLAB template from KB, generate/patch via ModelGenerationAgent, and save."""
        task_description = _coerce_task_prompt(task_description, **kwargs) or str(
            kwargs.get("script_name") or script_name or ""
        )
        try:
            from ..config_loader import get_config
            cfg = get_config()
            nmc = int(nmc if nmc is not None else cfg.matlab_rl_optimizer.nmc_per_eval)
            out_dir = Path(cfg.model_output.matlab_scripts_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            from multi_agent.integration.t4_low_risk import (
                clamp_autopilot_params,
                hermes_modify_law_prompt_suffix,
                is_t4_mission,
            )
            if is_t4_mission(mission_conditions, task_description) or mode == "MODIFY_LAW":
                autopilot_params = clamp_autopilot_params(autopilot_params)

            _modify_law_ok = True
            _modify_law_msg = ""
            rewrite_applied = False

            query = task_description or script_name

            # ── Mode-specific template selection ──────────────────────────────
            if mode == "REUSE_HISTORY":
                # Prefer best-matching PE model (fitness + keyword); fall back to
                # scripts folder, then KB template.
                pe_result = await self._load_pe_model(query=query)
                if pe_result:
                    content, source = pe_result
                else:
                    scripts_result = self._load_scripts_folder_model(query=query)
                    if scripts_result:
                        content, source = scripts_result
                        logger.info("[MATLAB] REUSE_HISTORY: using scripts folder model")
                    else:
                        content, source = await self._retrieve_template(query)
                        logger.info("[MATLAB] REUSE_HISTORY: no PE/scripts model found, using KB template")

                if not content:
                    return {
                        "status": "error",
                        "message": "REUSE_HISTORY: no model found in PE base or KB.",
                    }
                # Minimal patching: only Nmc + mission_conditions (no param tuning)
                content = re.sub(r"Nmc\s*=\s*\d+;", f"Nmc = {nmc};", content)
                if mission_conditions:
                    content = self._patch_mission_conditions(content, mission_conditions)
                rewrite_applied = False

            else:
                from multi_agent.integration.script_seed_policy import resolve_matlab_seed

                _seed_policy = str(getattr(cfg.simulation, "seed_policy", "iterative"))
                _gate_path = getattr(self, "_gate_passed_path", "") or ""
                _layer3_path = getattr(self, "_layer3_script_path", "") or ""

                _choice = await resolve_matlab_seed(
                    mode=mode,
                    task_description=query,
                    force_kb_template=force_kb_template,
                    prefer_kb_template=bool(
                        getattr(cfg.simulation, "prefer_kb_template", False)
                    ),
                    seed_policy=_seed_policy,
                    explicit_seed_path=self._iterative_seed_path,
                    gate_passed_path=_gate_path,
                    layer3_script_path=_layer3_path,
                    load_pe_model=self._load_pe_model,
                    load_scripts_folder_model=self._load_scripts_folder_model,
                    load_kb_template=self._retrieve_template,
                )
                content, source = _choice.content, _choice.source
                if not content:
                    return {
                        "status": "error",
                        "message": "No MATLAB template found in knowledge base or local files.",
                    }
                _retrieved_analysis = self._analyze_retrieved_script(content, source)

                if _retrieved_analysis:
                    logger.info(
                        f"[MATLAB] Generated {len(_retrieved_analysis)}-char code analysis "
                        f"of retrieved model for LLM context."
                    )

                # Build effective task description:
                #   1. Reflection suggestions from previous iteration
                #   2. Code analysis of the retrieved model
                #   3. Current task description
                _effective_desc = task_description or script_name
                _ctx_blocks = []
                if self._analysis_context:
                    _ctx_blocks.append(
                        f"[前序分析建议 — 必须优先满足]\n{self._analysis_context}"
                    )
                if _retrieved_analysis:
                    _ctx_blocks.append(_retrieved_analysis)
                if _ctx_blocks:
                    _effective_desc = (
                        "\n\n".join(_ctx_blocks)
                        + f"\n\n[当前任务描述]\n{_effective_desc}"
                    )
                _t4_suffix = hermes_modify_law_prompt_suffix(mission_conditions, task_description)
                if _t4_suffix and mode == "MODIFY_LAW":
                    _effective_desc = _effective_desc + _t4_suffix
                if _ctx_blocks:
                    logger.info(
                        "[MATLAB] Injected %d-char context into task_description "
                        "(analysis=%d, retrieved_code=%d).",
                        len(_effective_desc),
                        len(self._analysis_context),
                        len(_retrieved_analysis),
                    )

                agent = self._get_model_agent()
                _seed_content = content
                if agent:
                    content = await agent.generate_matlab(
                        task_description=_effective_desc,
                        template_content=content,
                        mode=mode,
                        autopilot_params=autopilot_params,
                        mission_conditions=mission_conditions,
                        nmc=nmc,
                        non_standard_task=non_standard_task,
                        seed_source=source,
                    )
                    rewrite_applied = (mode == "MODIFY_LAW")
                    if mode == "MODIFY_LAW":
                        from multi_agent.integration.design_path_policy import (
                            verify_modify_law_result,
                        )

                        _modify_law_ok, _modify_law_msg = verify_modify_law_result(
                            content, _seed_content, require_structural=True,
                        )
                        if not _modify_law_ok:
                            logger.warning(
                                "[MATLAB] MODIFY_LAW gf() verify failed: %s — retry once",
                                _modify_law_msg,
                            )
                            _retry_desc = (
                                f"{_effective_desc}\n\n[强制要求] {_modify_law_msg}。"
                                "必须重写 gf()：加入 APN 目标加速度前馈、末段 N 调度、"
                                "20g 指令限幅或低通滤波；禁止与种子脚本 gf() 完全相同。"
                            )
                            content = await agent.generate_matlab(
                                task_description=_retry_desc,
                                template_content=_seed_content,
                                mode=mode,
                                autopilot_params=autopilot_params,
                                mission_conditions=mission_conditions,
                                nmc=nmc,
                                non_standard_task=non_standard_task,
                                seed_source=source,
                            )
                            _modify_law_ok, _modify_law_msg = verify_modify_law_result(
                                content, _seed_content, require_structural=True,
                            )
                            if not _modify_law_ok:
                                logger.error(
                                    "[MATLAB] MODIFY_LAW retry failed: %s",
                                    _modify_law_msg,
                                )
                            else:
                                logger.info(
                                    "[MATLAB] MODIFY_LAW verified after retry: %s",
                                    _modify_law_msg,
                                )
                        from multi_agent.config_loader import get_config as _gc_pb
                        _sim_pb = _gc_pb().simulation
                        from multi_agent.integration.t4_low_risk import (
                            should_skip_physics_bounds_probe,
                        )
                        from multi_agent.simulation.sim_timeout import (
                            get_physics_bounds_timeout_sec,
                        )

                        if getattr(_sim_pb, "physics_bounds_enabled", True) and not (
                            should_skip_physics_bounds_probe(
                                content, mission_conditions, task_description,
                            )
                        ):
                            from multi_agent.simulation.engine_resolver import resolve_engine_config
                            from multi_agent.simulation.physics_bounds import verify_physics_bounds_text

                            _ec = resolve_engine_config(
                                engine=_sim_pb.engine,
                                octave_path=_sim_pb.octave_path,
                                matlab_path=_sim_pb.matlab_path,
                            )
                            _eng = _ec.engine
                            if _eng in ("matlab_engine", "python", "auto"):
                                _eng = "matlab"
                            _pb_ok, _pb_msg = await verify_physics_bounds_text(
                                content,
                                ny_limit_g=float(
                                    getattr(_sim_pb, "physics_bounds_ny_limit_g", 20.0)
                                ),
                                engine=_eng,
                                octave_path=_ec.octave_path,
                                matlab_path=_ec.matlab_path,
                                timeout_sec=get_physics_bounds_timeout_sec(),
                            )
                            if not _pb_ok:
                                logger.error(
                                    "[MATLAB] MODIFY_LAW physics bounds failed: %s",
                                    _pb_msg,
                                )
                                return {
                                    "status": "error",
                                    "message": _pb_msg,
                                    "physics_bounds": "FAIL",
                                }
                            logger.info("[MATLAB] MODIFY_LAW physics bounds OK: %s", _pb_msg)
                        elif should_skip_physics_bounds_probe(
                            content, mission_conditions, task_description,
                        ):
                            logger.info(
                                "[MATLAB] Skip physics bounds (deterministic T4 minimal APN)"
                            )
                    # Force-patch after LLM generation:
                    # LLM may have reset or omitted these values.
                    if autopilot_params:
                        content = self._patch_autopilot_params(content, autopilot_params)
                    if mission_conditions:
                        content = self._patch_mission_conditions(content, mission_conditions)
                else:
                    # Fallback: original patching logic
                    content = re.sub(r"Nmc\s*=\s*\d+;", f"Nmc = {nmc};", content)
                    if autopilot_params:
                        content = self._patch_autopilot_params(content, autopilot_params)
                    if mission_conditions:
                        content = self._patch_mission_conditions(content, mission_conditions)
                    rewrite_applied = False

            # ── Build a filename that is also a valid MATLAB identifier ──
            # MATLAB requires the function declaration's name to match the
            # file's basename; otherwise it raises a hard parse error on
            # line 1.  We sanitise the user-supplied script_name (so it is
            # a valid identifier prefix) and then rename the function
            # declaration inside the template to match the final basename.
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', script_name or "guidance_simulation")
            if not safe_name or not re.match(r'[a-zA-Z]', safe_name[0]):
                safe_name = f"sim_{safe_name}" if safe_name else "guidance_simulation"
            ts = int(time.time())
            out_basename = f"{safe_name}_{ts}"          # valid identifier
            out_name     = f"{out_basename}.m"
            out_path     = out_dir / out_name

            # Rename the inherited `function <name>` declaration (if any) to
            # match the new file basename.  Robust to optional `[out] = ...`
            # signature prefix (function out = name(args) → keep `out =`).
            func_decl_re = re.compile(
                r"^(\s*function\s+(?:[^=\s]+\s*=\s*)?)([a-zA-Z_]\w*)",
                re.MULTILINE,
            )
            m_decl = func_decl_re.search(content)
            if m_decl and m_decl.group(2) != out_basename:
                old_func = m_decl.group(2)
                content = func_decl_re.sub(
                    rf"\g<1>{out_basename}", content, count=1
                )
                # Also rename any explicit recursive self-calls inside the body
                # (rare, but keeps the script self-consistent).
                content = re.sub(
                    rf"\b{re.escape(old_func)}\s*\(",
                    f"{out_basename}(",
                    content,
                )
                logger.info(
                    f"[MATLAB] Renamed function declaration "
                    f"'{old_func}' → '{out_basename}' to match filename"
                )

            # ── Universal MATLAB script sanitiser ────────────────────────────
            # Handles all known LLM-generated script issues in one pass:
            #   1. Strip ✅/❌/⚠️ annotation lines (write sidecar if any)
            #   2. Inject missing dp guidance fields (N_guidance/R_switch/gama_max_deg)
            #   3. Add "global dp" before dp=struct(...) if missing
            #   4. Add "global dp; dp=sample.dp" to run_single_case_general
            content, _note_lines = _sanitize_matlab_guidance_script(content)
            if _note_lines:
                _sidecar = out_dir / f"{out_basename}_interface_check.txt"
                _sidecar.write_text("\n".join(_note_lines), encoding="utf-8")
                logger.info(
                    f"[MATLAB] Stripped {len(_note_lines)} annotation line(s) from script "
                    f"→ sidecar: {_sidecar.name}"
                )

            out_path.write_text(content, encoding="utf-8")
            try:
                # Also copy to model output dir for traceability.
                # Write to a temp file then rename to avoid WinError 32 when
                # MATLAB has the destination locked.
                model_out = Path(cfg.model_output.generated_dir)
                model_out.mkdir(parents=True, exist_ok=True)
                _dst = model_out / out_name
                _tmp = model_out / f"{out_basename}_tmp_{int(time.time())}.m"
                shutil.copy2(str(out_path), str(_tmp))
                try:
                    _tmp.replace(_dst)
                except OSError:
                    pass  # destination still locked; temp copy remains
            except Exception as exc:
                logger.warning(f"Could not copy script to model output dir: {exc}")

            logger.info(f"[MATLAB] Generated script → {out_path}  (source: {source})")
            self.last_script_path = str(out_path)
            if mode == "MODIFY_LAW":
                self.last_modification_desc = task_description or non_standard_task or ""
            _result = {
                "status":          "success",
                "script_path":     str(out_path),
                "model_copy_path": str(model_out / out_name),
                "template_source": source,
                "mode":            mode,
                "rewrite_applied": rewrite_applied,
                "nmc":             nmc,
            }
            if mode == "MODIFY_LAW":
                _result["modify_law_verified"] = _modify_law_ok
                _result["modify_law_message"] = _modify_law_msg
            return _result
        except Exception as exc:
            logger.error(f"MATLAB generation failed: {exc}")
            return {"status": "error", "message": str(exc)}


class RunSimulationTool:
    name = "run_simulation"
    description = """
    Run guidance system simulation.

    Two modes:
    1. **Direct KB template execution** (REUSE_HISTORY / MODIFY_LAW / TUNE_PARAMS):
       Runs the most-recently-generated .m script (or the KB template if none exists).
       Optionally supply `dp_params` to patch dp.* design parameters and
       `mission_conditions` to select subcases.

    2. **Python simulator** (fallback): provide `navigation_coefficient` and
       `damping_ratio` to run the internal Python simulation.

    **Error handling**: when status="error" the result contains a `stderr` field with
    the full MATLAB error text (e.g. "函数 'gf' 已在此作用域内声明").
    Pass `stderr` as the `error_message` argument to `syntax_check_matlab` so the
    LLM-agent fix loop can read the exact error location and correct the script
    automatically.  Typical flow on error:
      run_simulation → status=error → syntax_check_matlab(script_path, error_message=stderr)
      → re-run run_simulation
    """

    # Canonical KB template path — monte_carlo_single.m is the primary template
    KB_TEMPLATE_PATH: str = str(
        _KB_MATLAB / "guidance" / "monte_carlo_single.m"
    )
    # Legacy path kept for backward compat
    KB_TEMPLATE_PATH_LEGACY: str = str(
        _KB_MATLAB / "robust_analysis" / "MC_gongkuang_simulation_robust_all.m"
    )

    input_schema = {
        "type": "object",
        "properties": {
            "script_path": {
                "type": "string",
                "description": (
                    "Absolute path to an existing MATLAB script to run directly. "
                    "Default KB template: monte_carlo_single.m "
                    f"({str(_KB_MATLAB / 'guidance' / 'monte_carlo_single.m')})"
                ),
            },
            "dp_params": {
                "type": "object",
                "description": (
                    "Optional parameter overrides applied to the RL_PARAMS block "
                    "before execution. Keys: w1, zeta1, w2, zeta2, w3, zeta3, "
                    "N_pn, sw_dist."
                ),
                "additionalProperties": {"type": "number"},
            },
            "mission_conditions": {
                "type": "string",
                "description": (
                    "Mission conditions for monte_carlo_single.m. "
                    "Format: \"RUN_CASE='T';SUB_IDX=0;\" or \"RUN_CASE='ALL';SUB_IDX=0;\" "
                    "Categories: T=目标机动, G=交战几何, AP=驾驶仪退化, R=综合鲁棒"
                ),
            },
            "nmc": {
                "type": "integer",
                "description": "Number of Monte Carlo runs (Nmc) when running a template script.",
                "default": 20,
            },
            "mc_data_path": {
                "type": "string",
                "description": "Optional external MC .mat dataset path.",
            },
            "random_seed": {
                "type": "integer",
                "description": "Optional MATLAB rng seed for a single MC run.",
            },
            "mc_seeds": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Optional list of MATLAB rng seeds. When provided, the "
                    "script is run once per seed and metrics are aggregated "
                    "conservatively."
                ),
            },
            "validate_final": {
                "type": "boolean",
                "description": (
                    "When true, use configured final MC validation seeds if "
                    "mc_seeds is not provided."
                ),
            },
            "navigation_coefficient": {"type": "number"},
            "damping_ratio": {"type": "number"},
            "control_gain": {"type": "number"},
            "target_position": {"type": "array"},
            "duration": {"type": "number", "default": 100.0},
            "dt": {"type": "number", "default": 0.01},
        },
        "required": [],
    }

    def __init__(self, *args, **kwargs):
        self.simulator = None
        self.parameter_experience = None
        # Captures the raw MATLAB stdout from the most recent successful simulation
        # so that callers (e.g. cli_agent Step 3.5) can parse metrics without
        # having to re-run the simulation.
        self.last_stdout: str = ""
        # Pre-parsed metric dict from the most recent successful simulation.
        # Populated by execute() so cli_agent can read without re-parsing stdout.
        self.last_metrics: Dict[str, float] = {}
        self.last_seed_metrics: List[Dict[str, Any]] = []
        # Fallback mission conditions injected from cli_agent each iteration.
        # Used when Hermes omits the mission_conditions argument.
        self._default_mission_conditions: str = ""
        self._default_mc_data_path: str = ""
        super().__init__(*args, **kwargs)

    def set_simulator(self, simulator) -> None:
        self.simulator = simulator

    def set_parameter_experience(self, pe) -> None:
        self.parameter_experience = pe

    def set_mission_conditions(self, conditions_str: str) -> None:
        """Inject user-resolved mission conditions as fallback for Hermes tool calls."""
        self._default_mission_conditions = conditions_str or ""

    def set_mc_data_path(self, mc_data_path: str) -> None:
        """Inject an external MC dataset path for calls that omit it."""
        self._default_mc_data_path = mc_data_path or ""

    @staticmethod
    def _parse_run_case(mission_conditions: Optional[str]) -> Dict[str, Any]:
        """Extract RUN_CASE and SUB_IDX from mission_conditions string.

        Example: "RUN_CASE='T'; SUB_IDX=2;" -> {"RUN_CASE": "T", "SUB_IDX": 2}
        """
        result: Dict[str, Any] = {}
        if not mission_conditions:
            return result
        import re
        m_rc = re.search(r"RUN_CASE\s*=\s*'([^']+)'", mission_conditions)
        if m_rc:
            result["RUN_CASE"] = m_rc.group(1)
        m_si = re.search(r"SUB_IDX\s*=\s*(\d+)", mission_conditions)
        if m_si:
            result["SUB_IDX"] = int(m_si.group(1))
        return result

    async def _find_best_pe_model(
        self, mission_conditions: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Find the best PE model file (saved .m) and its associated parameters.

        Selection criteria:
          1. Task context match — RUN_CASE + SUB_IDX from mission_conditions
          2. Metrics satisfaction — fitness >= 0.7 (high confidence)
          3. Fallback — highest combined_score regardless of fitness

        Returns dict with keys: script_path, dp_params, source, fitness
        or None if no PE model is available.
        """
        import json as _json
        try:
            from ..config_loader import get_config
            pe_base = Path(get_config().model_output.experience_base_dir)
            pe_models = pe_base / "models"
            pe_params = pe_base / "params"
            if not pe_models.exists():
                return None

            # Collect all .m model files from long_term > short_term > flat
            candidates = []
            for subdir in ["long_term", "short_term", ""]:
                d = pe_models / subdir if subdir else pe_models
                if d.exists():
                    candidates.extend(d.glob("*.m"))

            if not candidates:
                return None

            # ── Priority 1: PE structural retrieval with task context ──
            if self.parameter_experience is not None:
                try:
                    # Build task_context with RUN_CASE/SUB_IDX for similarity matching
                    _task_ctx: Dict[str, Any] = {"task": "guidance_rl_optimization"}
                    _rc = self._parse_run_case(mission_conditions)
                    if _rc:
                        _task_ctx.update(_rc)
                    logger.info(
                        f"[run_simulation] PE retrieve_best with context: {_task_ctx}"
                    )

                    pe_results = await self.parameter_experience.retrieve_best(
                        task_context=_task_ctx,
                        top_k=10,
                    )

                    # Load config reward_weights for proper BW/PM ranges
                    _cfg_rw: Dict[str, float] = {}
                    try:
                        _cfg_rw = get_config().rl_optimization.get("reward_weights", {})
                    except Exception:
                        pass

                    # Recompute fitness from objectives using current config ranges
                    # (stored fitness may have been computed with wrong defaults)
                    from ..memory.parameter_experience import ParameterExperience as _PEcls
                    _recompute_kw = {}
                    if _cfg_rw.get("bw_min"):
                        _recompute_kw["bw_min"] = float(_cfg_rw["bw_min"])
                    if _cfg_rw.get("bw_max"):
                        _recompute_kw["bw_max"] = float(_cfg_rw["bw_max"])
                    if _cfg_rw.get("pm_min"):
                        _recompute_kw["pm_min"] = float(_cfg_rw["pm_min"])
                    if _cfg_rw.get("pm_max"):
                        _recompute_kw["pm_max"] = float(_cfg_rw["pm_max"])
                    if _cfg_rw.get("peak_ny_max"):
                        _recompute_kw["peak_ny_max"] = float(_cfg_rw["peak_ny_max"])

                    # Two-pass: first look for satisfied (recomputed fitness >= 0.7)
                    # then fallback to any model file with best combined_score
                    _satisfied: List[Dict[str, Any]] = []
                    _fallback: List[Dict[str, Any]] = []
                    for pe_rec in pe_results:
                        mem_id = pe_rec.get("memory_id", "")
                        if not mem_id:
                            continue
                        prefix = mem_id[:16]
                        matched = next(
                            (c for c in candidates if c.name.startswith(prefix)),
                            None,
                        )
                        if not matched or not matched.exists():
                            continue
                        # Recompute fitness from objectives with correct ranges
                        _objs = pe_rec.get("objectives", {})
                        _refit = _PEcls.compute_fitness_from_objectives(
                            _objs, **_recompute_kw
                        ) if _objs else pe_rec.get("fitness", 0)
                        _entry = {
                            "matched": matched,
                            "pe_rec": pe_rec,
                            "recomputed_fitness": _refit,
                        }
                        if _refit >= 0.7:
                            _satisfied.append(_entry)
                        else:
                            _fallback.append(_entry)

                    # Sort each tier by recomputed fitness (descending)
                    _satisfied.sort(
                        key=lambda e: e["recomputed_fitness"], reverse=True
                    )
                    _fallback.sort(
                        key=lambda e: e["recomputed_fitness"], reverse=True
                    )

                    # Pick the best from satisfied first, then fallback
                    _pick_list = _satisfied if _satisfied else _fallback
                    if _pick_list:
                        best_entry = _pick_list[0]
                        matched = best_entry["matched"]
                        pe_rec = best_entry["pe_rec"]
                        _refit = best_entry["recomputed_fitness"]
                        raw_params = pe_rec.get("parameters", {})
                        dp_params = {
                            (k[3:] if k.startswith("dp_") else k): float(v)
                            for k, v in raw_params.items()
                            if not k.startswith("phys_")
                        }
                        _stored_fit = pe_rec.get('fitness', 0)
                        _sim = pe_rec.get('similarity', 0)
                        _tier = "satisfied" if _satisfied else "fallback"
                        logger.info(
                            f"[run_simulation] PE model found ({_tier}): {matched.name} "
                            f"(refit={_refit:.3f}, stored_fit={_stored_fit:.3f}, "
                            f"sim={_sim:.3f}, "
                            f"Q={pe_rec.get('quality_score', 0):.3f})"
                        )
                        return {
                            "script_path": str(matched),
                            "dp_params": dp_params,
                            "source": f"pe_model:{matched.name}(fit={_refit:.2f},sim={_sim:.2f})",
                            "fitness": _refit,
                        }
                except Exception as exc:
                    logger.warning(f"[run_simulation] PE retrieve_best failed: {exc}")

            # ── Priority 2: Latest model by mtime + load params from JSON ────────
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            best = candidates[0]
            dp_params = {}
            # Try to find matching params JSON
            prefix = best.name.split("_")[0]
            for subdir in ["long_term", "short_term", ""]:
                jp = pe_params / subdir / f"{prefix}.json" if subdir else pe_params / f"{prefix}.json"
                if jp.exists():
                    try:
                        rec = _json.loads(jp.read_text(encoding="utf-8"))
                        raw_params = rec.get("parameters", {})
                        dp_params = {
                            (k[3:] if k.startswith("dp_") else k): float(v)
                            for k, v in raw_params.items()
                            if not k.startswith("phys_")
                        }
                    except Exception:
                        pass
                    break

            logger.info(f"[run_simulation] PE model (latest): {best.name}")
            return {
                "script_path": str(best),
                "dp_params": dp_params,
                "source": f"pe_model:{best.name}(latest)",
                "fitness": 0.0,
            }
        except Exception as exc:
            logger.debug(f"[run_simulation] _find_best_pe_model failed: {exc}")
            return None

    @staticmethod
    def _find_latest_generated_script() -> Optional[str]:
        """Return the most recently modified *valid* .m file from matlab_scripts_dir, or None.

        Skips files that don't pass ``_looks_like_matlab_script`` validation
        (e.g. natural-language notes or parameter dumps that Hermes mistakenly
        saved as .m files).
        """
        try:
            from ..config_loader import get_config
            from ..rl.matlab_rl_optimizer import _looks_like_matlab_script
            scripts_dir = Path(get_config().model_output.matlab_scripts_dir)
            candidates = sorted(
                scripts_dir.glob("*.m"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for cand in candidates:
                try:
                    text = cand.read_text(encoding="utf-8", errors="ignore")
                    is_valid, reason = _looks_like_matlab_script(text)
                    if is_valid:
                        logger.info(
                            f"[run_simulation] No script_path given; "
                            f"using latest Hermes-generated script: {cand.name}"
                        )
                        return str(cand)
                    else:
                        logger.debug(
                            f"[run_simulation] Skipping {cand.name}: {reason}"
                        )
                except Exception:
                    continue
        except Exception as exc:
            logger.debug(f"[run_simulation] Could not scan matlab_scripts_dir: {exc}")
        return None

    async def execute(
        self,
        script_path: Optional[str] = None,
        dp_params: Optional[Dict[str, float]] = None,
        mission_conditions: Optional[str] = None,
        nmc: Optional[int] = None,
        navigation_coefficient: float = 3.5,
        damping_ratio: float = 0.75,
        control_gain: float = 1.0,
        target_position: Optional[List[float]] = None,
        duration: float = 100.0,
        dt: float = 0.01,
        mc_data_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run simulation via Hermes script, KB template, or Python fallback."""
        script_path = _coerce_script_path(script_path, **kwargs) or script_path

        if nmc is None:
            from ..config_loader import get_config
            nmc = int(get_config().matlab_rl_optimizer.nmc_per_eval)

        _mc_seed_values: List[int] = []
        _mc_cfg_seed_values: List[int] = []
        _single_seed_requested = False
        try:
            from ..config_loader import get_config

            _mc_cfg = getattr(get_config().simulation, "mc_validation", None)
            if _mc_cfg is not None and bool(getattr(_mc_cfg, "enabled", False)):
                _mc_cfg_seed_values = _coerce_mc_seed_list(
                    getattr(_mc_cfg, "seeds", [])
                )
                if _mc_validation_auto_runs(getattr(_mc_cfg, "mode", "final")):
                    _mc_seed_values = list(_mc_cfg_seed_values)
        except Exception:
            _mc_seed_values = []
            _mc_cfg_seed_values = []

        _kw_seed_list = None
        for _seed_key in ("mc_seeds", "random_seeds", "seeds"):
            if _seed_key in kwargs:
                _kw_seed_list = kwargs.get(_seed_key)
                break
        if _kw_seed_list is not None:
            _mc_seed_values = _coerce_mc_seed_list(_kw_seed_list)
        _kw_single_seed = kwargs.get("random_seed", kwargs.get("seed", None))
        if _kw_single_seed is not None and _kw_seed_list is None:
            try:
                _mc_seed_values = [int(_kw_single_seed)]
                _single_seed_requested = True
            except (TypeError, ValueError):
                _mc_seed_values = []
        _final_validation_requested = any(
            bool(kwargs.get(k))
            for k in ("validate_final", "final_validation", "mc_validate")
        )
        if (
            _final_validation_requested
            and _kw_seed_list is None
            and _kw_single_seed is None
            and not _mc_seed_values
        ):
            _mc_seed_values = list(_mc_cfg_seed_values)

        _is_kb_template = (
            script_path
            and Path(script_path).resolve() == Path(self.KB_TEMPLATE_PATH).resolve()
        )
        _pe_source = ""

        # ── Resolve script_path ──────────────────────────────────────────────
        # When no script_path given OR caller explicitly passed KB template,
        # try to find a better starting point from historical models.
        #
        # NEW priority (task-context aware):
        #   1. PE model — matched by RUN_CASE + SUB_IDX, prefer fitness >= 0.7
        #   2. Latest valid Hermes-generated script from matlab_scripts_dir
        #   3. Latest valid script from guidance_output/scripts/
        #   4. KB template monte_carlo_single.m
        if not script_path or _is_kb_template:
            # 1. PE model (historical saved .m + associated params, task-context aware)
            _pe = await self._find_best_pe_model(mission_conditions)
            if _pe:
                script_path = _pe["script_path"]
                _pe_source = _pe["source"]
                # Merge PE params as defaults (caller dp_params take priority)
                if _pe["dp_params"]:
                    _merged = dict(_pe["dp_params"])
                    if dp_params:
                        _merged.update(dp_params)
                    dp_params = _merged
                    logger.info(
                        f"[run_simulation] REUSE_HISTORY: using PE model "
                        f"{_pe_source} with {len(dp_params)} params"
                    )
            else:
                # 2. Latest valid Hermes-generated script
                _latest = self._find_latest_generated_script()
                if _latest:
                    script_path = _latest
                else:
                    # 3. Latest valid script from guidance_output/scripts/
                    try:
                        from ..config_loader import get_config
                        from ..rl.matlab_rl_optimizer import _looks_like_matlab_script
                        _scripts_out = Path(get_config().model_output.matlab_scripts_dir).parent / "scripts"
                        if _scripts_out.exists():
                            _scr_cands = sorted(
                                _scripts_out.glob("*.m"),
                                key=lambda p: p.stat().st_mtime,
                                reverse=True,
                            )
                            for _sc in _scr_cands:
                                try:
                                    _txt = _sc.read_text(encoding="utf-8", errors="ignore")
                                    _ok, _ = _looks_like_matlab_script(_txt)
                                    if _ok:
                                        script_path = str(_sc)
                                        _pe_source = f"scripts_folder:{_sc.name}"
                                        logger.info(
                                            f"[run_simulation] Using scripts folder: {_sc.name}"
                                        )
                                        break
                                except Exception:
                                    continue
                    except Exception:
                        pass

                    # 4. KB template fallback
                    if (not script_path or _is_kb_template) and Path(self.KB_TEMPLATE_PATH).exists():
                        script_path = self.KB_TEMPLATE_PATH
                    logger.info(
                            "[run_simulation] No historical model found; "
                            "falling back to KB template: monte_carlo_single.m"
                        )

        _cfg_mc_path = ""
        try:
            from ..config_loader import get_config

            _cfg_mc_path = str(getattr(get_config().simulation, "mc_data_path", "") or "")
        except Exception:
            pass
        _mc_path = (
            mc_data_path
            or kwargs.get("mc_data_path")
            or self._default_mc_data_path
            or _cfg_mc_path
        )
        if _mc_path:
            _external_script = script_path or self.KB_TEMPLATE_PATH
            _external_n_runs = int(
                kwargs.get("external_n_runs")
                or kwargs.get("external_mc_n_runs")
                or 100
            )
            _external_result = run_mc_with_external_data(
                _external_script,
                str(_mc_path),
                n_runs=_external_n_runs,
                kill_radius=float(kwargs.get("kill_radius", 10.0)),
                dp_params=dp_params or {},
            )
            if _external_result.get("success"):
                self.last_seed_metrics = []
                self.last_metrics = {
                    "hit_rate": float(_external_result.get("hit_rate", 0.0)),
                    "SEP": float(_external_result.get("SEP", 0.0)),
                    "miss_distance": float(_external_result.get("SEP", 0.0)),
                    "peak_ny": float(_external_result.get("peak_nz", 0.0)),
                    "peak_n": float(_external_result.get("peak_nz", 0.0)),
                    "pitch_PM": float(_external_result.get("PM", 0.0)),
                    "pitch_BW": float(_external_result.get("BW", 0.0)),
                }
                self.last_stdout = ""
                return {
                    "status": "success",
                    "metrics": {
                        "hit_rate_pct": round(self.last_metrics["hit_rate"], 2),
                        "SEP_m": round(self.last_metrics["SEP"], 3),
                        "PeakNy_g": round(self.last_metrics["peak_ny"], 2),
                        "PM_deg": round(self.last_metrics["pitch_PM"], 2),
                        "BW_rads": round(self.last_metrics["pitch_BW"], 2),
                    },
                    "external_mc_result": _external_result,
                    "script_used": Path(_external_script).name,
                }
            return {
                "status": "error",
                "message": _external_result.get("error") or "external MC failed",
                "external_mc_result": _external_result,
            }

        # ── Path 1: Direct script execution ──────────────────────────────────
        if script_path:
            if _mc_seed_values:
                seed_metrics: List[Dict[str, Any]] = []
                full_stdout_parts: List[str] = []
                last_result: Dict[str, Any] = {}

                for _seed in _mc_seed_values:
                    logger.info(
                        "[run_simulation] MC seed validation run: seed=%s nmc=%s",
                        _seed,
                        nmc,
                    )
                    _seed_result = await self._run_script_directly(
                        script_path,
                        dp_params or {},
                        mission_conditions,
                        nmc,
                        random_seed=_seed,
                    )
                    last_result = dict(_seed_result or {})
                    if (
                        not isinstance(_seed_result, dict)
                        or _seed_result.get("status") != "success"
                    ):
                        self.last_seed_metrics = seed_metrics
                        return {
                            "status": "error",
                            "message": (
                                f"MC seed validation failed at seed={_seed}: "
                                f"{(_seed_result or {}).get('message', 'unknown error')}"
                            ),
                            "failed_seed": _seed,
                            "partial_mc_seed_results": seed_metrics,
                            "script_path": script_path,
                            "stderr": (_seed_result or {}).get("stderr", ""),
                        }

                    _stdout = _seed_result.get(
                        "_full_stdout",
                        _seed_result.get("stdout", ""),
                    )
                    full_stdout_parts.append(f"\n%% MC_SEED={_seed}\n{_stdout}")
                    try:
                        from multi_agent.rl.matlab_rl_optimizer import parse_sim_stdout as _psim
                        from multi_agent.rl.metric_utils import normalize_peak_ny_aliases

                        _metrics = normalize_peak_ny_aliases(_psim(_stdout))
                    except Exception:
                        _metrics = {}
                    _metrics["mc_seed"] = int(_seed)
                    _metrics["random_seed"] = int(_seed)
                    seed_metrics.append(_metrics)

                self.last_stdout = "\n".join(full_stdout_parts)
                self.last_seed_metrics = seed_metrics
                self.last_metrics = _aggregate_mc_seed_metrics(seed_metrics)
                if not _single_seed_requested and len(_mc_seed_values) > 1:
                    logger.info(
                        "[run_simulation] MC seed validation aggregate: "
                        "hit=%.1f SEP=%.3f PeakNy=%.2f seeds=%s",
                        float(self.last_metrics.get("hit_rate", 0.0)),
                        float(self.last_metrics.get("SEP", 0.0)),
                        float(self.last_metrics.get("peak_ny", 0.0)),
                        _mc_seed_values,
                    )

                _result = last_result
                _result["metrics"] = {
                    "hit_rate_pct": round(self.last_metrics.get("hit_rate", 0.0), 2),
                    "SEP_m": round(
                        self.last_metrics.get(
                            "SEP",
                            self.last_metrics.get("miss_distance", 99.0),
                        ),
                        3,
                    ),
                    "PeakNy_g": round(
                        self.last_metrics.get(
                            "peak_ny",
                            self.last_metrics.get("peak_n", 0.0),
                        ),
                        2,
                    ),
                    "PeakNy_max_g": round(
                        self.last_metrics.get(
                            "peak_ny_max",
                            self.last_metrics.get(
                                "peak_n_max",
                                self.last_metrics.get("peak_ny", 0.0),
                            ),
                        ),
                        2,
                    ),
                    "PM_deg": round(self.last_metrics.get("pitch_PM", 0.0), 2),
                    "PM_min_deg": round(self.last_metrics.get("pitch_PM_min", 0.0), 2),
                    "PM_max_deg": round(self.last_metrics.get("pitch_PM_max", 0.0), 2),
                    "BW_rads": round(self.last_metrics.get("pitch_BW", 0.0), 2),
                    "BW_min_rads": round(self.last_metrics.get("pitch_BW_min", 0.0), 2),
                    "BW_max_rads": round(self.last_metrics.get("pitch_BW_max", 0.0), 2),
                    "MC_seed_count": len(seed_metrics),
                }
                _result["mc_seed_results"] = seed_metrics
                _result["mc_seed_values"] = _mc_seed_values
                _result["mc_seed_aggregate"] = dict(self.last_metrics)
                if _pe_source:
                    _result["model_source"] = _pe_source
                _result["script_used"] = Path(script_path).name
                return _result

            _result = await self._run_script_directly(
                script_path, dp_params or {}, mission_conditions, nmc
            )
            # Capture full stdout for parse_sim_stdout (summary is at the end;
            # do NOT truncate here — _run_script_directly already truncates the
            # dict value to 4000 chars for Hermes, which may cut off the summary
            # when nmc is large).
            if isinstance(_result, dict) and _result.get("status") == "success":
                self.last_seed_metrics = []
                self.last_stdout = _result.get("_full_stdout", _result.get("stdout", ""))
                # Pre-parse metrics and inject into the return dict so:
                # a) cli_agent can read last_metrics without re-parsing stdout
                # b) Hermes LLM sees numeric values in the tool response
                try:
                    from multi_agent.rl.matlab_rl_optimizer import parse_sim_stdout as _psim
                    from multi_agent.rl.metric_utils import normalize_peak_ny_aliases
                    self.last_metrics = normalize_peak_ny_aliases(_psim(self.last_stdout))
                    if self.last_metrics:
                        _result["metrics"] = {
                            "hit_rate_pct": round(self.last_metrics.get("hit_rate", 0.0), 2),
                            "SEP_m": round(self.last_metrics.get("SEP", self.last_metrics.get("miss_distance", 99.0)), 3),
                            "PeakNy_g": round(self.last_metrics.get("peak_ny", self.last_metrics.get("peak_n", 0.0)), 2),
                            "PeakNy_max_g": round(
                                self.last_metrics.get(
                                    "peak_ny_max",
                                    self.last_metrics.get(
                                        "peak_n_max",
                                        self.last_metrics.get("peak_ny", 0.0),
                                    ),
                                ),
                                2,
                            ),
                            "PM_deg": round(self.last_metrics.get("pitch_PM", 0.0), 2),
                            "BW_rads": round(self.last_metrics.get("pitch_BW", 0.0), 2),
                        }
                except Exception:
                    pass
                # Annotate which model was actually used
                if _pe_source:
                    _result["model_source"] = _pe_source
                _result["script_used"] = Path(script_path).name
            return _result

        # ── Path 2: Python internal simulator ────────────────────────────────
        try:
            from ..simulation import GuidanceSimulator, GuidanceParameters

            params = GuidanceParameters(
                navigation_coefficient=navigation_coefficient,
                damping_ratio=damping_ratio,
                control_gain=control_gain,
                target_position=target_position or [20000.0, 2000.0, 5000.0],
            )
            sim = GuidanceSimulator()
            results = await sim.generate_and_simulate(
                params=params, duration=duration, dt=dt,
                generate_sysml=False, generate_matlab=False,
            )
            return {"status": "success", "result": results["simulation_result"]}
        except Exception as e:
            logger.error(f"Python simulation failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _run_script_directly(
        self,
        script_path: str,
        dp_params: Dict[str, float],
        mission_conditions: Optional[str],
        nmc: int,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Patch dp.* params + mission flags into *script_path* (in-memory),
        write to a temp file alongside the original, execute via Octave/MATLAB,
        and return parsed metrics.  The original KB template is never modified.
        """
        import subprocess, tempfile, time as _time, re as _re

        path = Path(script_path)
        if not path.exists():
            return {"status": "error", "message": f"Script not found: {script_path}"}

        content = _read_kb_file(path)
        if not content:
            return {"status": "error", "message": f"Could not read script: {script_path}"}

        # Patch N_MC (monte_carlo_single) and Nmc (legacy)
        content = _re.sub(r"\bN_MC\s*=\s*\d+\s*;", f"N_MC = {nmc};", content)
        content = _re.sub(r"\bNmc\s*=\s*\d+\s*;",  f"Nmc = {nmc};",  content)

        if random_seed is not None:
            content = _inject_matlab_rng_seed(content, int(random_seed))

        # Patch dp.* params
        if dp_params:
            content = GenerateMATLABTool._patch_autopilot_params(content, dp_params)

        # Patch mission conditions (use caller arg; fall back to injected default)
        _cond = mission_conditions or self._default_mission_conditions
        if _cond:
            content = GenerateMATLABTool._patch_mission_conditions(content, _cond)

        # Re-run sanitizer on the patched content to catch any residual mixed-end
        # style issues that syntax_check_matlab may have left, or that patching
        # could theoretically reintroduce.  This is the last gate before MATLAB
        # sees the _direct_ variant — keeping it here prevents the recurring
        # "函数 'mst' 已通过 'end' 关闭" parse error on the temp file.
        try:
            _san_content, _san_notes = _sanitize_matlab_guidance_script(content)
            if _san_notes:
                logger.debug(
                    "[run_simulation] _direct_ sanitizer: %d fix(es): %s",
                    len(_san_notes), "; ".join(_san_notes[:5]),
                )
            content = _san_content
        except Exception as _san_exc:
            logger.warning("[run_simulation] _direct_ sanitizer failed: %s", _san_exc)

        # Write temp file (named after the function to satisfy MATLAB/Octave).
        # IMPORTANT: filename basename must be a valid MATLAB identifier when
        # the engine resolves the script as a function.  We append a uuid hex
        # suffix (letter-prefix safe + collision free for parallel callers).
        import uuid as _uuid
        func_match = _re.match(r"^\s*function\s+(?:[^=\s]+\s*=\s*)?([a-zA-Z_]\w*)", content)
        func_name = func_match.group(1) if func_match else None
        tmp_dir = path.parent
        unique_tag = _uuid.uuid4().hex[:8]
        suffix = f"_direct_{unique_tag}"
        if func_name:
            tmp_name = f"{func_name}{suffix}.m"
            content = _re.sub(
                r"^function\s+" + _re.escape(func_name),
                f"function {func_name}{suffix}",
                content, count=1, flags=_re.MULTILINE,
            )
            call_name = f"{func_name}{suffix}"
        else:
            # No leading underscore — keeps basename a valid identifier.
            tmp_name  = f"directrun_{unique_tag}.m"
            call_name = None

        tmp_path = tmp_dir / tmp_name
        try:
            tmp_path.write_text(content, encoding="utf-8")

            # Determine engine from simulator's executor
            executor = getattr(getattr(self.simulator, "executor", None), "__dict__", {})
            engine   = executor.get("engine", "octave")
            oct_path = executor.get("octave_path", "octave")
            mat_path = executor.get("matlab_path", "matlab")
            abs_path = str(tmp_path).replace("\\", "/")
            script_dir = str(tmp_dir).replace("\\", "/")

            # Use shared Octave batch constants: --no-window-system + figure
            # suppression + warning silencing to keep the run_simulation tool
            # call from blowing past the timeout on the KB template's
            # ~30 figure() rendering operations.
            from multi_agent.simulation.guidance_simulator import (
                OCTAVE_BATCH_FLAGS,
                build_octave_eval_string,
            )
            from multi_agent.simulation.sim_timeout import get_matlab_timeout_sec

            _timeout = get_matlab_timeout_sec(nmc)
            # ── Branch 0: in-process matlab.engine ────────────────────────
            # Skips subprocess startup entirely.  Returns the result early
            # rather than going through subprocess.run() below.  Falls
            # through to the subprocess MATLAB branch on backend errors.
            import sys as _sys
            if _sys.version_info >= (3, 13) and engine == "matlab_engine":
                logger.info(
                    "[run_simulation] Skipping matlab_engine (Python %d.%d); "
                    "using subprocess matlab -batch.",
                    _sys.version_info.major, _sys.version_info.minor,
                )
                engine = "matlab"
            if engine == "matlab_engine":
                executor_obj = getattr(self.simulator, "executor", None)
                backend = getattr(executor_obj, "matlab_engine_backend", None)
                if backend is not None and backend.started:
                    # run_script() is synchronous — wrap in executor so we
                    # don't block the event loop.
                    _eng_timeout = float(_timeout)
                    _loop = asyncio.get_event_loop()
                    _eng_ok = True
                    try:
                        stdout, stderr, ok = await asyncio.wait_for(
                            _loop.run_in_executor(
                                None,
                                lambda: backend.run_script(
                                    abs_path,
                                    call_name=call_name,
                                    timeout_sec=_eng_timeout,
                                ),
                            ),
                            timeout=_eng_timeout + 30,  # grace period beyond MATLAB timeout
                        )
                    except (asyncio.CancelledError, KeyboardInterrupt):
                        raise
                    except Exception as _eng_exc:
                        logger.warning(
                            f"matlab.engine failed on '{tmp_name}': "
                            f"{type(_eng_exc).__name__}: {_eng_exc}; "
                            f"falling back to MATLAB subprocess."
                        )
                        engine = "matlab"
                        _eng_ok = False
                    if _eng_ok and ok:
                        logger.info(
                            f"[run_simulation] in-process matlab.engine complete, "
                            f"stdout={len(stdout)} chars"
                        )
                        _s = stdout or ""
                        return {
                            "status":       "success",
                            "script_path":  script_path,
                            "stdout":       _s[-4000:] if len(_s) > 4000 else _s,
                            "_full_stdout": _s,
                            "stderr":       (stderr or "")[:500],
                            "note":         "KB template executed via matlab.engine (in-process)",
                        }
                    if _eng_ok and not ok:
                        # run_script returned but reported failure
                        if not backend.started:
                            logger.warning(
                                f"matlab.engine crashed on '{tmp_name}'; "
                                f"falling back to MATLAB subprocess."
                            )
                            engine = "matlab"
                        else:
                            logger.warning(
                                f"matlab.engine returned error on '{tmp_name}':\n"
                                f"{(stderr or '')[:2000]}"
                            )
                            return {
                                "status":  "error",
                                "message": f"matlab.engine: {(stderr or '')[:200]}",
                                "stderr":  (stderr or "")[:2000],
                            }
                if engine == "matlab_engine":  # backend unavailable from the start
                    logger.info(
                        "matlab.engine backend not started; falling back to "
                        "MATLAB subprocess for this run_simulation call."
                    )
                    engine = "matlab"

            if engine == "octave":
                # source() runs the file in the current scope without trying to
                # resolve its basename as an identifier (avoids parse errors
                # on temp filenames that look like numeric literals).
                eval_str = build_octave_eval_string(abs_path, call_name)
                cmd = [oct_path, *OCTAVE_BATCH_FLAGS, "--eval", eval_str]
            elif engine == "matlab":
                # MATLAB equivalent: -nosplash + -nodesktop + -nodisplay +
                # set DefaultFigureVisible='off' avoids GUI/figure rendering.
                if call_name:
                    eval_str = (
                        f"set(0,'DefaultFigureVisible','off'); "
                        f"addpath('{script_dir}'); {call_name}; exit"
                    )
                else:
                    eval_str = (
                        f"set(0,'DefaultFigureVisible','off'); "
                        f"run('{abs_path}'); exit"
                    )
                cmd = [mat_path, "-nosplash", "-nodesktop", "-nodisplay",
                       "-batch", eval_str]
            else:
                return {"status": "error", "message": f"Unsupported engine: {engine}"}

            # Use Popen + async poll instead of asyncio.create_subprocess_exec.
            # Python 3.13 on Windows has a ProactorEventLoop bug where pipe
            # transports get closed prematurely, causing communicate() to hang
            # indefinitely.  Popen + asyncio.sleep poll yields to the event
            # loop every 2s so KeyboardInterrupt can be delivered promptly.
            _poll_sec = 2.0
            import time as _time_sim
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _t0_sim = _time_sim.monotonic()
            while proc.poll() is None:
                if _time_sim.monotonic() - _t0_sim > _timeout:
                    proc.kill(); proc.wait(timeout=5)
                    raise subprocess.TimeoutExpired(cmd, _timeout)
                await asyncio.sleep(_poll_sec)
            import locale as _loc_sim
            _enc = _loc_sim.getpreferredencoding(False) or "utf-8"
            stdout = (proc.stdout.read() or b"").decode(_enc, errors="replace")
            stderr = (proc.stderr.read() or b"").decode(_enc, errors="replace")

            if proc.returncode != 0:
                logger.warning(
                    f"Script returned code {proc.returncode} on "
                    f"'{tmp_name}'.\n"
                    f"  cmd: {' '.join(cmd)}\n"
                    f"  stderr (up to 2000 chars):\n{(stderr or '').strip()[:2000]}"
                )
                return {
                    "status":      "error",
                    "message":     f"returncode={proc.returncode}",
                    "script_path": script_path,
                    "stderr":      (stderr or "")[:2000],
                }

            logger.info(f"[run_simulation] Direct KB execution complete, stdout={len(stdout)} chars")
            return {
                "status":       "success",
                "script_path":  script_path,
                "stdout":       stdout[-4000:] if len(stdout) > 4000 else stdout,
                "_full_stdout": stdout,
                "stderr":       stderr[:500] if stderr else "",
                "note":         "KB template executed directly — no new script generated",
            }
        except subprocess.TimeoutExpired:
            return {
                "status":  "error",
                "message": (
                    f"Simulation timed out (>{_timeout}s); "
                    f"set MATLAB_TIMEOUT_SEC env var to override"
                ),
            }
        except (asyncio.CancelledError, KeyboardInterrupt):
            try:
                proc.kill(); proc.wait(timeout=5)
            except Exception:
                pass
            raise
        except FileNotFoundError as exc:
            return {"status": "error", "message": f"Engine executable not found: {exc}"}
        except Exception as exc:
            try:
                proc.kill(); proc.wait(timeout=5)
            except Exception:
                pass
            logger.error(f"Direct script execution failed: {exc}")
            return {"status": "error", "message": str(exc)}
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

