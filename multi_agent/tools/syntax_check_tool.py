#!/usr/bin/env python3
"""
SyntaxCheckMATLABTool — Layer 2 Hermes tool.

Validates and auto-fixes a generated monte_carlo_single-style .m file
before it is handed to run_simulation or the RL optimizer.

Checks performed (static):
  1. File existence
  2. Content looks like MATLAB (not LLM prose)
  3. RL_PARAMS_BEGIN / RL_PARAMS_END block present
  4. function declaration matches filename
  5. Unicode / markdown contamination
  6. Variable conflicts: rl_* re-assigned outside block, duplicate function
     definitions, MATLAB builtin name shadowing
  7. Interface consistency: rl_* ↔ base=struct() bridge, called-but-undefined
     local functions
  8. Guidance-law interface (MODIFY_LAW): cg() g.xxx produced vs gs.xxx consumed,
     gf()/cf() output arity vs call-site, p.xxx in cg()/gf() vs base=struct()

LLM-agent fix loop (dynamic):
  When error_message is provided (MATLAB runtime error from run_simulation)
  or when static checks detect unfixable issues, an LLM agent reads the error
  context and produces a complete corrected script.  Iterates up to
  max_fix_attempts times until no issues remain.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from multi_agent.tools.tool_arg_aliases import coerce_script_path as _coerce_script_path

logger = logging.getLogger(__name__)


class SyntaxCheckMATLABTool:
    name = "syntax_check_matlab"
    description = (
        "验证并自动修正生成的 MATLAB .m 脚本文件。"
        "先做静态检查（RL_PARAMS 块、函数名、变量冲突、接口一致性等），"
        "再调用 LLM-agent 修正循环：根据 MATLAB 运行时报错（error_message）"
        "和静态检查结果逐轮修复脚本，直到无错误或达到最大尝试次数。"
        "返回 {status, valid, issues, fixed_path, fix_attempts}。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "script_path": {
                "type": "string",
                "description": "待检查的 .m 文件绝对路径",
            },
            "error_message": {
                "type": "string",
                "description": "run_simulation 返回的 MATLAB 运行时报错文本（可选）；"
                               "提供后 LLM-agent 将据此定位并修复代码错误",
                "default": "",
            },
            "auto_fix": {
                "type": "boolean",
                "description": "是否自动修复并覆盖写回（默认 true）",
                "default": True,
            },
            "max_fix_attempts": {
                "type": "integer",
                "description": "LLM-agent 最大修复迭代次数（默认 3）",
                "default": 3,
            },
            "llm_fix": {
                "type": "boolean",
                "description": "是否对静态告警调用 LLM 修复（默认 true；Layer2 gate 建议 false）",
                "default": True,
            },
        },
        "required": ["script_path"],
    }

    # ── Public entry point ────────────────────────────────────────────────────

    async def execute(
        self,
        script_path: str = "",
        error_message: str = "",
        auto_fix: bool = True,
        max_fix_attempts: int = 3,
        llm_fix: bool = True,
        **kwargs: Any,
    ) -> str:
        script_path = _coerce_script_path(script_path, **kwargs)
        result: Dict[str, Any] = {
            "status": "error",
            "valid": False,
            "issues": [],
            "fixed_path": script_path,
            "fix_attempts": 0,
        }

        if not os.path.isfile(script_path):
            result["issues"].append(f"文件不存在: {script_path}")
            return json.dumps(result, ensure_ascii=False)

        try:
            with open(script_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except Exception as exc:
            result["issues"].append(f"读取文件失败: {exc}")
            return json.dumps(result, ensure_ascii=False)

        from multi_agent.rl.matlab_rl_optimizer import _looks_like_matlab_script

        # ── Phase 1: static sanitizer pass ───────────────────────────────────
        fixed_content = content
        if auto_fix:
            try:
                from multi_agent.tools.simulation_tool import _sanitize_matlab_guidance_script
                fixed_content, _notes = _sanitize_matlab_guidance_script(content)
                if fixed_content != content:
                    with open(script_path, "w", encoding="utf-8") as fh:
                        fh.write(fixed_content)
                    logger.info(
                        "[SyntaxCheck] Sanitizer applied %d fix(es) to '%s'",
                        len(_notes), os.path.basename(script_path),
                    )
            except Exception as _fix_exc:
                logger.warning("[SyntaxCheck] Sanitizer failed: %s", _fix_exc)

        # ── Phase 1.5: deterministic mixed-end style fix ──────────────────
        # Mixed function-end style is 100 % deterministic to fix: find the
        # function-closing 'end' lines in functions that don't belong to the
        # dominant style and delete them.  This avoids the expensive LLM loop
        # for a purely structural MATLAB parse error.
        if auto_fix:
            try:
                _me_fixed, _me_changed = SyntaxCheckMATLABTool._fix_mixed_end_style(
                    fixed_content
                )
                if _me_changed:
                    fixed_content = _me_fixed
                    with open(script_path, "w", encoding="utf-8") as fh:
                        fh.write(fixed_content)
                    logger.info(
                        "[SyntaxCheck] Deterministically fixed mixed function-end style in '%s'",
                        os.path.basename(script_path),
                    )
            except Exception as _me_exc:
                logger.warning("[SyntaxCheck] _fix_mixed_end_style failed: %s", _me_exc)

        # ── Phase 2: static checks ────────────────────────────────────────────
        static_issues, has_rl_params = self._run_static_checks(fixed_content, script_path)
        blocking_static, advisory_static = self._partition_issues(
            static_issues, fixed_content,
        )

        # ── Phase 3: LLM-agent fix loop ───────────────────────────────────────
        # Trigger when: runtime error provided, OR blocking static issues remain.
        fix_attempts = 0
        current_content = fixed_content
        pending_error = error_message.strip()
        pending_blocking = list(blocking_static)
        advisory_issues = list(advisory_static)

        if (
            auto_fix
            and llm_fix
            and max_fix_attempts > 0
            and (pending_error or pending_blocking)
        ):
            for attempt in range(1, max_fix_attempts + 1):
                if not pending_error and not pending_blocking:
                    break
                logger.info(
                    "[SyntaxCheck] LLM fix attempt %d/%d — error=%r issues=%d",
                    attempt, max_fix_attempts,
                    pending_error[:80] if pending_error else "", len(pending_blocking),
                )
                new_content = await self._llm_fix_script(
                    script_path=script_path,
                    content=current_content,
                    error_message=pending_error,
                    static_issues=pending_blocking,
                )
                if not new_content or new_content.strip() == current_content.strip():
                    logger.info("[SyntaxCheck] LLM returned identical content — stopping.")
                    break

                # Validate LLM output before writing
                _ok, _why = _looks_like_matlab_script(new_content)
                if not _ok:
                    logger.warning(
                        "[SyntaxCheck] LLM output not valid MATLAB (%s) — discarding.", _why
                    )
                    break

                current_content = new_content
                with open(script_path, "w", encoding="utf-8") as fh:
                    fh.write(current_content)
                fix_attempts = attempt
                logger.info(
                    "[SyntaxCheck] Wrote LLM-fixed script (%d chars).", len(current_content)
                )

                # Re-run static checks on new content
                _all_issues, has_rl_params = self._run_static_checks(
                    current_content, script_path
                )
                pending_blocking, _new_adv = self._partition_issues(
                    _all_issues, current_content,
                )
                advisory_issues.extend(_new_adv)
                # After a successful fix, clear the original runtime error
                # so the next attempt only re-checks for NEW issues.
                pending_error = ""

                if not pending_blocking:
                    logger.info("[SyntaxCheck] All blocking static issues resolved after LLM fix.")
                    break
        elif advisory_static and not pending_error:
            logger.info(
                "[SyntaxCheck] %d advisory issue(s) only — skipping LLM fix loop",
                len(advisory_static),
            )

        # ── Final verdict ─────────────────────────────────────────────────────
        is_valid_after, _ = _looks_like_matlab_script(current_content)
        # Mixed function-end style is a MATLAB parse error — treat as critical
        # even if _looks_like_matlab_script cannot detect it.
        _has_mixed_end = any(
            "函数结尾风格混用" in iss or "mixed.*end" in iss
            for iss in pending_blocking
        )
        final_valid = is_valid_after and has_rl_params and not _has_mixed_end
        final_issues = (
            pending_blocking + advisory_issues
            if (pending_blocking or advisory_issues)
            else ["脚本检查通过，无问题"]
        )

        result.update({
            "status":              "success" if final_valid else "warning",
            "valid":               final_valid,
            "issues":              final_issues,
            "blocking_issues":     pending_blocking,
            "advisory_issues":     advisory_issues,
            "fixed_path":          script_path,
            "has_rl_params_block": has_rl_params,
            "fix_attempts":        fix_attempts,
        })
        return json.dumps(result, ensure_ascii=False)

    # ── LLM-agent fix helper ──────────────────────────────────────────────────

    @staticmethod
    async def _llm_fix_script(
        script_path: str,
        content: str,
        error_message: str,
        static_issues: List[str],
    ) -> Optional[str]:
        """
        Call LLM to repair a MATLAB script given runtime error + static issues.
        Returns the complete corrected script text, or None on failure.
        """
        try:
            from multi_agent.config_loader import get_config
            import openai as _openai

            cfg = get_config().llm
            client = _openai.AsyncOpenAI(
                api_key=cfg.api_key or "sk-dummy",
                base_url=cfg.base_url or "https://api.openai.com/v1",
                timeout=120,
                max_retries=3,
            )

            # ── Build context: error location + surrounding lines ─────────────
            lines = content.splitlines()
            error_ctx_lines = ""
            # Try multiple error-line formats:
            #   Chinese MATLAB : "行 89" / "行:89"
            #   Octave         : "at line 89 column" / "called from ... at line 89"
            #   MATLAB English : "(line 89)" / "Error in ... (line 89)"
            _line_re = (
                re.search(r"行[:：\s]*(\d+)", error_message)
                or re.search(r"at line (\d+)", error_message)
                or re.search(r"\(line (\d+)\)", error_message)
                or re.search(r"line (\d+)", error_message, re.IGNORECASE)
            )
            if _line_re:
                err_lineno = int(_line_re.group(1))
                lo = max(0, err_lineno - 15)
                hi = min(len(lines), err_lineno + 15)
                numbered = [f"{lo+i+1:4d} | {lines[lo+i]}" for i in range(hi - lo)]
                error_ctx_lines = "\n".join(numbered)

            # ── Function index ─────────────────────────────────────────────────
            func_list = [
                f"  Line {i+1}: {lines[i].strip()}"
                for i, l in enumerate(lines)
                if re.match(r"^\s*function\b", l)
            ]
            func_index = "\n".join(func_list) if func_list else "  (none found)"

            # ── Issues summary ─────────────────────────────────────────────────
            issues_block = ""
            if static_issues:
                issues_block = (
                    "\nSTATIC ANALYSIS ISSUES:\n"
                    + "\n".join(f"  - {iss}" for iss in static_issues[:20])
                )

            # ── Diagnostic hint for known error patterns ───────────────────────
            # Map common MATLAB/Octave runtime error substrings → targeted repair hint.
            _DIAG_HINTS = [
                (
                    ["索引超过", "index.*bound", "index.*exceed", "不能超过"],
                    "INDEX OUT OF BOUNDS: A variable is scalar (size 1) but is indexed with i>1.\n"
                    "Common causes in Monte-Carlo guidance scripts:\n"
                    "  a) Condition arrays (T_conds/G_conds/AP_conds/R_conds) used as\n"
                    "     conds_T(i)/conds_G(i) inside a loop — ensure they are row/col vectors,\n"
                    "     not scalars. Add numel() guard or iterate over actual array length.\n"
                    "  b) Results accumulation: results(i) = val; where results was never\n"
                    "     pre-allocated or was reset to a scalar.\n"
                    "  c) hit_flag(i) / sep_val(i) inside nmc loop when array was initialised\n"
                    "     as zeros(1,1) instead of zeros(1,nmc).\n"
                    "FIX: Replace bare array(i) accesses with safe patterns, or pre-allocate\n"
                    "with zeros(1, num_trials) before the loop.",
                ),
                (
                    ["未定义.*变量", "undefined.*variable", "'\\w+' undefined"],
                    "UNDEFINED VARIABLE: A variable is used before assignment.\n"
                    "Check that all variables are initialised before loops/conditions.",
                ),
                (
                    ["除以零", "division by zero", "Inf", "NaN"],
                    "DIVISION BY ZERO / Inf/NaN: Add a guard: use max(R, 0.1) for missile-target\n"
                    "range, and max(T_go, 0.01) for time-to-go. Protect all denominators.",
                ),
                (
                    ["矩阵维度", "nonconformant", "dimension mismatch", "inner matrix"],
                    "DIMENSION MISMATCH: Ensure vector operations use element-wise operators\n"
                    "(.*, ./, .^) instead of matrix operators (*, /, ^) where applicable.",
                ),
            ]
            diag_hint = ""
            err_lower = error_message.lower()
            for patterns, hint in _DIAG_HINTS:
                if any(re.search(p, err_lower) for p in patterns):
                    diag_hint = hint
                    break

            # ── Prompt ─────────────────────────────────────────────────────────
            # Full script is included so the LLM has complete context.
            # Limit to 12000 chars to stay within token budget.
            script_block = content if len(content) <= 12000 else (
                content[:6000] + "\n...[ middle truncated ]...\n" + content[-6000:]
            )

            prompt = (
                "You are a MATLAB script repair agent. "
                "Fix ALL errors in the script below and return the COMPLETE corrected script.\n"
                "Rules:\n"
                "  1. Return ONLY the raw MATLAB code — no markdown fences, no explanation.\n"
                "  2. Fix the EXACT error(s) reported; do NOT change unrelated logic.\n"
                "  3. For duplicate function definitions: keep the LAST definition "
                "(it is the newest version) and REMOVE earlier duplicate(s).\n"
                "  4. Preserve the RL_PARAMS_BEGIN / RL_PARAMS_END block exactly.\n"
                "  5. The output must be a complete, runnable .m file.\n"
                "  6. MIXED FUNCTION-END STYLE: MATLAB requires ALL functions in a file to use\n"
                "     the SAME style — either ALL closed with 'end', or NONE.\n"
                "     If any function already uses 'end', add closing 'end' to every function\n"
                "     that is missing one (after its last statement, before the next 'function').\n\n"
            )
            if error_message:
                prompt += f"MATLAB RUNTIME ERROR:\n{error_message.strip()}\n\n"
            if diag_hint:
                prompt += f"DIAGNOSTIC GUIDANCE:\n{diag_hint}\n\n"
            if error_ctx_lines:
                prompt += f"CODE CONTEXT (around error line):\n{error_ctx_lines}\n\n"
            prompt += f"ALL function definitions:\n{func_index}\n"
            prompt += issues_block
            prompt += f"\nFULL SCRIPT:\n{script_block}\n"
            prompt += "\nReturn the complete corrected MATLAB script now:"

            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8192,
            )
            # P2-1 户部：记一次真实用量（未装载计量时 record() 是纯 no-op）
            from multi_agent import cost_meter as _cost
            _cost.record(resp, model=cfg.model, label="syntax_check.llm_fix")
            raw = resp.choices[0].message.content or ""

            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            return raw.strip()

        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.warning("[SyntaxCheck] LLM fix call failed: %s", exc)
            return None

    # ── Static check runner ───────────────────────────────────────────────────

    def _run_static_checks(
        self, content: str, script_path: str
    ) -> Tuple[List[str], bool]:
        """Run all static checks; return (issues, has_rl_params)."""
        from multi_agent.rl.matlab_rl_optimizer import _looks_like_matlab_script
        issues: List[str] = []

        is_valid, reason = _looks_like_matlab_script(content)
        if not is_valid:
            issues.append(f"内容不像 MATLAB 脚本: {reason}")

        has_rl_params = bool(re.search(r"RL_PARAMS_BEGIN", content))
        if not has_rl_params:
            issues.append("缺少 RL_PARAMS_BEGIN 注入块（RL 优化器无法 patch 参数）")

        func_match = re.search(r"^function\s+(\w+)", content, re.MULTILINE)
        if func_match:
            func_name = func_match.group(1)
            file_base = os.path.splitext(os.path.basename(script_path))[0]
            if func_name != file_base:
                issues.append(
                    f"函数名 '{func_name}' 与文件名 '{file_base}' 不一致"
                )

        if "```" in content:
            issues.append("含有 Markdown 代码围栏 (```)，需清除")
        if any(c in content for c in ("×", "÷", "≥", "≤", "≠")):
            issues.append("含有 Unicode 数学运算符，需替换为 MATLAB 等价符")

        issues.extend(self._check_variable_conflicts(content))
        issues.extend(self._check_interface_consistency(content))
        issues.extend(self._check_guidance_interface(content))
        issues.extend(self._check_mixed_end_style(content))
        issues.extend(self._check_common_runtime_patterns(content))

        return issues, has_rl_params

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _check_common_runtime_patterns(content: str) -> list:
        """Detect static patterns that commonly cause MATLAB runtime errors."""
        issues = []
        if re.search(r"\bendfunction\b", content, re.IGNORECASE):
            issues.append(
                "含 Octave 语法 endfunction（MATLAB 应使用 end）；"
                "运行前需替换或通过 syntax_check 自动修正"
            )
        if (
            "Placeholder simulation function" in content
            or "sim_s(p) %#ok<INUSD>" in content
        ):
            issues.append(
                "sim_s 为 PLACEHOLDER 占位实现，RL/MATLAB 仿真无效；"
                "请以 monte_carlo_single.m 的 sim_s 替换"
            )
        lines = content.splitlines()

        # Locate nmc loop body (for i = 1:nmc or for i_mc = ...)
        nmc_loop_re = re.compile(r"^\s*for\s+(\w+)\s*=\s*1\s*:\s*(\w+)", re.IGNORECASE)
        nmc_end_re  = re.compile(r"^\s*end\b")
        loop_var: Optional[str] = None
        loop_depth = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("%"):
                continue

            m = nmc_loop_re.match(line)
            if m and m.group(2) in ("nmc", "N_mc", "n_mc", "num_mc", "num_trials"):
                loop_var = m.group(1)
                loop_depth = 1
                continue

            if loop_var:
                # track nesting depth
                if re.match(r"^\s*(for|while|if|switch)\b", line):
                    loop_depth += 1
                elif nmc_end_re.match(line):
                    loop_depth -= 1
                    if loop_depth == 0:
                        loop_var = None
                        continue

                # Detect plain scalar init followed by indexed assignment inside loop
                # Pattern: results(i) = ... where results was never pre-allocated
                idx_assign = re.findall(r"(\w+)\s*\(\s*" + re.escape(loop_var or "i") + r"\s*\)\s*=", line)
                for var in idx_assign:
                    # Check if var is pre-allocated with zeros/NaN outside the loop
                    prealloc_re = re.compile(
                        rf"\b{re.escape(var)}\s*=\s*(zeros|NaN|nan|ones)\s*\(",
                    )
                    if not any(prealloc_re.search(l) for l in lines[:i]):
                        issues.append(
                            f"第 {i+1} 行: '{var}({loop_var})' 在 nmc 循环中被索引赋值，"
                            f"但未检测到预分配（zeros/NaN/ones）——可能导致索引越界错误"
                        )

        return issues

    @staticmethod
    def _check_variable_conflicts(content: str) -> list:
        """Detect rl_* redefinition outside RL_PARAMS, duplicate functions, builtin shadowing."""
        issues = []
        lines = content.splitlines()

        # Locate RL_PARAMS block
        rl_begin, rl_end = -1, -1
        for i, line in enumerate(lines):
            if "RL_PARAMS_BEGIN" in line:
                rl_begin = i
            elif "RL_PARAMS_END" in line:
                rl_end = i
                break

        # Extract declared rl_* variables
        rl_vars: dict = {}
        if 0 <= rl_begin < rl_end:
            for i in range(rl_begin + 1, rl_end):
                m = re.match(r"^\s*(rl_\w+)\s*=", lines[i])
                if m:
                    rl_vars[m.group(1)] = i

        # rl_* reassigned outside block
        for i, line in enumerate(lines):
            if rl_begin <= i <= rl_end:
                continue
            if line.strip().startswith("%"):
                continue
            for var, decl_line in rl_vars.items():
                if re.search(rf"\b{re.escape(var)}\s*=(?!=)", line):
                    issues.append(
                        f"第 {i + 1} 行: RL 参数 '{var}' 在 RL_PARAMS 块外被重新赋值"
                        f"（将覆盖 RL 优化器注入值，原声明在第 {decl_line + 1} 行）"
                    )

        # Duplicate local function definitions
        func_defs: dict = {}
        for i, line in enumerate(lines):
            m = re.match(r"^\s*function\s+(?:[\w,\[\]\s]+=\s*)?(\w+)\s*\(", line)
            if m:
                name = m.group(1)
                if name in func_defs:
                    issues.append(
                        f"第 {i + 1} 行: 本地函数 '{name}' 重复定义"
                        f"（首次在第 {func_defs[name] + 1} 行）"
                    )
                else:
                    func_defs[name] = i

        # MATLAB builtin name shadowing
        risky_builtins = {
            "length", "size", "sum", "mean", "std", "pi", "inf", "nan",
            "eps", "zeros", "ones", "eye", "rand", "randn", "mod", "rem",
        }
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("%"):
                continue
            if re.match(r"^\s*function\s", line):
                continue
            for b in risky_builtins:
                if re.search(rf"(?<!\w){re.escape(b)}\s*=(?!=)", line):
                    issues.append(
                        f"第 {i + 1} 行: 变量名 '{b}' 与 MATLAB 内建函数同名（可能导致遮蔽）"
                    )

        return issues

    @staticmethod
    def _strip_matlab_comments(content: str) -> str:
        lines: List[str] = []
        for line in content.splitlines():
            if "%" in line:
                line = line.split("%", 1)[0]
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _is_advisory_issue(issue: str, content: str) -> bool:
        """Non-blocking static findings (RL runtime-patched fields, etc.)."""
        if "已声明但未注入" in issue and "rl_ny_lim" in issue:
            if re.search(r"\bny_lim\s*=", content):
                return True
        if "已声明但未注入" in issue and "rl_gama_max_deg" in issue:
            if re.search(r"\bgama_max_deg\b", content):
                return True
        return False

    @classmethod
    def _partition_issues(
        cls, issues: List[str], content: str,
    ) -> Tuple[List[str], List[str]]:
        blocking: List[str] = []
        advisory: List[str] = []
        for iss in issues:
            if cls._is_advisory_issue(iss, content):
                advisory.append(iss)
            else:
                blocking.append(iss)
        return blocking, advisory

    @staticmethod
    def _check_interface_consistency(content: str) -> list:
        """Check rl_*↔base=struct() bridge and called-but-undefined local functions."""
        issues = []
        lines = content.splitlines()

        # ── RL_PARAMS declared variables ──────────────────────────────────────
        rl_begin, rl_end = -1, -1
        for i, line in enumerate(lines):
            if "RL_PARAMS_BEGIN" in line:
                rl_begin = i
            elif "RL_PARAMS_END" in line:
                rl_end = i
                break

        rl_declared: dict = {}
        if 0 <= rl_begin < rl_end:
            for i in range(rl_begin + 1, rl_end):
                m = re.match(r"^\s*(rl_\w+)\s*=", lines[i])
                if m:
                    rl_declared[m.group(1)] = i

        # ── base=struct(...) block ─────────────────────────────────────────────
        base_match = re.search(
            r"\bbase\s*=\s*struct\s*\((.+?)\)\s*;", content, re.DOTALL
        )
        rl_used_in_base: set = set()
        if base_match:
            rl_used_in_base = set(re.findall(r"\b(rl_\w+)\b", base_match.group(1)))

        if rl_declared:
            # Declared but not injected
            for var in sorted(set(rl_declared) - rl_used_in_base):
                issues.append(
                    f"第 {rl_declared[var] + 1} 行: RL 参数 '{var}' 已声明但未注入"
                    f" base=struct()（RL 优化器调整此参数将不起作用）"
                )
            # Used in base but not declared in block
            for var in sorted(rl_used_in_base - set(rl_declared)):
                issues.append(
                    f"base=struct() 中引用了 '{var}' 但 RL_PARAMS 块中未声明"
                    f"（运行时未定义变量错误）"
                )

        # ── Called-but-undefined local functions ──────────────────────────────
        defined_funcs: set = set(
            re.findall(
                r"^\s*function\s+(?:[\w,\[\]\s]+=\s*)?(\w+)\s*\(",
                content,
                re.MULTILINE,
            )
        )
        matlab_builtins = {
            "if", "for", "while", "switch", "case", "otherwise", "try", "catch", "end",
            "return", "break", "continue",
            "fprintf", "sprintf", "disp", "error", "warning", "assert",
            "numel", "length", "size", "fieldnames", "isfield", "isnan", "isinf",
            "isempty", "isreal", "ischar", "isnumeric", "islogical", "iscell",
            "nargin", "nargout", "varargin", "varargout",
            "abs", "sqrt", "exp", "log", "log2", "log10", "sin", "cos", "tan",
            "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
            "max", "min", "sum", "mean", "std", "var", "quantile", "sort",
            "norm", "cross", "dot", "det", "inv", "pinv", "eig", "svd",
            "zeros", "ones", "eye", "cell", "struct", "nan", "inf", "pi", "true", "false",
            "linspace", "logspace", "colon", "horzcat", "vertcat",
            "plot", "histogram", "scatter", "bar", "figure", "subplot",
            "xlabel", "ylabel", "title", "legend", "grid", "axis",
            "hold", "xline", "yline", "sgtitle", "lines", "close", "saveas", "print",
            "strcat", "num2str", "int2str", "str2num", "str2double",
            "strcmpi", "strcmp", "strsplit", "strtrim", "upper", "lower",
            "feedback", "margin", "bandwidth", "tf", "roots", "poly", "conv",
            "real", "imag", "conj", "angle", "unwrap",
            "randn", "rand", "rng", "interp1", "interp2", "griddata",
            "repmat", "reshape", "transpose", "ctranspose",
            "find", "any", "all", "unique", "intersect", "union", "ismember",
            "logical", "double", "single", "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64", "char",
            "mod", "rem", "floor", "ceil", "round", "fix", "sign",
            "cellfun", "arrayfun", "structfun",
            "clc", "clear", "input", "pause", "tic", "toc", "clock",
            "ode45", "ode23", "ode15s", "deval",
            "feval", "fmincon", "fsolve", "lsqnonlin",
        }
        # Strip MATLAB single-quoted strings and comments before extracting calls.
        _code_only = SyntaxCheckMATLABTool._strip_matlab_comments(content)
        _code_only = re.sub(r"'(?:[^'\n]|'')*'", "''", _code_only)
        called_funcs: set = set(re.findall(r"\b([a-zA-Z_]\w*)\s*\(", _code_only))
        # Identifiers that appear on the LHS of an assignment are variables (not functions).
        # Both "var = ..." and "var(i) = ..." patterns are collected.
        assigned_vars: set = set(
            re.findall(r"\b([a-zA-Z_]\w*)\s*(?:\([^)\n]*\))?\s*=(?!=)", content)
        )
        undefined_calls = {
            f for f in called_funcs
            if f not in matlab_builtins
            and f not in defined_funcs
            and f not in assigned_vars
            and not f.startswith("rl_")
            and not f.startswith("RL_")
            and len(f) > 2
        }
        if undefined_calls:
            issues.append(
                f"以下函数被调用但未在本文件中找到本地定义"
                f"（若为外部文件请确认文件存在）: {sorted(undefined_calls)}"
            )

        return issues

    @staticmethod
    def _check_guidance_interface(content: str) -> list:
        """
        Check guidance-law interface consistency — critical after MODIFY_LAW.

        A. cg(p) output fields (g.xxx) vs consumer reads (gs.xxx in cf/sim_s)
        B. gf() return-value count vs call-site expected count
        C. cf() return-value count vs call-site expected count
        D. p.xxx fields accessed in cg()/gf() but absent from base=struct()
        """
        issues = []

        def _func_body(name: str) -> str:
            m = re.search(
                rf"(?m)^function\s+.*?\b{re.escape(name)}\b[^\n]*\n([\s\S]*?)(?=^function|\Z)",
                content,
            )
            return m.group(1) if m else ""

        def _func_span(name: str):
            m = re.search(
                rf"(?m)^function\s+.*?\b{re.escape(name)}\b[^\n]*\n[\s\S]*?(?=^function|\Z)",
                content,
            )
            return (m.start(), m.end()) if m else None

        # ── A. cg() output fields ↔ gs.xxx consumer reads ────────────────────
        cg_body = _func_body("cg")
        if cg_body:
            g_written: set = set(re.findall(r"\bg\.(\w+)\s*=(?!=)", cg_body))

            cg_span = _func_span("cg")
            if cg_span:
                outside = content[: cg_span[0]] + content[cg_span[1] :]
            else:
                outside = content
            gs_read: set = set(re.findall(r"\bgs\.(\w+)\b", outside))

            unused = g_written - gs_read
            if unused:
                issues.append(
                    f"cg() 输出字段 {sorted(unused)} 已写入但从未被 cf()/sim_s() 读取"
                    f"（制导律修改后存在废弃接口）"
                )

            missing = gs_read - g_written
            if missing:
                issues.append(
                    f"cf()/sim_s() 读取了 gs.{sorted(missing)} 但 cg() 未输出该字段"
                    f"（MODIFY_LAW 后接口缺失，运行时将崩溃）"
                )

        # ── B. gf() output arity ─────────────────────────────────────────────
        gf_def = re.search(r"function\s+\[([^\]]+)\]\s*=\s*gf\s*\(", content)
        gf_call = re.search(r"\[([^\]]+)\]\s*=\s*gf\s*\(", content)
        if gf_def and gf_call:
            def_outs = [x.strip() for x in gf_def.group(1).split(",")]
            call_outs = [x.strip() for x in gf_call.group(1).split(",")]
            if len(def_outs) != len(call_outs):
                issues.append(
                    f"gf() 定义返回 {len(def_outs)} 个值 {def_outs}，"
                    f"但调用处接收 {len(call_outs)} 个 {call_outs}"
                    f"（输出数量不匹配）"
                )

        # ── C. cf() output arity ──────────────────────────────────────────────
        cf_def = re.search(r"function\s+\[([^\]]+)\]\s*=\s*cf\s*\(", content)
        cf_call = re.search(r"\[([^\]]+)\]\s*=\s*cf\s*\(", content)
        if cf_def and cf_call:
            def_outs = [x.strip() for x in cf_def.group(1).split(",")]
            call_outs = [x.strip() for x in cf_call.group(1).split(",")]
            if len(def_outs) != len(call_outs):
                issues.append(
                    f"cf() 定义返回 {len(def_outs)} 个值，"
                    f"但调用处接收 {len(call_outs)} 个"
                    f"（输出数量不匹配）"
                )

        # ── D. p.xxx in cg()/gf() vs base=struct() ────────────────────────────
        base_m = re.search(r"\bbase\s*=\s*struct\s*\((.+?)\)\s*;", content, re.DOTALL)
        base_fields: set = set()
        if base_m:
            base_fields = set(re.findall(r"'(\w+)'\s*,", base_m.group(1)))

        if base_fields:
            for func_name in ("cg", "gf"):
                body = _func_body(func_name)
                if not body:
                    continue
                p_refs: set = set(re.findall(r"\bp\.(\w+)\b", body))
                undefined_refs = p_refs - base_fields
                if undefined_refs:
                    issues.append(
                        f"{func_name}() 访问了 p.{sorted(undefined_refs)} "
                        f"但 base=struct() 中未定义这些字段"
                        f"（MODIFY_LAW 新增参数未同步到 base 或 RL_PARAMS 块）"
                    )

        return issues

    @staticmethod
    def _fix_mixed_end_style(content: str) -> "Tuple[str, bool]":
        """Deterministically fix mixed function-end style.

        Determines the dominant end-style (majority of functions with or
        without a function-closing 'end') and normalises all functions to
        match.  Currently only implements the strip direction (dominant =
        no-end, minority has extra end) which covers the LLM-adds-end-to-gf
        pattern observed in practice.

        Returns (fixed_content, was_changed).
        """
        lines = content.splitlines(keepends=True)
        n = len(lines)

        func_indices = [
            i for i in range(n) if re.match(r'^\s*function\b', lines[i])
        ]
        if len(func_indices) < 2:
            return content, False

        _OPEN_RE = re.compile(
            r'^\s*(if|for|while|switch|try|parfor)\b', re.IGNORECASE
        )
        _END_RE  = re.compile(r'^\s*end\b', re.IGNORECASE)
        _SKIP_RE = re.compile(
            r'^\s*(else|elseif|case|otherwise|catch)\b', re.IGNORECASE
        )

        # (has_func_end: bool, closing_line_idx: int|-1)
        func_info: List[Tuple[bool, int]] = []

        for idx, fi in enumerate(func_indices):
            next_fi = func_indices[idx + 1] if idx + 1 < len(func_indices) else n
            depth = 0
            closing_end_line = -1

            for li in range(fi + 1, next_fi):
                for part in re.split(r';', lines[li]):
                    p = part.strip()
                    if not p or p.startswith('%'):
                        continue
                    if _SKIP_RE.match(p):
                        continue
                    if _OPEN_RE.match(p):
                        depth += 1
                    elif _END_RE.match(p):
                        if depth == 0:
                            closing_end_line = li  # track last depth-0 end
                        else:
                            depth -= 1

            func_info.append((closing_end_line >= 0, closing_end_line))

        n_with    = sum(1 for has, _ in func_info if has)
        n_without = len(func_info) - n_with

        if n_with == 0 or n_without == 0:
            return content, False  # already consistent

        if n_without < n_with:
            # Dominant style: WITH end — would need to add ends to minority.
            # Not implemented here; fall through to LLM fix.
            return content, False

        # Dominant style: WITHOUT end — strip function-closing 'end' from
        # the minority of functions that have one.
        to_strip = sorted(
            [cel for has, cel in func_info if has and cel >= 0],
            reverse=True,  # process bottom-up so indices stay valid
        )
        if not to_strip:
            return content, False

        lines_list = list(lines)
        stripped = 0
        for cel in to_strip:
            raw = lines_list[cel].rstrip()
            if re.match(r'^\s*end\b', raw):
                del lines_list[cel]
                stripped += 1

        if not stripped:
            return content, False

        logger.debug(
            "[SyntaxCheck] _fix_mixed_end_style: stripped %d function-closing 'end'(s)",
            stripped,
        )
        return "".join(lines_list), True

    @staticmethod
    def _check_mixed_end_style(content: str) -> list:
        """
        Detect mixed function-end style (some local functions closed with 'end',
        others not) — MATLAB raises a parse error in this case.

        Uses the same depth-tracking algorithm as _sanitize_matlab_guidance_script
        step 12 to correctly distinguish function-closing 'end' from control-flow
        'end' keywords.  Reports a critical issue if mixed style is found after
        the sanitizer has already run.
        """
        issues = []
        lines = content.splitlines()
        n = len(lines)

        func_indices = [i for i in range(n) if re.match(r'^\s*function\b', lines[i])]
        if len(func_indices) < 2:
            return issues

        _fname_re = re.compile(
            r'^\s*function\s+(?:[^=\s]+\s*=\s*)?([a-zA-Z_]\w*)\s*[\(\s]'
        )
        _OPEN_RE = re.compile(r'^\s*(if|for|while|switch|try|parfor)\b', re.IGNORECASE)
        _END_RE  = re.compile(r'^\s*end\b', re.IGNORECASE)
        _SKIP_RE = re.compile(r'^\s*(else|elseif|case|otherwise|catch)\b', re.IGNORECASE)

        func_end_flags: list = []   # True = has function-closing end
        func_names:     list = []

        for idx, fi in enumerate(func_indices):
            fname = ""
            m = _fname_re.match(lines[fi])
            if m:
                fname = m.group(1)
            next_fi = func_indices[idx + 1] if idx + 1 < len(func_indices) else n
            func_names.append(fname)

            # Count function-closing ends using depth tracking
            depth = 0
            func_ends = 0

            # Handle one-liner patterns on the declaration line itself
            decl = lines[fi]
            d_open = sum(1 for p in re.split(r';', decl)
                         if _OPEN_RE.match(p.strip()) and p.strip())
            d_end  = sum(1 for p in re.split(r';', decl)
                         if _END_RE.match(p.strip()) and p.strip())
            if d_end > d_open:
                func_ends += d_end - d_open

            for li in range(fi + 1, next_fi):
                for part in re.split(r';', lines[li]):
                    p = part.strip()
                    if not p or p.startswith('%'):
                        continue
                    if _SKIP_RE.match(p):
                        continue
                    if _OPEN_RE.match(p):
                        depth += 1
                    elif _END_RE.match(p):
                        if depth == 0:
                            func_ends += 1
                        else:
                            depth -= 1

            func_end_flags.append(func_ends > 0)

        any_with    = any(func_end_flags)
        any_without = not all(func_end_flags)

        if any_with and any_without:
            # All functions in monte_carlo_single.m have explicit 'end' —
            # verified against template source.  No known exceptions.
            _KNOWN_NO_END = frozenset()
            unexpected_no_end = [
                func_names[i] or f"line{func_indices[i]+1}"
                for i in range(len(func_indices))
                if not func_end_flags[i] and func_names[i] not in _KNOWN_NO_END
            ]
            if unexpected_no_end:
                with_end = [func_names[i] or f"line{func_indices[i]+1}"
                            for i in range(len(func_indices)) if func_end_flags[i]]
                issues.append(
                    f"函数结尾风格混用（MATLAB 解析错误）："
                    f"有 'end' 的函数: {with_end[:6]}；"
                    f"非预期无 'end' 的函数: {unexpected_no_end[:6]}。"
                    f"需统一为全部使用 'end' 关闭（推荐：为缺少 'end' 的函数补充关闭 end）"
                )

        return issues
