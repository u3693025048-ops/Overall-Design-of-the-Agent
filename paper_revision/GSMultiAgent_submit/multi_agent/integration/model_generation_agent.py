#!/usr/bin/env python3
"""
Model Generation Agent
LLM-powered dynamic generation of MATLAB and SysML guidance-system models.

Routing logic
-------------
TUNE_PARAMS / REUSE_HISTORY
    Pure template-patching: no LLM call needed, fast and deterministic.
MODIFY_LAW
    LLM extracts the guidance-law function body from the template, rewrites
    it according to the task description, then patches it back.  Parameter
    values are patched afterwards with the normal regex-based logic.
Non-standard working conditions (no A-F mapping)
    LLM receives the task description and maps the physical scenario to the
    closest A-F category flags, enabling the standard template to execute.
SysML customisation
    LLM adapts the retrieved XML template to task-specific architecture
    requirements (parameter values, component descriptions, etc.).
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Unicode characters that mark LLM verification/annotation lines
_VERIFY_PREFIXES: tuple = (
    '\u2705',  # ✅
    '\u274c',  # ❌
    '\u26a0',  # ⚠
    '\u2714',  # ✔
    '\u2716',  # ✖
    '\U0001f50d',  # 🔍
    '\U0001f4dd',  # 📝
)


def _strip_verification_lines(text: str) -> Tuple[str, str]:
    """Separate pure MATLAB lines from LLM annotation lines (✅ / ❌ …).

    Returns
    -------
    (clean_code, notes)
      clean_code : lines that belong in the .m file
      notes      : annotation lines to write to the sidecar file
    """
    code, notes = [], []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped and stripped[0] in _VERIFY_PREFIXES:
            notes.append(line)
        else:
            code.append(line)
    return '\n'.join(code), '\n'.join(notes)


# ── Template function inventory helper ──────────────────────────────────────────
_FUNC_DECL_RE = re.compile(
    r"(?m)^function\s+(?:[^=\n]*=\s*)?([a-zA-Z_]\w*)\s*\(",
)
_END_CLOSING_RE = re.compile(r"(?m)^\s*end\s*(%[^\n]*)?\s*$")


def _build_func_inventory(script: str) -> str:
    """
    Scan *script* and return a compact inventory string describing every
    subfunction: its name, whether it has an explicit closing ``end``, and
    whether the LLM may rewrite it.

    Example output::

        脚本中现有函数（函数名, 是否有末尾end, 是否允许修改）：
          monte_carlo_single  → 无末尾end  [不可修改]
          run_category        → 有末尾end  [不可修改]
          gf                  → 无末尾end  [可修改制导律]
          cf                  → 无末尾end  [不可修改]
          ...
    """
    lines = script.splitlines()
    n = len(lines)
    decls = [
        (m.start(), m.group(1))
        for m in _FUNC_DECL_RE.finditer(script)
    ]
    if not decls:
        return ""
    rows = []
    for i, (pos, fname) in enumerate(decls):
        start_line = script[:pos].count("\n")
        end_line   = decls[i + 1][0] if i + 1 < len(decls) else len(script)
        end_line   = script[:end_line].count("\n")
        span_lines = lines[start_line: end_line]
        # Count bare control-flow openers vs ends to detect function-closing end
        openers = sum(
            1 for ln in span_lines
            if re.match(r"\s*(if|for|while|switch|try|parfor)\b", ln)
        )
        ends = sum(
            1 for ln in span_lines[1:]
            if re.match(r"\s*end\b", ln)
        )
        has_end = ends > openers
        editable = fname in ("gf", "guidance_local", "guidance_func")
        rows.append(
            f"  {fname:<24s} → {'有末尾end' if has_end else '无末尾end'}  "
            f"{'[可修改制导律]' if editable else '[不可修改]'}"
        )
    header = "脚本中现有函数（函数名, 是否有末尾end, 是否允许修改）：\n"
    return header + "\n".join(rows)


# Functions the LLM may rewrite during MODIFY_LAW; every other function
# in the template must be preserved verbatim (body + interface + variable names).
_EDITABLE_FUNCS: frozenset = frozenset({
    "gf", "guidance_local", "guidance_func", "guidance_calc",
})

_FUNC_STYLE_CONSTRAINT = (
    "\n【函数结构约束 — 必须严格遵守】\n"
    "1. 【唯一允许修改的函数】制导律函数（gf / guidance_local / guidance_func）的函数体。\n"
    "2. 【绝对禁止修改】以下函数的代码必须与模板完全一致，一字不差：\n"
    "   monte_carlo_single, run_category, sim_s, cg, cf, af, df, rk4f,\n"
    "   print_table, print_summary, plot_results, mst, anoise\n"
    "3. 制导律函数的接口（函数名、输入参数列表、输出变量列表）必须与模板完全一致。\n"
    "4. 所有函数内部变量名必须与模板保持一致，不得重命名现有变量。\n"
    "5. 每个函数是否有末尾 end 必须与上表完全一致：有就保留，无就不加。\n"
    "6. if/for/while/switch/try 等控制流块的 end 不受此约束，必须保留。\n"
    "7. 不要添加上表之外的任何新函数（如dump_metrics/print_result/output_data等）。\n"
    "8. 不要在任何非制导律函数末尾新增 fprintf/disp 输出语句。\n"
)


class ModelGenerationAgent:
    """
    LLM-powered model generation agent.

    Uses the same LLM backend as ReflectionAgent (configured via config.yaml).
    Call ``generate_matlab()`` or ``generate_sysml()`` from the corresponding
    Hermes tools.
    """

    def __init__(self):
        from ..config_loader import get_config
        from langchain_openai import ChatOpenAI

        cfg = get_config().llm
        extra_body = {"include_reasoning": True}
        self.llm = ChatOpenAI(
            model=cfg.model,
            openai_api_key=cfg.api_key or "sk-dummy",
            openai_api_base=cfg.base_url,
            temperature=0.1,
            extra_body=extra_body,
        )

    # ── public API ────────────────────────────────────────────────────────────

    async def generate_matlab(
        self,
        task_description: str,
        template_content: str,
        mode: str = "TUNE_PARAMS",
        autopilot_params: Optional[Dict[str, float]] = None,
        mission_conditions: Optional[str] = None,
        nmc: int = 100,
        non_standard_task: str = "",
        seed_source: str = "",
    ) -> str:
        """
        Generate / modify a MATLAB simulation script.

        Parameters
        ----------
        task_description : natural-language task (used by LLM in MODIFY_LAW)
        template_content : raw MATLAB template string (already retrieved from KB)
        mode             : "TUNE_PARAMS" | "REUSE_HISTORY" | "MODIFY_LAW"
        autopilot_params : {param_name: value} overrides for dp.* fields
        mission_conditions : pre-built conditions string (run_X=…;sub_X=…;)
        nmc              : Monte Carlo run count
        non_standard_task: free-text description of non-standard working
                           conditions (used when mission_conditions is None)

        Returns
        -------
        Modified MATLAB script as a string (not written to disk here).
        """
        script = template_content

        # Always update the Monte Carlo count
        script = re.sub(r"Nmc\s*=\s*\d+;", f"Nmc = {nmc};", script)

        if mode == "MODIFY_LAW":
            from multi_agent.integration.t4_low_risk import (
                apply_t4_deterministic_gf_if_enabled,
                clamp_autopilot_params,
            )
            from multi_agent.integration.script_seed_policy import (
                should_apply_deterministic_apn_on_seed,
            )

            if should_apply_deterministic_apn_on_seed(
                mission_conditions=mission_conditions,
                task_prompt=task_description,
                seed_source=seed_source,
                mode=mode,
            ):
                script, _t4_apn = apply_t4_deterministic_gf_if_enabled(
                    script, mission_conditions, task_description,
                )
                if _t4_apn:
                    _ap = clamp_autopilot_params(autopilot_params)
                    script = self._patch_autopilot_params(script, _ap)
                    if mission_conditions:
                        script = self._patch_mission_conditions(script, mission_conditions)
                    logger.info(
                        "[ModelGen] T4 low-risk: deterministic minimal APN gf() "
                        "applied on KB seed (LLM skipped)"
                    )
                    return script

            script = await self._rewrite_guidance_law(
                script, task_description, autopilot_params, mission_conditions,
                original_template=template_content,
            )
        else:
            from multi_agent.integration.t4_low_risk import (
                apply_t4_deterministic_gf_if_enabled,
                clamp_autopilot_params,
            )
            from multi_agent.integration.script_seed_policy import (
                should_apply_deterministic_apn_on_seed,
            )

            if should_apply_deterministic_apn_on_seed(
                mission_conditions=mission_conditions,
                task_prompt=task_description,
                seed_source=seed_source,
                mode=mode,
            ):
                script, _t4_apn = apply_t4_deterministic_gf_if_enabled(
                    script, mission_conditions, task_description,
                )
            else:
                _t4_apn = False
            if _t4_apn:
                logger.info(
                    "[ModelGen] T4 low-risk: deterministic minimal APN gf() applied (mode=%s)",
                    mode,
                )
            _ap = clamp_autopilot_params(autopilot_params) if _t4_apn else autopilot_params
            if _ap:
                script = self._patch_autopilot_params(script, _ap)
            elif autopilot_params:
                script = self._patch_autopilot_params(script, autopilot_params)
            if mission_conditions:
                script = self._patch_mission_conditions(script, mission_conditions)

        # Non-standard working conditions: LLM generates parameter values directly
        # (does NOT map to A-F flags — caller controls run_X via mission_conditions)
        if non_standard_task:
            script = await self._generate_custom_conditions(script, non_standard_task)

        return script

    async def generate_sysml(
        self,
        task_description: str,
        template: str,
        diagram_type: str,
        substitutions: Dict[str, str],
    ) -> str:
        """
        Generate a SysML XML diagram.

        For simple tasks: fills ``{placeholder}`` slots in the template.
        For tasks with specific architectural requirements: LLM customises the
        XML to match the task description before returning.

        Parameters
        ----------
        task_description : natural-language task description
        template         : raw XML template string
        diagram_type     : "bdd" | "parametric" | "ibd"
        substitutions    : {placeholder_key: value} for basic fill

        Returns
        -------
        Customised XML string.
        """
        result = template
        for key, val in substitutions.items():
            result = result.replace(f"{{{key}}}", str(val))

        if task_description and len(task_description) > 30:
            adapted = await self._customize_sysml(result, task_description, diagram_type)
            if adapted and len(adapted) > 100:
                result = adapted

        return result

    # ── LLM-based guidance-function locator ────────────────────────────────────

    async def _llm_locate_guidance_function(self, script: str):
        """Ask the LLM to identify which function in *script* implements the
        guidance law.  Returns ``(section_text, func_name)`` on success, or
        ``('', '')`` if the LLM cannot identify a suitable function.

        Strategy
        --------
        1. Extract every ``function`` declaration from the script (name + first
           3 lines of body) — avoids sending the whole script to the LLM.
        2. Ask the LLM to return just the function NAME.
        3. Extract and return the full body of that function.
        """
        # Step 1: build a condensed function index
        func_summaries = []
        for m in re.finditer(
            r"(?m)^(function\s+(?:[^\n=]*=\s*)?([a-zA-Z_]\w*)\s*\([^\n]*\n(?:[^\n]*\n){0,3})",
            script,
        ):
            fname  = m.group(2)
            header = m.group(1)[:300].replace("\n", " | ")
            func_summaries.append(f"  {fname}: {header}")

        if not func_summaries:
            return "", ""

        func_list = "\n".join(func_summaries[:20])  # cap at 20 functions
        prompt = (
            "以下是一个MATLAB制导系统仿真脚本中的所有子函数列表（函数名 + 前几行代码）。\n"
            "请判断哪个函数实现了\"制导律\"（即计算导弹加速度指令或导引头视线角速率的函数），"
            "只返回该函数名，不要任何解释。\n\n"
            f"{func_list}\n\n"
            "制导律函数名（只返回一个标识符，如 gf 或 guidance_calc）："
        )
        response = (await self._llm_call(prompt)).strip()
        # Clean: keep only the first valid MATLAB identifier
        match = re.match(r"[a-zA-Z_]\w*", response.strip())
        if not match:
            logger.debug(
                "[ModelGen] LLM locator returned non-identifier: %r", response[:80]
            )
            return "", ""

        func_name = match.group(0)

        # Step 2: extract the function body by name
        body_m = re.search(
            rf"(?m)^(function\s+(?:[^\n=]*=\s*)?{re.escape(func_name)}\s*\([^\n]*\n[\s\S]*?)(?=^function|\Z)",
            script,
        )
        if body_m:
            return body_m.group(1)[:3000], func_name

        logger.debug(
            "[ModelGen] LLM locator named '%s' but body not found in script.",
            func_name,
        )
        return "", ""

    # ── MODIFY_LAW helpers ────────────────────────────────────────────────────

    async def _rewrite_guidance_law(
        self,
        script: str,
        task_description: str,
        autopilot_params: Optional[Dict],
        mission_conditions: Optional[str],
        original_template: Optional[str] = None,
    ) -> str:
        """
        LLM rewrites the guidance-law function body, then patches params.

        Extraction strategy (first match wins):
        1. Named ``guidance_local`` (original template)
        2. Any function whose name contains "guidance" or "nav"
        3. Any function body that references ``N_guidance`` or LOS-rate terms
        4. Fallback: ask LLM to rewrite the full script (guidance section only)
        """
        guidance_section, func_name = self._extract_guidance_section_robust(script)

        if guidance_section and func_name:
            # ── targeted function rewrite ──────────────────────────────────
            _inv = _build_func_inventory(script)
            prompt = (
                "你是MATLAB制导律设计专家。请严格按照以下约束修改制导律函数代码。\n\n"
                f"任务需求: {task_description}\n\n"
                f"{_inv}\n"
                f"{_FUNC_STYLE_CONSTRAINT}\n"
                f"当前制导律函数名: {func_name}\n"
                f"当前制导律代码（此函数体可修改）:\n```matlab\n{guidance_section[:2500]}\n```\n\n"
                "严格要求（违反将导致仿真失败）:\n"
                "1. 只允许修改上面给出的制导律函数体内部逻辑\n"
                "2. 函数签名（function [N1,N2,G1,G2,gc]=gf(Tx,Ty,Tz,XK,TVx,TVy,TVz,Np) 等）"
                "   必须与原函数完全一致，输入/输出参数名、顺序不变\n"
                "3. 输出变量 N1,N2,G1,G2,gc 的语义和单位（g单位过载）必须与原模板一致\n"
                "4. 直接返回修改后的完整制导律函数代码，不加任何说明文字或 markdown 格式\n"
                "5. 不要输出其他函数（sim_s/cf/cg等），只输出制导律函数本身\n"
            )
            new_section = await self._llm_call(prompt)
            new_section = re.sub(r"^```(?:matlab)?\s*\n?", "", new_section.strip(), flags=re.IGNORECASE)
            new_section = re.sub(r"\n?```\s*$", "", new_section.strip())
            new_section, _sec_notes = _strip_verification_lines(new_section)
            if _sec_notes:
                logger.debug(
                    "[ModelGen] Stripped %d verification line(s) from LLM guidance section.",
                    len(_sec_notes.splitlines()),
                )

            if new_section and self._is_valid_matlab(new_section):
                script = self._replace_guidance_section_by_name(script, func_name, new_section)
                logger.info(
                    f"[ModelGen] MODIFY_LAW: guidance-law function '{func_name}' "
                    "successfully rewritten and embedded."
                )
            else:
                logger.error(
                    f"[ModelGen] MODIFY_LAW: LLM output for '{func_name}' failed MATLAB "
                    "validation — falling back to full-script rewrite."
                )
                script = await self._full_script_rewrite(
                    script, task_description, func_name,
                    original_template=original_template,
                )
        else:
            # ── Pattern miss: try LLM-based locator before full-script rewrite ──
            logger.warning(
                "[ModelGen] MODIFY_LAW: regex patterns missed — trying LLM "
                "guidance-function locator."
            )
            guidance_section, func_name = await self._llm_locate_guidance_function(script)
            if guidance_section and func_name:
                logger.info(
                    "[ModelGen] MODIFY_LAW: LLM located guidance function '%s'.",
                    func_name,
                )
                # Reuse the targeted rewrite path with LLM-identified section
                _inv2 = _build_func_inventory(script)
                prompt = (
                    "你是MATLAB制导律设计专家。请严格按照以下约束修改制导律函数代码。\n\n"
                    f"任务需求: {task_description}\n\n"
                    f"{_inv2}\n"
                    f"{_FUNC_STYLE_CONSTRAINT}\n"
                    f"当前制导律函数名: {func_name}\n"
                    f"当前制导律代码（此函数体可修改）:\n```matlab\n{guidance_section[:2500]}\n```\n\n"
                    "严格要求:\n"
                    "1. 只修改制导律算法逻辑，不修改自动驾驶仪传递函数结构\n"
                    "2. 函数签名（函数名、输入/输出参数名、顺序）必须与原函数完全一致\n"
                    "3. 输出变量 N1,N2,G1,G2,gc 语义和单位必须与模板一致\n"
                    "4. 直接返回修改后的完整制导律函数代码，不加任何说明文字或 markdown 格式\n"
                    "5. 不要输出其他函数，只输出制导律函数本身\n"
                )
                new_section = await self._llm_call(prompt)
                new_section = re.sub(r"^```(?:matlab)?\s*\n?", "", new_section.strip(), flags=re.IGNORECASE)
                new_section = re.sub(r"\n?```\s*$", "", new_section.strip())
                new_section, _ = _strip_verification_lines(new_section)
                if new_section and self._is_valid_matlab(new_section):
                    script = self._replace_guidance_section_by_name(script, func_name, new_section)
                    logger.info(
                        "[ModelGen] MODIFY_LAW: LLM-located function '%s' rewritten.",
                        func_name,
                    )
                else:
                    logger.warning(
                        "[ModelGen] MODIFY_LAW: LLM rewrite of '%s' failed validation "
                        "— falling back to full-script rewrite.",
                        func_name,
                    )
                    script = await self._full_script_rewrite(
                        script, task_description, func_name,
                        original_template=original_template,
                    )
            else:
                logger.warning(
                    "[ModelGen] MODIFY_LAW: LLM locator also failed — "
                    "requesting full-script rewrite."
                )
                script = await self._full_script_rewrite(
                    script, task_description, "",
                    original_template=original_template,
                )

        # Always patch params after guidance rewrite
        if autopilot_params:
            script = self._patch_autopilot_params(script, autopilot_params)
        if mission_conditions:
            script = self._patch_mission_conditions(script, mission_conditions)

        # ── Post-generation safety net ────────────────────────────────────────
        # Restore any non-editable function that the LLM may have drifted from
        # the template, regardless of which code path was taken above.
        if original_template:
            script, _restored = self._restore_template_functions(
                script, original_template
            )
            if _restored:
                logger.info(
                    "[ModelGen] _restore_template_functions: reset %d non-editable "
                    "function(s) to template verbatim: %s",
                    len(_restored), _restored,
                )

        return script

    async def _full_script_rewrite(
        self,
        script: str,
        task_description: str,
        hint_func: str,
        original_template: Optional[str] = None,
    ) -> str:
        """
        Ask the LLM to identify and rewrite the guidance section in the full
        script.  Used when targeted extraction fails.
        """
        hint = f"函数名参考: {hint_func}" if hint_func else "请自行定位制导律计算部分"
        _inv3 = _build_func_inventory(script)
        prompt = (
            "你是MATLAB制导律设计专家。以下是一个完整的制导系统仿真脚本。\n"
            "请根据任务需求，定位并【只修改】制导律函数（gf/guidance_local）的内部逻辑，其余所有函数代码保持与模板完全一致。\n\n"
            f"任务需求: {task_description}\n"
            f"{hint}\n\n"
            f"{_inv3}\n"
            f"{_FUNC_STYLE_CONSTRAINT}\n"
            f"完整脚本（前3500字符）:\n```matlab\n{script[:3500]}\n```\n\n"
            "严格要求:\n"
            "1. 只修改制导律函数体内部算法逻辑，绝对不修改其他任何函数\n"
            "2. sim_s / cf / cg / af / df / rk4f / run_category 等函数必须与输入脚本完全一致\n"
            "3. 制导律函数的输入/输出参数签名必须与原函数完全一致\n"
            "4. 返回修改后的完整MATLAB脚本，不加说明或 markdown 格式\n"
        )
        result = await self._llm_call(prompt)
        result = re.sub(r"^```(?:matlab)?\s*\n?", "", result.strip(), flags=re.IGNORECASE)
        result = re.sub(r"\n?```\s*$", "", result.strip())
        result, _res_notes = _strip_verification_lines(result)
        if _res_notes:
            logger.debug(
                "[ModelGen] Stripped %d verification line(s) from full-script rewrite.",
                len(_res_notes.splitlines()),
            )
        if result and self._is_valid_matlab(result) and len(result) > 500:
            # Restore any non-editable functions that LLM may have drifted
            if original_template:
                result, _restored = self._restore_template_functions(result, original_template)
                if _restored:
                    logger.info(
                        "[ModelGen] full-script rewrite: restored %d function(s): %s",
                        len(_restored), _restored,
                    )
            logger.info("[ModelGen] MODIFY_LAW: full-script rewrite applied.")
            return result
        logger.error(
            "[ModelGen] MODIFY_LAW: full-script rewrite also failed validation; "
            "script is unchanged — check LLM output."
        )
        return script

    # ── Template-fidelity enforcement ─────────────────────────────────────────

    def _restore_template_functions(
        self,
        script: str,
        template: str,
        editable_funcs: Optional[frozenset] = None,
    ) -> Tuple[str, List[str]]:
        """
        Post-generation safety net for MODIFY_LAW.

        Scans every function in *template*.  For each function that is NOT in
        *editable_funcs*, its body in *script* is replaced with the verbatim
        template body — regardless of what the LLM generated.

        This guarantees sim_s / cf / cg / af / df / rk4f / run_category etc.
        remain letter-perfect copies of the template even when the LLM
        accidentally rewrites them.

        Parameters
        ----------
        script          : generated/modified script
        template        : original template string (passed through unchanged)
        editable_funcs  : set of function names the LLM is allowed to rewrite;
                          defaults to module-level ``_EDITABLE_FUNCS``

        Returns
        -------
        (patched_script, list_of_restored_function_names)
        """
        if editable_funcs is None:
            editable_funcs = _EDITABLE_FUNCS

        # Regex that captures one complete function block (up to next function or EOF)
        _FBLOCK = re.compile(
            r"(?m)^(function\s+(?:[^\n=]*=\s*)?([a-zA-Z_]\w*)\s*\([^\n]*\n"
            r"[\s\S]*?)(?=^function\b|\Z)"
        )

        # Build {fname: verbatim_block} from template
        tmpl_funcs: Dict[str, str] = {}
        for m in _FBLOCK.finditer(template):
            tmpl_funcs[m.group(2)] = m.group(1)

        restored: List[str] = []
        result = script

        for fname, tmpl_body in tmpl_funcs.items():
            if fname in editable_funcs:
                continue  # LLM is allowed to rewrite this function

            # Locate the same function in the generated script
            pat = re.compile(
                rf"(?m)^function\s+(?:[^\n=]*=\s*)?{re.escape(fname)}\s*\([^\n]*\n"
                r"[\s\S]*?(?=^function\b|\Z)"
            )
            m2 = pat.search(result)
            if m2 is None:
                continue  # function absent in generated script — leave as-is

            gen_body = m2.group(0)
            if gen_body.rstrip() == tmpl_body.rstrip():
                continue  # already identical

            # Replace with verbatim template body
            result = pat.sub(lambda _, _tb=tmpl_body: _tb, result, count=1)
            restored.append(fname)

        # ── Second pass: normalize editable functions' end-style ──────────────
        # After restoring non-editable functions the script has a consistent
        # end-style (matching the template).  If the LLM-generated guidance
        # function uses a different style (e.g. adds 'end' when template
        # doesn't, or omits it when template does), MATLAB will raise a
        # "mixed function-end style" parse error.  Fix it deterministically.
        for fname, tmpl_body in tmpl_funcs.items():
            if fname not in editable_funcs:
                continue
            pat = re.compile(
                rf"(?m)^function\s+(?:[^\n=]*=\s*)?{re.escape(fname)}\s*\([^\n]*\n"
                r"[\s\S]*?(?=^function\b|\Z)"
            )
            m2 = pat.search(result)
            if m2 is None:
                continue
            gen_body = m2.group(0)
            tmpl_has_end = self._body_has_func_end(tmpl_body)
            gen_has_end  = self._body_has_func_end(gen_body)
            if tmpl_has_end == gen_has_end:
                continue  # already consistent
            if not tmpl_has_end and gen_has_end:
                # Template style: no closing end — strip the extra end
                new_body = self._strip_last_end(gen_body)
                result = pat.sub(lambda _, _nb=new_body: _nb, result, count=1)
                restored.append(f"{fname}(-end)")
                logger.info(
                    "[ModelGen] _restore_template_functions: stripped extra 'end' "
                    "from editable func '%s' to match template style.", fname
                )
            else:
                # Template style: has closing end — add missing end
                new_body = gen_body.rstrip() + "\nend\n"
                result = pat.sub(lambda _, _nb=new_body: _nb, result, count=1)
                restored.append(f"{fname}(+end)")
                logger.info(
                    "[ModelGen] _restore_template_functions: added missing 'end' "
                    "to editable func '%s' to match template style.", fname
                )

        return result, restored

    @staticmethod
    def _body_has_func_end(func_body: str) -> bool:
        """Return True if the last non-blank non-comment line of *func_body* is 'end'."""
        for line in reversed(func_body.rstrip().splitlines()):
            s = line.strip()
            if not s or s.startswith('%'):
                continue
            return bool(re.match(r'^end\b', s))
        return False

    @staticmethod
    def _strip_last_end(func_body: str) -> str:
        """Remove the last standalone 'end' line from a function body.

        Only removes the line when the last non-blank non-comment line IS
        'end'.  Safe to call even when no such line exists (returns unchanged).
        """
        lines = func_body.rstrip('\n').split('\n')
        for i in range(len(lines) - 1, -1, -1):
            s = lines[i].strip()
            if not s or s.startswith('%'):
                continue
            if re.match(r'^end\b', s):
                del lines[i]
                # Trim any blank lines left at the end
                while lines and not lines[-1].strip():
                    lines.pop()
                return '\n'.join(lines) + '\n'
            break  # last code line is not 'end' — nothing to strip
        return func_body

    # ── non-standard condition: direct free-text → parameter generation ────────

    async def _generate_custom_conditions(
        self, script: str, non_standard_task: str
    ) -> str:
        """
        Generate MATLAB parameter values directly from a free-text working-condition
        description.  Does NOT map to A-F category flags.

        1. Extracts the parameter/initialisation block from the template.
        2. LLM outputs only the assignment lines that need to change
           (format: ``var = value;``).
        3. Each assignment is patched into the script with a targeted regex;
           only variables that already exist in the template are touched.
        4. A-F ``run_X`` / ``sub_X`` flags are intentionally NOT modified here.
        """
        param_section = self._extract_param_section(script)

        prompt = (
            "你是MATLAB仿真专家。根据以下自由文本工况描述，确定需要修改的仿真参数值。\n\n"
            f"任务工况（自由文本，无需映射到A-F类别）: {non_standard_task}\n\n"
            f"当前参数初始化代码块:\n{param_section[:2500]}\n\n"
            "要求:\n"
            "1. 只输出需要修改的参数赋值语句，每行一条，格式严格为: 变量名 = 值;\n"
            "   例如: dp.H0 = 5000;  或  dp.V0 = 300;\n"
            "2. 不要修改 run_A/run_B/sub_A 等工况开关\n"
            "3. 只修改与任务描述直接相关的参数，不相关参数不输出\n"
            "4. 不加任何说明文字或 markdown 格式\n"
        )

        assignments_text = await self._llm_call(prompt)
        if not assignments_text:
            return script

        applied = 0
        for raw_line in assignments_text.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("%") or line.startswith("#") or line.startswith("```"):
                continue
            m = re.match(r"^([\w.]+)\s*=\s*(.+?)\s*;?\s*$", line)
            if not m:
                continue
            var, val = m.group(1), m.group(2).rstrip(";")
            # Only patch variables that already exist in the script
            if re.search(rf"{re.escape(var)}\s*=", script):
                script = re.sub(
                    rf"({re.escape(var)}\s*=\s*)[^;\n]+;",
                    f"{var} = {val};",
                    script,
                    count=1,
                )
                applied += 1

        logger.info(
            f"[ModelGen] Non-standard conditions: applied {applied} "
            "parameter override(s) from free-text description."
        )
        return script

    # ── SysML LLM customisation ───────────────────────────────────────────────

    async def _customize_sysml(
        self, xml_content: str, task_description: str, diagram_type: str
    ) -> str:
        """LLM adapts SysML XML to task-specific architecture requirements."""
        prompt = (
            f"你是SysML制导系统建模专家。请根据任务需求调整以下 SysML {diagram_type.upper()} 图。\n\n"
            f"任务需求: {task_description}\n\n"
            f"当前模板（前1500字符）:\n{xml_content[:1500]}\n\n"
            "要求:\n"
            "1. 保持XML结构完全有效\n"
            "2. 调整与任务需求相关的参数值和组件/端口描述\n"
            "3. 直接返回完整XML，不加任何说明文字或 markdown 格式\n"
        )
        result = await self._llm_call(prompt)
        result = re.sub(r"^```(?:xml)?\s*\n?", "", result.strip(), flags=re.IGNORECASE)
        result = re.sub(r"\n?```\s*$", "", result.strip())
        return result

    # ── Syntax-error repair ───────────────────────────────────────────────────

    async def repair_matlab(self, content: str, error_msg: str) -> str:
        """Ask the LLM to fix a MATLAB/Octave syntax error in *content*.

        Parameters
        ----------
        content   : the MATLAB source code that failed to execute
        error_msg : the stderr output from Octave/MATLAB (≤1500 chars)

        Returns the repaired source, or *content* unchanged if LLM fails.
        """
        prompt = (
            "你是MATLAB语法修复专家。以下MATLAB脚本在Octave中运行时出现了语法错误，"
            "请修复脚本中的语法问题。\n\n"
            f"错误信息:\n{error_msg}\n\n"
            f"需要修复的脚本（前4000字符）:\n```matlab\n{content[:4000]}\n```\n\n"
            "修复要求:\n"
            "1. 只修复导致错误的语法问题，不改变算法逻辑\n"
            "2. 常见问题: struct(, 多余逗号、全角标点符号、Unicode数学符号、"
               "dp变量命名冲突、缺少global声明\n"
            "3. 直接返回完整的修复后MATLAB脚本，不加任何说明或markdown格式\n"
            "4. 【禁止】不要添加脚本中原本不存在的新函数（如dump_metrics/print_result等）\n"
            "5. 【禁止】不要修改已有函数的末尾end状态（有end的保留，无end的不加）\n"
        )
        try:
            result = await self._llm_call(prompt)
            result = re.sub(r"^```(?:matlab)?\s*\n?", "", result.strip(), flags=re.IGNORECASE)
            result = re.sub(r"\n?```\s*$", "", result.strip())
            result, _notes = _strip_verification_lines(result)
            if result and self._is_valid_matlab(result) and len(result) > 200:
                logger.info("[ModelGen] repair_matlab: repaired script accepted.")
                return result
            logger.warning("[ModelGen] repair_matlab: LLM output failed validation; returning original.")
        except Exception as exc:
            logger.warning(f"[ModelGen] repair_matlab error: {exc}")
        return content

    # ── LLM call helper ───────────────────────────────────────────────────────

    async def _llm_call(self, prompt: str) -> str:
        """Invoke LLM and return the text response."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content="你是制导系统工程专家，擅长MATLAB编程和SysML建模。"),
                HumanMessage(content=prompt),
            ]
            response = await self.llm.ainvoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            return content.strip()
        except Exception as exc:
            logger.warning(f"[ModelGen] LLM call failed: {exc}")
            return ""

    # ── static patching helpers (mirrors GenerateMATLABTool) ─────────────────

    # ── Inline-guidance keyword sets ──────────────────────────────────────────
    # Used by Pattern 8 to detect guidance math embedded inside a sim function.
    _INLINE_GUIDANCE_VARS = re.compile(
        r"\b(?:ay_cmd|az_cmd|a_cmd_y|a_cmd_z|N_y|N_z|N_lat|N_lon|Nz_cmd|Ny_cmd"
        r"|lam_dot|lambda_dot|sigma_dot_los|Vc\s*=|Vr\s*="
        r"|N_pn\s*\*|Np\s*\*|N_PN\s*\*"
        r"|制导律|guidance\s+law|guidance_law)",
        re.IGNORECASE,
    )
    # Functions that ARE the simulation loop, not the guidance law itself —
    # only match these for inline detection (Pattern 8).
    _SIM_FUNC_NAMES = re.compile(
        r"\b(?:sim_engagement|sim_s|simulate_engagement|run_engagement"
        r"|sim_loop|run_sim|simulate|doSim|mc_run)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _extract_guidance_section_robust(script: str):
        """Locate the guidance-law section in a MATLAB script.

        Returns ``(section_text, function_name)``.  Tries patterns in priority
        order; returns ``("", "")`` only when all fail (triggering full-script
        LLM rewrite).

        Patterns cover three structural styles:
        ① **Separate guidance subfunction** — dedicated function with a
           recognisable name/signature (Patterns 1-7).
        ② **Inline guidance** — guidance math embedded inside the main
           simulation function, no dedicated subfunction (Pattern 8).
        ③ **Fallback** — full-script LLM rewrite (caller's responsibility).

        Pattern priority (first match wins):
          1  Exact name ``guidance_local``  (monte_carlo_single template)
          2  Function name contains "guidance" / ends with "_nav"
          3  Function body references ``dp.N_guidance``
          4  Function body contains LOS-rate keywords (los_rate / dq_ / …)
          5  Function returns navigation commands: [N1,N2,…] / [ay_cmd,az_cmd,…]
          6  Short guidance-function name: gf / gc / glaw / pn_law  (Chinese
             simulation convention — handles ``[outputs]=gf(…)`` correctly)
          7  Function body contains PN-math ``Np*Vc`` / ``N_pn*Vc``
          8  Inline: guidance variables computed inside a known simulation
             function (HAOG / OGL / inline-style scripts)
        """
        # ── Pattern 1: exact name guidance_local ──────────────────────────────
        m = re.search(
            r"(?m)^(function[^\n]*?\b(guidance_local)\b[^\n]*\n[\s\S]*?)(?=^function|\Z)",
            script,
        )
        if m:
            return m.group(1)[:3000], m.group(2)

        # ── Pattern 2: function name contains "guidance" / ends with "_nav" ───
        # Requires at least one argument ``([^)]+)`` so zero-argument entry
        # functions (e.g. ``function guidance_simulation_G3_xxx()``) are
        # excluded — those are the main script wrapper, not the guidance law.
        m = re.search(
            r"(?m)^(function[^\n]*?\b([\w]*(?:guidance|_nav)[\w]*)\b[^\n]*\([^)]+\)[^\n]*\n[\s\S]*?)(?=^function|\Z)",
            script,
            re.IGNORECASE,
        )
        if m:
            return m.group(1)[:3000], m.group(2)

        # ── Pattern 3: body references dp.N_guidance ──────────────────────────
        m = re.search(
            r"(?m)^(function\s+(?:[^\n=]*=\s*)?([a-zA-Z_]\w*)\s*\([^\n]*\n[\s\S]{0,2000}?dp\.N_guidance[\s\S]*?)(?=^function|\Z)",
            script,
        )
        if m:
            return m.group(1)[:3000], m.group(2)

        # ── Pattern 4: body contains LOS-rate keywords ────────────────────────
        m = re.search(
            r"(?m)^(function\s+(?:[^\n=]*=\s*)?([a-zA-Z_]\w*)\s*\([^\n]*\n[\s\S]{0,2000}?(?:los_rate|dq_|sigma_dot|eta_)[\s\S]*?)(?=^function|\Z)",
            script,
            re.IGNORECASE,
        )
        if m:
            return m.group(1)[:3000], m.group(2)

        # ── Pattern 5: returns navigation-command variables ───────────────────
        # Covers: ``function [N1,N2,G1,G2,gc]=gf(…)``  (monte_carlo_single)
        #         ``function [ay_cmd,az_cmd,…]=gf(…)``  (APN/HAOG refactored)
        #         ``function [Nz,Ny,…]=pn_calc(…)``     (custom naming)
        # Captures the function name from after ``=``.
        _nav_outputs = (
            r"N[1-9c]|nav|Nz|Ny|N_lat|N_lon|Nz_cmd|Ny_cmd"
            r"|ay_cmd|az_cmd|a_cmd_y|a_cmd_z|a_lat|a_lon"
        )
        m = re.search(
            rf"(?m)^(function\s+\[(?:{_nav_outputs})[^\]]*\]\s*=\s*([a-zA-Z_]\w*)[^\n]*\n[\s\S]*?)(?=^function|\Z)",
            script,
            re.IGNORECASE,
        )
        if m:
            return m.group(1)[:3000], m.group(2)

        # ── Pattern 6: short guidance-function names (gf/gc/glaw/pn_law) ──────
        # Correctly skips the output list ``[…]=`` before the function name so
        # the captured group is always the function name, not an output variable.
        m = re.search(
            r"(?m)^(function\s+(?:[^\n=]*=\s*)?(g[fc]|g[_]?law|pn_law|pnlaw|pn_calc)\s*\([^\n]*\n[\s\S]*?)(?=^function|\Z)",
            script,
            re.IGNORECASE,
        )
        if m:
            return m.group(1)[:3000], m.group(2)

        # ── Pattern 7: body contains PN-math (Np*Vc / N_pn*Vc) ───────────────
        m = re.search(
            r"(?m)^(function\s+(?:[^\n=]*=\s*)?([a-zA-Z_]\w*)\s*\([^\n]*\n[\s\S]{0,2000}?(?:Np|N_pn|N_PN)\s*\*\s*Vc[\s\S]*?)(?=^function|\Z)",
            script,
        )
        if m:
            return m.group(1)[:3000], m.group(2)

        # ── Pattern 8: inline guidance inside a simulation function ───────────
        # Handles scripts where the guidance law is NOT a separate subfunction
        # but is computed inline inside ``sim_engagement`` / ``sim_s`` /
        # ``mc_run`` etc.  (HAOG, OGL, MODIFY_LAW_v2 style).
        # Strategy: find a known simulation function whose body contains at
        # least one guidance-math indicator (ay_cmd, lam_dot, N_pn*, …).
        # We return the whole simulation function so the LLM can locate and
        # rewrite the guidance section in context.
        _inline_kw = ModelGenerationAgent._INLINE_GUIDANCE_VARS
        _sim_fn   = ModelGenerationAgent._SIM_FUNC_NAMES
        for m in re.finditer(
            r"(?m)^(function\s+(?:[^\n=]*=\s*)?([a-zA-Z_]\w*)\s*\([^\n]*\n[\s\S]*?)(?=^function|\Z)",
            script,
        ):
            func_name = m.group(2)
            body      = m.group(1)
            # Only consider recognised simulation-loop function names
            if not _sim_fn.search(func_name):
                continue
            # Require at least two guidance-indicator hits in the body
            hits = _inline_kw.findall(body)
            if len(hits) >= 2:
                return body[:3500], func_name

        return "", ""

    @staticmethod
    def _replace_guidance_section_by_name(script: str, func_name: str, new_section: str) -> str:
        """Replace the named function body with ``new_section``."""
        pattern = re.compile(
            rf"(?m)^(function[^\n]*?\b{re.escape(func_name)}\b[^\n]*\n[\s\S]*?)(?=^function|\Z)"
        )
        replaced = pattern.sub(new_section.rstrip() + "\n", script, count=1)
        if replaced == script:
            # Fallback: append new section at end (last resort)
            logger.warning(
                f"[ModelGen] _replace_guidance_section_by_name: could not find "
                f"'{func_name}' in script; appending new section."
            )
            return script.rstrip() + "\n\n" + new_section
        return replaced

    # Python param key → MATLAB rl_* variable name (mirrors GenerateMATLABTool)
    _RL_PARAM_MAP: Dict[str, str] = {
        "w1":    "rl_w1",    "zeta1": "rl_zeta1", "tao1": "rl_tao1",
        "w2":    "rl_w2",    "zeta2": "rl_zeta2", "tao2": "rl_tao2",
        "w3":    "rl_w3",    "zeta3": "rl_zeta3", "tao3": "rl_tao3",
        "N_pn":      "rl_N_pn",
        "N_guidance": "rl_N_pn",
    }

    @classmethod
    def _patch_autopilot_params(cls, script: str, params: Dict) -> str:
        """Patch both autopilot (w/zeta/tao) and guidance (N_pn) parameters.

        Strategy A — RL_PARAMS_BEGIN/END block (monte_carlo_single style):
            replaces rl_<name> variables directly in the block.
        Strategy B — bare rl_<name> = value anywhere in script (block absent):
            same variable names, any location.
        Strategy C — dp.<name> = value (legacy templates):
            fallback for scripts that don't use the RL_PARAMS pattern.
        """
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
        # Strategy B
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
        _ALL_CATS = list("ABCDEF")
        enabled: set = set()
        for m in re.finditer(r"run_([A-F])\s*=\s*(true|false)", conditions_str, re.IGNORECASE):
            cat, val = m.group(1).upper(), m.group(2).lower()
            if val == "true":
                enabled.add(cat)
        for cat in _ALL_CATS:
            val = "true" if cat in enabled else "false"
            script = re.sub(
                rf"run_{cat}\s*=\s*(true|false)\s*;", f"run_{cat} = {val};", script
            )
        for m in re.finditer(r"sub_([A-F])\s*=\s*\[(.*?)\]", conditions_str, re.IGNORECASE):
            cat, content = m.group(1).upper(), m.group(2).strip()
            script = re.sub(
                rf"sub_{cat}\s*=\s*\[.*?\]\s*;", f"sub_{cat} = [{content}];", script
            )
        return script

    @staticmethod
    def _extract_param_section(script: str) -> str:
        """
        Extract the parameter / initialisation block from the script.
        Looks for the dp.* assignment block or a marked initialisation section.
        Falls back to the first 2000 characters if neither is found.
        """
        # Try to find the dp.* parameter block (at least 3 consecutive dp. lines)
        m = re.search(
            r"((?:(?:dp\.[\w]+\s*=\s*[^\n]+;\s*\n){3,}))",
            script,
        )
        if m:
            # Return from the start of that block, up to 2500 chars
            start = max(0, m.start() - 200)
            return script[start : start + 2500]
        # Fallback: first 2000 chars of script (usually contains init params)
        return script[:2000]

    @staticmethod
    def _is_valid_matlab(content: str) -> bool:
        """Lightweight MATLAB content check."""
        if "```" in content:
            return False
        if len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+\S", content)) >= 2:
            return False
        return bool(re.search(r"\w+\s*=\s*.+;", content))
