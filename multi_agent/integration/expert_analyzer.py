"""
Expert模式 - 仿真文件分析与参数调优建议

功能：
1. 分析仿真结果（性能指标）
2. 分析仿真文件（代码逻辑）
3. 生成参数调优建议
4. 执行参数调整并重新仿真
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ExpertAnalyzer:
    """Expert模式下的仿真文件和结果分析器"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def analyze_simulation_file(
        self,
        script_path: str,
        task_prompt: str,
        current_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        分析仿真文件的代码逻辑和结构
        
        Returns:
            {
                "file_analysis": "代码分析结果",
                "issues": ["问题1", "问题2", ...],
                "suggestions": ["建议1", "建议2", ...],
            }
        """
        try:
            with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
                script_content = f.read()
        except Exception as exc:
            logger.error(f"[ExpertAnalyzer] Failed to read script: {exc}")
            return {
                "file_analysis": f"无法读取文件: {exc}",
                "issues": [],
                "suggestions": [],
            }

        # 提取关键代码段
        guidance_law = self._extract_guidance_law(script_content)
        autopilot_params = self._extract_autopilot_params(script_content)
        control_law = self._extract_control_law(script_content)

        # 构建分析提示
        analysis_prompt = self._build_file_analysis_prompt(
            task_prompt=task_prompt,
            current_metrics=current_metrics,
            guidance_law=guidance_law,
            autopilot_params=autopilot_params,
            control_law=control_law,
        )

        # 调用LLM进行分析
        if not self.llm_client:
            return {
                "file_analysis": "LLM客户端不可用",
                "issues": [],
                "suggestions": [],
            }

        try:
            response = await self._call_llm(analysis_prompt)
            result = self._parse_analysis_response(response)
            return result
        except Exception as exc:
            logger.error(f"[ExpertAnalyzer] LLM analysis failed: {exc}")
            return {
                "file_analysis": f"分析失败: {exc}",
                "issues": [],
                "suggestions": [],
            }

    async def generate_tuning_suggestions(
        self,
        current_metrics: Dict[str, float],
        task_prompt: str,
        file_analysis: Dict[str, Any],
        optimization_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        基于仿真结果和文件分析，生成参数调优建议
        
        Returns:
            {
                "param_suggestions": {
                    "w1": {"current": 40, "suggested": 45, "reason": "..."},
                    ...
                },
                "tuning_strategy": "调优策略描述",
                "priority": ["参数1", "参数2", ...],  # 优先调整顺序
            }
        """
        hit = current_metrics.get("hit_rate", 0.0)
        sep = current_metrics.get("SEP", current_metrics.get("miss_distance", 99.0))
        ny = current_metrics.get("peak_ny", current_metrics.get("peak_n", 0.0))
        pm = current_metrics.get("pitch_PM", 0.0)
        bw = current_metrics.get("pitch_BW", 0.0)

        # 分析哪些指标不满足要求
        failing_metrics = self._identify_failing_metrics(
            hit=hit, sep=sep, ny=ny, pm=pm, bw=bw
        )

        # 构建调优建议提示
        tuning_prompt = self._build_tuning_prompt(
            current_metrics=current_metrics,
            failing_metrics=failing_metrics,
            file_analysis=file_analysis,
            task_prompt=task_prompt,
            optimization_history=optimization_history,
        )

        if not self.llm_client:
            return {
                "param_suggestions": {},
                "tuning_strategy": "LLM客户端不可用",
                "priority": [],
            }

        try:
            response = await self._call_llm(tuning_prompt)
            result = self._parse_tuning_response(response)
            return result
        except Exception as exc:
            logger.error(f"[ExpertAnalyzer] Tuning suggestion failed: {exc}")
            return {
                "param_suggestions": {},
                "tuning_strategy": f"建议生成失败: {exc}",
                "priority": [],
            }

    # ─────────────────────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_guidance_law(self, script_content: str) -> str:
        """提取制导律代码段"""
        match = re.search(
            r"function\s+a_c\s*=\s*gf\s*\([^)]*\).*?end",
            script_content,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(0)[:500]  # 限制长度
        return "未找到制导律函数"

    def _extract_autopilot_params(self, script_content: str) -> str:
        """提取自动驾驶仪参数"""
        match = re.search(
            r"rl_w1\s*=\s*([\d.]+).*?rl_zeta1\s*=\s*([\d.]+).*?rl_tao1\s*=\s*([\d.]+)",
            script_content,
            re.DOTALL,
        )
        if match:
            return f"w1={match.group(1)}, zeta1={match.group(2)}, tao1={match.group(3)}"
        return "未找到自动驾驶仪参数"

    def _extract_control_law(self, script_content: str) -> str:
        """提取控制律代码段"""
        match = re.search(
            r"function\s+\[.*?\]\s*=\s*cf\s*\([^)]*\).*?end",
            script_content,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(0)[:500]  # 限制长度
        return "未找到控制律函数"

    def _identify_failing_metrics(
        self, hit: float, sep: float, ny: float, pm: float, bw: float
    ) -> Dict[str, str]:
        """识别不满足要求的指标"""
        failing = {}
        if hit < 92:
            failing["hit_rate"] = f"当前{hit:.1f}%, 要求>=92%"
        if sep > 7:
            failing["SEP"] = f"当前{sep:.2f}m, 要求<=7m"
        if ny > 20:
            failing["peak_ny"] = f"当前{ny:.1f}g, 要求<=20g"
        if pm < 45 or pm > 70:
            failing["pitch_PM"] = f"当前{pm:.1f}°, 要求45~70°"
        if bw < 20 or bw > 85:
            failing["pitch_BW"] = f"当前{bw:.1f}rad/s, 要求20~85"
        return failing

    def _build_file_analysis_prompt(
        self,
        task_prompt: str,
        current_metrics: Dict[str, float],
        guidance_law: str,
        autopilot_params: str,
        control_law: str,
    ) -> str:
        """构建仿真文件分析提示"""
        return f"""
你是导弹制导系统的专家。请分析以下仿真文件的代码逻辑和结构。

【任务要求】
{task_prompt[:300]}

【当前仿真指标】
命中率: {current_metrics.get('hit_rate', 0):.1f}%
SEP: {current_metrics.get('SEP', 99):.2f}m
峰值过载: {current_metrics.get('peak_ny', 0):.1f}g
相位裕度(PM): {current_metrics.get('pitch_PM', 0):.1f}°
带宽(BW): {current_metrics.get('pitch_BW', 0):.1f}rad/s

【制导律代码】
{guidance_law}

【自动驾驶仪参数】
{autopilot_params}

【控制律代码】
{control_law}

请分析：
1. 代码逻辑是否正确？
2. 参数设置是否合理？
3. 存在哪些潜在问题？
4. 如何改进？

返回JSON格式：
{{
  "file_analysis": "代码分析结果",
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]
}}
"""

    def _build_tuning_prompt(
        self,
        current_metrics: Dict[str, float],
        failing_metrics: Dict[str, str],
        file_analysis: Dict[str, Any],
        task_prompt: str,
        optimization_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """构建参数调优建议提示"""
        history_str = "无"
        if optimization_history:
            _recent = optimization_history[-3:]
            history_str = "\n".join(
                f"  轮{h.get('round', '?')}: "
                f"hit={h.get('metrics', {}).get('hit_rate', 0):.0f}% "
                f"SEP={h.get('metrics', {}).get('SEP', 99):.1f}m"
                for h in _recent
            )

        failing_str = "\n".join(f"  {k}: {v}" for k, v in failing_metrics.items())

        return f"""
你是导弹制导系统的参数调优专家。

【当前指标】
{json.dumps(current_metrics, ensure_ascii=False, indent=2)}

【不满足的要求】
{failing_str if failing_str else "所有指标都满足要求"}

【文件分析结果】
{file_analysis.get('file_analysis', '无')}

【优化历史】
{history_str}

【参数效应说明】
- w1↑ → BW↑, PM↓ (带宽增大但相位裕度降低)
- zeta1↑ → PM↑ (阻尼增大→稳定性增强)
- tao1↑ → PM↓↓ (时间常数增大→严重降低稳定性)
- N_pn↑ → 命中率↑, SEP↓, PeakNy↑

请给出具体的参数调优建议，优先级从高到低。

返回JSON格式：
{{
  "param_suggestions": {{
    "w1": {{"current": 40, "suggested": 45, "reason": "..."}},
    "zeta1": {{"current": 0.75, "suggested": 0.80, "reason": "..."}},
    ...
  }},
  "tuning_strategy": "调优策略描述",
  "priority": ["w1", "zeta1", ...]
}}
"""

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        if not self.llm_client:
            raise ValueError("LLM客户端不可用")

        try:
            # 尝试不同的LLM接口
            if hasattr(self.llm_client, "chat"):
                # OpenAI兼容接口
                response = await self.llm_client.chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000,
                )
                return response.choices[0].message.content or ""
            elif hasattr(self.llm_client, "run_conversation"):
                # Hermes接口
                response = await self.llm_client.run_conversation(prompt)
                if isinstance(response, dict):
                    return response.get("final_response", "")
                return str(response)
            else:
                raise ValueError("未知的LLM客户端类型")
        except Exception as exc:
            logger.error(f"[ExpertAnalyzer] LLM call failed: {exc}")
            raise

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """解析文件分析响应"""
        try:
            # 提取JSON
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return {
                    "file_analysis": response,
                    "issues": [],
                    "suggestions": [],
                }
            data = json.loads(match.group())
            return {
                "file_analysis": data.get("file_analysis", ""),
                "issues": data.get("issues", []),
                "suggestions": data.get("suggestions", []),
            }
        except Exception as exc:
            logger.warning(f"[ExpertAnalyzer] Parse analysis failed: {exc}")
            return {
                "file_analysis": response,
                "issues": [],
                "suggestions": [],
            }

    def _parse_tuning_response(self, response: str) -> Dict[str, Any]:
        """解析调优建议响应"""
        try:
            # 提取JSON
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return {
                    "param_suggestions": {},
                    "tuning_strategy": response,
                    "priority": [],
                }
            data = json.loads(match.group())
            return {
                "param_suggestions": data.get("param_suggestions", {}),
                "tuning_strategy": data.get("tuning_strategy", ""),
                "priority": data.get("priority", []),
            }
        except Exception as exc:
            logger.warning(f"[ExpertAnalyzer] Parse tuning failed: {exc}")
            return {
                "param_suggestions": {},
                "tuning_strategy": response,
                "priority": [],
            }
