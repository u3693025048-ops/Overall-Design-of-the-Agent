"""
测试 ExpertAnalyzer 的仿真文件分析和参数调优建议功能
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from multi_agent.integration.expert_analyzer import ExpertAnalyzer


class TestExpertAnalyzer:
    """ExpertAnalyzer 单元测试"""

    @pytest.fixture
    def analyzer(self):
        """创建分析器实例"""
        mock_client = Mock()
        return ExpertAnalyzer(llm_client=mock_client)

    @pytest.fixture
    def sample_metrics(self):
        """示例仿真指标"""
        return {
            "hit_rate": 85.0,
            "SEP": 8.5,
            "peak_ny": 22.0,
            "pitch_PM": 40.0,
            "pitch_BW": 75.0,
        }

    @pytest.fixture
    def sample_script(self, tmp_path):
        """创建示例MATLAB脚本"""
        script_content = """
% 制导律
function a_c = gf(T_go, v_t, v_m, N_pn)
    a_c = N_pn * v_m / T_go;
end

% 自动驾驶仪参数
rl_w1 = 40;
rl_zeta1 = 0.75;
rl_tao1 = 0.20;
rl_w2 = 35;
rl_zeta2 = 0.75;
rl_tao2 = 0.20;
rl_w3 = 40;
rl_zeta3 = 0.70;
rl_N_pn = 4.0;

% 控制律
function [a_cmd] = cf(error, rate)
    a_cmd = error * 10 + rate * 2;
end
"""
        script_file = tmp_path / "test_script.m"
        script_file.write_text(script_content)
        return str(script_file)

    def test_extract_guidance_law(self, analyzer, sample_script):
        """测试提取制导律代码"""
        with open(sample_script, "r") as f:
            content = f.read()
        
        guidance_law = analyzer._extract_guidance_law(content)
        assert "function a_c = gf" in guidance_law
        assert "N_pn" in guidance_law

    def test_extract_autopilot_params(self, analyzer, sample_script):
        """测试提取自动驾驶仪参数"""
        with open(sample_script, "r") as f:
            content = f.read()
        
        params = analyzer._extract_autopilot_params(content)
        assert "w1=40" in params
        assert "zeta1=0.75" in params
        assert "tao1=0.20" in params

    def test_extract_control_law(self, analyzer, sample_script):
        """测试提取控制律代码"""
        with open(sample_script, "r") as f:
            content = f.read()
        
        control_law = analyzer._extract_control_law(content)
        assert "function [a_cmd] = cf" in control_law

    def test_identify_failing_metrics(self, analyzer):
        """测试识别不满足的指标"""
        failing = analyzer._identify_failing_metrics(
            hit=85.0,  # < 92
            sep=8.5,   # > 7
            ny=22.0,   # > 20
            pm=40.0,   # < 45
            bw=75.0,   # 在范围内
        )
        
        assert "hit_rate" in failing
        assert "SEP" in failing
        assert "peak_ny" in failing
        assert "pitch_PM" in failing
        assert "pitch_BW" not in failing

    def test_parse_analysis_response_valid_json(self, analyzer):
        """测试解析有效的分析响应"""
        response = """
        {
            "file_analysis": "代码分析结果",
            "issues": ["问题1", "问题2"],
            "suggestions": ["建议1", "建议2"]
        }
        """
        
        result = analyzer._parse_analysis_response(response)
        assert result["file_analysis"] == "代码分析结果"
        assert len(result["issues"]) == 2
        assert len(result["suggestions"]) == 2

    def test_parse_analysis_response_markdown_json(self, analyzer):
        """测试解析带markdown的JSON响应"""
        response = """
        ```json
        {
            "file_analysis": "代码分析结果",
            "issues": ["问题1"],
            "suggestions": ["建议1"]
        }
        ```
        """
        
        result = analyzer._parse_analysis_response(response)
        assert result["file_analysis"] == "代码分析结果"

    def test_parse_tuning_response_valid_json(self, analyzer):
        """测试解析有效的调优响应"""
        response = """
        {
            "param_suggestions": {
                "w1": {"current": 40, "suggested": 45, "reason": "增加带宽"},
                "zeta1": {"current": 0.75, "suggested": 0.80, "reason": "增加稳定性"}
            },
            "tuning_strategy": "优先调整俯仰通道",
            "priority": ["w1", "zeta1"]
        }
        """
        
        result = analyzer._parse_tuning_response(response)
        assert "w1" in result["param_suggestions"]
        assert "zeta1" in result["param_suggestions"]
        assert result["priority"] == ["w1", "zeta1"]

    def test_build_file_analysis_prompt(self, analyzer, sample_metrics):
        """测试构建文件分析提示"""
        prompt = analyzer._build_file_analysis_prompt(
            task_prompt="测试任务",
            current_metrics=sample_metrics,
            guidance_law="制导律代码",
            autopilot_params="w1=40, zeta1=0.75, tao1=0.20",
            control_law="控制律代码",
        )
        
        assert "制导律代码" in prompt
        assert "w1=40" in prompt
        assert "命中率: 85.0%" in prompt
        assert "SEP: 8.50m" in prompt

    def test_build_tuning_prompt(self, analyzer, sample_metrics):
        """测试构建调优提示"""
        file_analysis = {
            "file_analysis": "代码分析结果",
            "issues": ["问题1"],
            "suggestions": ["建议1"],
        }
        
        failing_metrics = {
            "hit_rate": "当前85.0%, 要求>=92%",
            "SEP": "当前8.50m, 要求<=7m",
        }
        
        prompt = analyzer._build_tuning_prompt(
            current_metrics=sample_metrics,
            failing_metrics=failing_metrics,
            file_analysis=file_analysis,
            task_prompt="测试任务",
            optimization_history=None,
        )
        
        assert "当前85.0%" in prompt
        assert "当前8.50m" in prompt
        assert "代码分析结果" in prompt

    @pytest.mark.asyncio
    async def test_analyze_simulation_file_with_mock(self, analyzer, sample_script):
        """测试分析仿真文件（使用mock LLM）"""
        # Mock LLM响应
        analyzer.llm_client = AsyncMock()
        analyzer.llm_client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content="""
                            {
                                "file_analysis": "制导律代码结构正确",
                                "issues": ["w1过小"],
                                "suggestions": ["增加w1到45"]
                            }
                            """
                        )
                    )
                ]
            )
        )
        
        result = await analyzer.analyze_simulation_file(
            script_path=sample_script,
            task_prompt="测试任务",
            current_metrics={"hit_rate": 85.0, "SEP": 8.5},
        )
        
        assert "file_analysis" in result
        assert "issues" in result
        assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_generate_tuning_suggestions_with_mock(self, analyzer, sample_metrics):
        """测试生成调优建议（使用mock LLM）"""
        # Mock LLM响应
        analyzer.llm_client = AsyncMock()
        analyzer.llm_client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content="""
                            {
                                "param_suggestions": {
                                    "w1": {"current": 40, "suggested": 45, "reason": "增加带宽"}
                                },
                                "tuning_strategy": "优先调整俯仰通道",
                                "priority": ["w1"]
                            }
                            """
                        )
                    )
                ]
            )
        )
        
        file_analysis = {
            "file_analysis": "代码分析结果",
            "issues": [],
            "suggestions": [],
        }
        
        result = await analyzer.generate_tuning_suggestions(
            current_metrics=sample_metrics,
            task_prompt="测试任务",
            file_analysis=file_analysis,
            optimization_history=None,
        )
        
        assert "param_suggestions" in result
        assert "tuning_strategy" in result
        assert "priority" in result


class TestExpertAnalyzerIntegration:
    """ExpertAnalyzer 集成测试"""

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, tmp_path):
        """测试完整的分析工作流"""
        # 创建示例脚本
        script_content = """
        rl_w1 = 40;
        rl_zeta1 = 0.75;
        rl_tao1 = 0.20;
        function a_c = gf(T_go, v_t, v_m, N_pn)
            a_c = N_pn * v_m / T_go;
        end
        """
        script_file = tmp_path / "test.m"
        script_file.write_text(script_content)
        
        # 创建分析器（不使用LLM）
        analyzer = ExpertAnalyzer(llm_client=None)
        
        # 测试文件提取
        with open(script_file, "r") as f:
            content = f.read()
        
        guidance_law = analyzer._extract_guidance_law(content)
        autopilot_params = analyzer._extract_autopilot_params(content)
        
        assert "function a_c = gf" in guidance_law
        assert "w1=40" in autopilot_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
