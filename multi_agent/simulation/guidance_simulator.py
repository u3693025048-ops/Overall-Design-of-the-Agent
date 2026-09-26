#!/usr/bin/env python3
"""
Guidance System Simulation Module
自动生成 SysML 模型和 MATLAB/Simulink 脚本进行迭代模拟验证
"""

import math
import os
import re
import json
import locale
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


# ── Octave batch-mode constants (shared by all subprocess call sites) ───────
# These prevent figure-rendering overhead from dominating Octave runtime
# when the KB template is executed for RL or batch reporting purposes.
#
# OCTAVE_BATCH_FLAGS    : CLI flags that disable GUI / window-system init.
#                         Combined with --eval, Octave runs in pure batch mode
#                         on systems with or without an X server / display.
# OCTAVE_BATCH_PREFIX   : Eval-string prefix applied before every script call.
#                         Suppresses warnings, disables pager and figure
#                         visibility, and pre-loads the control package once
#                         so bode/margin work without per-call lookup cost.
# DEFAULT_SUBPROCESS_TIMEOUT_SEC : default subprocess timeout in seconds.
#                                  Overridable via env MATLAB_TIMEOUT_SEC.
#                                  20 minutes by default, comfortable margin
#                                  for an RL episode on the heaviest KB template
#                                  (Nmc=20 × 6 subcases + robustness analysis).
OCTAVE_BATCH_FLAGS: List[str] = ["--quiet", "--no-window-system"]
OCTAVE_BATCH_PREFIX: str = (
    "warning('off','all'); "
    "more off; "
    "set(0,'DefaultFigureVisible','off'); "
    "pkg load control; "
)


def _resolve_subprocess_timeout(default_sec: int = 1200) -> int:
    """Return the subprocess timeout in seconds.

    Reads ``MATLAB_TIMEOUT_SEC`` env var if set (must be a positive integer);
    otherwise falls back to *default_sec*.  Negative or non-integer values
    are silently ignored and the default is used.
    """
    raw = os.environ.get("MATLAB_TIMEOUT_SEC", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            logger.warning(
                f"MATLAB_TIMEOUT_SEC={raw!r} is not a positive integer; "
                f"using default {default_sec}s."
            )
    return default_sec


DEFAULT_SUBPROCESS_TIMEOUT_SEC: int = _resolve_subprocess_timeout()


def build_octave_eval_string(script_abs_path: str,
                             call_name: Optional[str] = None) -> str:
    """Build an Octave ``--eval`` argument that runs *script_abs_path*.

    Embeds OCTAVE_BATCH_PREFIX (graphics suppression + control package).
    If *call_name* is provided, appends ``call_name()`` so a function-file
    template is invoked after being source()'d.  Forward-slashes the path
    so Octave's source() / addpath() don't choke on backslashes on Windows.
    """
    abs_path = str(script_abs_path).replace("\\", "/")
    if call_name:
        return f"{OCTAVE_BATCH_PREFIX}source('{abs_path}'); {call_name}()"
    return f"{OCTAVE_BATCH_PREFIX}source('{abs_path}')"


@dataclass
class GuidanceParameters:
    """制导系统参数"""

    navigation_coefficient: float = 3.0
    damping_ratio: float = 0.3
    control_gain: float = 1.0
    target_position: List[float] = field(default_factory=lambda: [20000.0, 2000.0, 5000.0])
    target_velocity: List[float] = field(default_factory=lambda: [250.0, 0.0, 0.0])
    target_velocity_expr: List[str] = field(default_factory=lambda: ["250.0", "0.0", "0.0"])
    initial_position: List[float] = field(default_factory=lambda: [0.0, 7000.0, 0.0])
    initial_velocity: List[float] = field(default_factory=lambda: [960.0, 0.0, 0.0])
    initial_angles: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0]) # upsilon, phi, gama
    gama_max: float = 45.0
    scenario_id: int = 2 # 默认使用工况2（标准基准工况）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "navigation_coefficient": self.navigation_coefficient,
            "damping_ratio": self.damping_ratio,
            "control_gain": self.control_gain,
            "target_position": self.target_position,
            "target_velocity": self.target_velocity,
            "target_velocity_expr": self.target_velocity_expr,
            "initial_position": self.initial_position,
            "initial_velocity": self.initial_velocity,
            "initial_angles": self.initial_angles,
            "gama_max": self.gama_max,
            "scenario_id": self.scenario_id,
        }


@dataclass
class SimulationResult:
    """仿真结果"""

    miss_distance: float
    control_energy: float
    max_overshoot: float
    settling_time: float
    final_position: List[float]
    final_velocity: List[float]
    trajectory: List[Dict[str, float]]
    execution_time: float
    success: bool = True
    error: Optional[str] = None


class SysMLModelGenerator:
    """SysML 模型生成器"""

    def __init__(self, output_dir: str = "./sysml_models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Reference template directory
        self.kb_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "knowledge_base", "sysml"
        )

    def _read_template(self, filename: str, fallback: str) -> str:
        filepath = os.path.join(self.kb_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"SysML reference template not found at {filepath}, using fallback.")
            return fallback

    def generate_block_definition_diagram(
        self, params: GuidanceParameters, model_name: str = "GuidanceSystem"
    ) -> str:
        """生成 SysML Block Definition Diagram (BDD)"""

        fallback = f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by Multi-Agent System: {datetime.now().isoformat()} -->
<SysML xmlns="http://www.omg.org/spec/SysML/20190109/SysML">
  <package name="{{model_name}}">
    <block name="GuidanceController" isAbstract="false">
      <ownedAttribute name="navigationCoefficient" type="Real" multiplicity="1">
        <defaultValue value="{{navigation_coefficient}}"/>
      </ownedAttribute>
      <ownedAttribute name="dampingRatio" type="Real" multiplicity="1">
        <defaultValue value="{{damping_ratio}}"/>
      </ownedAttribute>
    </block>
    <block name="Autopilot">
      <ownedAttribute name="controlGain" type="Real" multiplicity="1">
        <defaultValue value="{{control_gain}}"/>
      </ownedAttribute>
      <operation name="three_loop_autopilot"/>
    </block>
  </package>
</SysML>"""

        template = self._read_template("sysml_bdd_reference.xml", fallback)
        return template.format(
            model_name=model_name,
            navigation_coefficient=params.navigation_coefficient,
            damping_ratio=params.damping_ratio,
            control_gain=params.control_gain,
        )

    def generate_parametric_diagram(
        self, params: GuidanceParameters, model_name: str = "GuidanceSystem"
    ) -> str:
        """生成 SysML Parametric Diagram"""

        fallback = f"""<?xml version="1.0" encoding="UTF-8"?>
<SysML xmlns="http://www.omg.org/spec/SysML/20190109/SysML">
  <block name="GuidanceEquations">
    <constraint name="navigationEquation">
      <ownedAttribute name="N" type="Real" value="{{navigation_coefficient}}"/>
    </constraint>
    <constraint name="dampingEquation">
      <ownedAttribute name="c" type="Real" value="{{damping_ratio}}"/>
    </constraint>
    <constraint name="missDistanceEquation">
      <ownedAttribute name="targetPosition" type="Vector3D" value="({{target_position_0}}, {{target_position_1}}, {{target_position_2}})"/>
    </constraint>
  </block>
</SysML>"""

        template = self._read_template("sysml_parametric_reference.xml", fallback)
        return template.format(
            model_name=model_name,
            navigation_coefficient=params.navigation_coefficient,
            damping_ratio=params.damping_ratio,
            target_position_0=params.target_position[0],
            target_position_1=params.target_position[1],
            target_position_2=params.target_position[2],
        )

    def generate_internal_block_diagram(
        self, params: GuidanceParameters, model_name: str = "GuidanceSystem"
    ) -> str:
        """生成 SysML Internal Block Diagram (IBD)"""

        fallback = f"""<?xml version="1.0" encoding="UTF-8"?>
<SysML xmlns="http://www.omg.org/spec/SysML/20190109/SysML">
  <block name="InterceptorAssembly">
    <part name="sensor" type="TargetTracker"/>
    <part name="guidance" type="GuidanceController"/>
  </block>
</SysML>"""

        template = self._read_template("sysml_ibd_reference.xml", fallback)
        return template.format(
            model_name=model_name,
        )

    def save_model(
        self, params: GuidanceParameters, model_name: str = "GuidanceSystem"
    ) -> Dict[str, str]:
        """生成并保存完整的 SysML 模型"""

        files = {
            f"{model_name}_bdd.xml": self.generate_block_definition_diagram(params, model_name),
            f"{model_name}_parametric.xml": self.generate_parametric_diagram(params, model_name),
            f"{model_name}_ibd.xml": self.generate_internal_block_diagram(params, model_name),
        }

        saved_files = {}
        for filename, content in files.items():
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            saved_files[filename] = filepath
            logger.info(f"Generated SysML file: {filepath}")

        return saved_files


class MATLABScriptGenerator:
    """MATLAB 脚本生成器"""

    def __init__(self, output_dir: str = "./matlab_scripts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_simulation_script(
        self,
        params: GuidanceParameters,
        duration: float = 100.0,
        dt: float = 0.01,
        script_name: str = "guidance_simulation.m",
    ) -> str:
        """生成 MATLAB 仿真脚本 (6-DOF)"""
        
        # 尝试读取知识库中的原始 6-DOF 模型
        base_script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "knowledge_base", "matlab", "guidance", "monte_carlo_single.m"
        )
        
        try:
            with open(base_script_path, "r", encoding="utf-8") as f:
                script = f.read()
        except FileNotFoundError:
            logger.error(f"Base MATLAB script not found at {base_script_path}")
            script = "% Error: Base script not found\n"
            
        # 根据 params 替换初始条件和目标运动
        
        # 替换参数
        init_x, init_y, init_z = params.initial_position
        init_v = params.initial_velocity[0] # 简化的初速提取
        init_upsilon, init_phi, init_gama = params.initial_angles
        gama_max = params.gama_max
        
        target_x, target_y, target_z = params.target_position
        target_vx, target_vy, target_vz = params.target_velocity_expr
        
        nav_coeff = params.navigation_coefficient
        damping = params.damping_ratio
        
        # 使用正则表达式替换来更新初始条件，更健壮
        # 修改函数名
        script = re.sub(
            r"function\s+chengxu_robust_analysis_singlefile\b",
            f"function [miss_distance, control_energy, trajectory] = {script_name.replace('.m', '')}()",
            script
        )

        # 替换目标初始位置 T_x=T_y=T_z=
        script = re.sub(
            r"T_x=20000;T_y=2000;T_z=5000;",
            f"T_x={target_x};T_y={target_y};T_z={target_z};",
            script
        )

        # 替换初始条件 (多种变体)
        script = re.sub(
            r"y=7000(;x=0;z=0;V=960;|;x=0;z=0;V=960;)",
            f"y={init_y};x={init_x};z={init_z};V={init_v};",
            script
        )

        # 替换角度初始化
        script = re.sub(
            r"upsilon=-0/C45;gama=0/C45;phi=0/C45;theta=-0/C45;phiv=0;",
            f"upsilon={init_upsilon}/C45;gama={init_gama}/C45;phi={init_phi}/C45;theta={init_upsilon}/C45;phiv={init_phi};",
            script
        )

        # 替换最大可用过载
        script = re.sub(
            r"gama_max=45/C45;",
            f"gama_max={gama_max}/C45;",
            script
        )

        # 替换目标速度 (多种变体)
        script = re.sub(
            r"T_Vx=250;T_Vy=50;T_Vz=50\*cos\(0\.2\*t\);",
            f"T_Vx={target_vx};T_Vy={target_vy};T_Vz={target_vz};",
            script
        )

        script = re.sub(
            r"T_Vx_end\s*=\s*250;\s*T_Vy_end\s*=\s*50;\s*T_Vz_end\s*=\s*50\*cos\(0\.2\*t\);",
            f"T_Vx_end = {target_vx};\n    T_Vy_end = {target_vy};\n    T_Vz_end = {target_vz};",
            script
        )

        # 注入 nav_coeff 和 damping_ratio 到制导律
        script = re.sub(
            r"ayc_A=3\*sqrt\(Vrx\^2\+Vry\^2\)\*qz_dot;",
            f"ayc_A={nav_coeff}*sqrt(Vrx^2+Vry^2)*qz_dot - {damping}*Vy_A;",
            script
        )
        script = re.sub(
            r"azc_A=-3\*sqrt\(Vrx\^2\+Vrz\^2\)\*qy_dot;",
            f"azc_A=-{nav_coeff}*sqrt(Vrx^2+Vrz^2)*qy_dot + {damping}*Vz_A;",
            script
        )
        
        # 确保输出格式兼容现有的解析逻辑
        script += """
% --- Generated output for parser ---
function dump_metrics(miss_min, Ju)
    fprintf('Simulation Complete\\n');
    fprintf('Miss Distance: %.6f m\\n', miss_min);
    fprintf('Control Energy: %.6f J\\n', Ju);
end
"""
        script = script.replace(
            "fprintf('==========================================\\n');",
            "fprintf('==========================================\\n');\ndump_metrics(timeMetrics.miss_min, timeMetrics.Ju);\nmiss_distance = timeMetrics.miss_min;\ncontrol_energy = timeMetrics.Ju;\ntrajectory = Track;\n"
        )
        
        return script

    def generate_optimization_script(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        objectives: List[str],
        script_name: str = "optimize_guidance.m",
    ) -> str:
        """生成 MATLAB 优化脚本"""

        nav_range = param_ranges.get("navigation_coefficient", (0.3, 0.8))
        damp_range = param_ranges.get("damping_ratio", (0.2, 0.5))

        script = f"""% optimize_guidance.m
% Auto-generated by Multi-Agent System: {datetime.now().isoformat()}
% Optimization Script for Guidance Parameters

function [best_params, best_metrics] = optimize_guidance()
    %% Parameter Ranges
    nav_range = [{nav_range[0]}, {nav_range[1]}];
    damp_range = [{damp_range[0]}, {damp_range[1]}];
    
    %% Objectives: 'miss_distance', 'control_energy', 'overshoot'
    objectives = {{'{objectives[0] if objectives else "miss_distance"}'}};
    
    %% Optimization options
    options = optimoptions('ga', ...
        'PopulationSize', 50, ...
        'MaxGenerations', 30, ...
        'CrossoverFraction', 0.8, ...
        'Display', 'iter');
    
    %% Objective function
    fitnessfcn = @(params) evaluate_params(params);
    
    %% Variable bounds: [nav_coeff, damping_ratio]
    lb = [nav_range(1), damp_range(1)];
    ub = [nav_range(2), damp_range(2)];
    
    %% Run Genetic Algorithm
    [best_params, ~] = ga(fitnessfcn, 2, [], [], [], [], lb, ub, [], options);
    
    %% Evaluate best parameters
    [miss_dist, ctrl_energy, ~] = evaluate_params(best_params);
    
    best_metrics = struct();
    best_metrics.navigation_coefficient = best_params(1);
    best_metrics.damping_ratio = best_params(2);
    best_metrics.miss_distance = miss_dist;
    best_metrics.control_energy = ctrl_energy;
    
    fprintf('\\nOptimization Complete\\n');
    fprintf('Best Navigation Coefficient: %.4f\\n', best_params(1));
    fprintf('Best Damping Ratio: %.4f\\n', best_params(2));
    fprintf('Best Miss Distance: %.6f m\\n', miss_dist);
    fprintf('Best Control Energy: %.6f J\\n', ctrl_energy);
end

function [miss_dist, ctrl_energy, traj] = evaluate_params(params)
    nav_coeff = params(1);
    damping = params(2);
    
    %% Run simulation with given parameters
    sim_params = struct();
    sim_params.navigation_coefficient = nav_coeff;
    sim_params.damping_ratio = damping;
    sim_params.target_position = [10, 0];
    sim_params.initial_position = [0, 0];
    sim_params.initial_velocity = [1, 0];
    
    [traj, metrics] = run_simulation(sim_params);
    
    miss_dist = metrics.miss_distance;
    ctrl_energy = metrics.control_energy;
end

function [trajectory, metrics] = run_simulation(params)
    %% Inline simulation (same as guidance_simulation.m)
    dt = 0.01;
    t_final = 100;
    time = 0:dt:t_final;
    n_steps = length(time);
    
    target_x = params.target_position(1);
    target_y = params.target_position(2);
    
    x = params.initial_position(1);
    y = params.initial_position(2);
    vx = params.initial_velocity(1);
    vy = params.initial_velocity(2);
    
    trajectory = zeros(n_steps, 6);
    trajectory(1,:) = [x, y, vx, vy, 0, 0];
    
    for i = 2:n_steps
        rel_pos = [target_x - x, target_y - y];
        rel_vel = [0 - vx, 0 - vy];
        range = norm(rel_pos);
        t_go = range / (norm(rel_vel) + 1e-6);
        
        if t_go > 0.1
            acc_cmd = params.navigation_coefficient * rel_vel / t_go;
        else
            acc_cmd = [0; 0];
        end
        
        damping_force = -params.damping_ratio * [vx, vy];
        ax = acc_cmd(1) + damping_force(1);
        ay = acc_cmd(2) + damping_force(2);
        
        x = x + vx * dt;
        y = y + vy * dt;
        vx = vx + ax * dt;
        vy = vy + ay * dt;
        
        trajectory(i,:) = [x, y, vx, vy, ax, ay];
    end
    
    miss_dist = norm([x, y] - [target_x, target_y]);
    ctrl_energy = sum(sum(trajectory(:,5:6).^2)) * dt;
    
    metrics = struct('miss_distance', miss_dist, 'control_energy', ctrl_energy);
end
"""
        return script

    def save_scripts(
        self,
        params: GuidanceParameters,
        duration: float = 100.0,
        dt: float = 0.01,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        objectives: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """生成并保存 MATLAB 脚本"""

        scripts = {
            "guidance_simulation.m": self.generate_simulation_script(params, duration, dt),
        }

        if param_ranges:
            scripts["optimize_guidance.m"] = self.generate_optimization_script(
                param_ranges, objectives or ["miss_distance"]
            )

        saved_files = {}
        for filename, content in scripts.items():
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            saved_files[filename] = filepath
            logger.info(f"Generated MATLAB script: {filepath}")

        return saved_files


class SimulationExecutor:
    """仿真执行器，支持 Python / Octave / MATLAB / MATLAB Engine API.

    The actual engine selection goes through
    :func:`multi_agent.simulation.engine_resolver.resolve_engine_config`,
    which honours the ``SIMULATION_ENGINE``, ``MATLAB_PATH`` and
    ``OCTAVE_PATH`` env vars and resolves ``"auto"`` to a concrete choice
    based on what is installed on the host.

    Attributes
    ----------
    engine, octave_path, matlab_path
        Compatibility aliases for the 3 subprocess call sites that read
        these attributes directly.  Always populated with the *resolved*
        values (env-var overridden, absolute paths when discoverable).
    engine_cfg
        Full :class:`EngineConfig` with availability flags.
    """

    def __init__(
        self,
        engine: str = "auto",  # "auto" | "python" | "octave" | "matlab" | "matlab_engine"
        octave_path: str = "octave",
        matlab_path: str = "matlab",
        output_dir: str = "./simulation_results",
    ):
        # Resolve engine + paths once; this is the single source of
        # truth used by every subprocess call site below.
        from multi_agent.simulation.engine_resolver import resolve_engine_config
        self.engine_cfg = resolve_engine_config(
            engine=engine,
            octave_path=octave_path,
            matlab_path=matlab_path,
        )
        # Public aliases — keep as plain str so legacy ``self.engine == "octave"``
        # checks in the 3 subprocess call sites continue to work unchanged.
        self.engine = self.engine_cfg.engine
        self.octave_path = self.engine_cfg.octave_path
        self.matlab_path = self.engine_cfg.matlab_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # In-process MATLAB Engine backend, lazy-initialised on first
        # access via :pyattr:`matlab_engine_backend`.
        self._matlab_engine_backend = None

    @property
    def matlab_engine_backend(self):
        """Return the shared :class:`MatlabEngineBackend` (lazy-started).

        Returns ``None`` if the MATLAB Engine API is not importable or the
        engine fails to start — callers should fall back to the subprocess
        path in that case.

        Crash recovery: if the engine was previously alive (``_ever_started``
        is ``True``) but is now dead (``started`` is ``False``), one automatic
        restart attempt is made.  Permanent failures (MATLAB never launched)
        are not retried to avoid repeated slow start-up attempts.
        """
        if self._matlab_engine_backend is not None:
            if self._matlab_engine_backend.started:
                return self._matlab_engine_backend
            # Dead backend — distinguish transient crash from permanent failure.
            if (self.engine == "matlab_engine"
                    and getattr(self._matlab_engine_backend, "_ever_started", False)):
                logger.info(
                    "matlab.engine backend died; attempting auto-restart…"
                )
                if self._matlab_engine_backend.restart():
                    logger.info("matlab.engine backend restarted successfully.")
                    return self._matlab_engine_backend
                logger.warning(
                    "matlab.engine restart failed; falling back to subprocess."
                )
            return None

        # Only attempt to start when the resolved engine actually wants it.
        if self.engine != "matlab_engine":
            return None

        from multi_agent.simulation.matlab_engine_backend import MatlabEngineBackend
        backend = MatlabEngineBackend()
        if backend.start():
            self._matlab_engine_backend = backend
            return backend
        # Failed to start — log once, remember failure to avoid re-trying.
        logger.warning(
            "matlab.engine backend requested but failed to start; "
            "subsequent calls will need to fall back to subprocess."
        )
        self._matlab_engine_backend = backend  # store the failed instance to skip retries
        return None

    def close(self) -> None:
        """Tear down any in-process MATLAB engine session.  Idempotent."""
        if self._matlab_engine_backend is not None:
            try:
                self._matlab_engine_backend.stop()
            except Exception as exc:  # pragma: no cover
                logger.debug("MATLAB engine backend stop raised %s; ignoring.", exc)
            self._matlab_engine_backend = None

    async def run_simulation(
        self,
        params: GuidanceParameters,
        duration: float = 100.0,
        dt: float = 0.01,
        script_path: Optional[str] = None,
    ) -> SimulationResult:
        """执行仿真（根据配置的引擎）"""

        start_time = asyncio.get_event_loop().time()

        try:
            if self.engine == "python":
                result = await self._python_simulation(params, duration, dt)
            elif self.engine in ("octave", "matlab", "matlab_engine") and script_path:
                result = await self.run_external_simulation(script_path)
            else:
                result = await self._python_simulation(params, duration, dt)
            result.execution_time = asyncio.get_event_loop().time() - start_time
            return result
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return SimulationResult(
                miss_distance=float('nan'),
                control_energy=float('nan'),
                max_overshoot=float('nan'),
                settling_time=float('nan'),
                final_position=[float('nan'), float('nan')],
                final_velocity=[float('nan'), float('nan')],
                trajectory=[],
                execution_time=0,
                success=False,
                error=str(e),
            )

    async def _python_simulation(
        self,
        params: GuidanceParameters,
        duration: float,
        dt: float,
    ) -> SimulationResult:
        """使用 Python 进行 3D 制导仿真计算"""

        nav_coeff = params.navigation_coefficient
        damping = params.damping_ratio
        
        target_x, target_y, target_z = params.target_position
        target_vx, target_vy, target_vz = params.target_velocity

        x, y, z = params.initial_position
        vx, vy, vz = params.initial_velocity

        trajectory = []
        control_energies = []

        num_steps = int(duration / dt)
        
        # 简单模拟 3D 运动
        for step in range(num_steps):
            current_time = step * dt
            trajectory.append(
                {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz, "ax": 0, "ay": 0, "az": 0, "time": current_time}
            )

            rel_pos = [target_x - x, target_y - y, target_z - z]
            rel_vel = [target_vx - vx, target_vy - vy, target_vz - vz]
            range_to_target = (rel_pos[0] ** 2 + rel_pos[1] ** 2 + rel_pos[2] ** 2) ** 0.5
            v_close = -((rel_pos[0]*rel_vel[0] + rel_pos[1]*rel_vel[1] + rel_pos[2]*rel_vel[2]) / (range_to_target + 1e-6))
            
            t_go = range_to_target / (v_close + 1e-6) if v_close > 0 else 999.0

            if t_go > 0.1 and range_to_target > 5.0 and v_close > 0:
                # 3D 比例导引律
                omega_x = (rel_pos[1]*rel_vel[2] - rel_pos[2]*rel_vel[1]) / (range_to_target**2 + 1e-6)
                omega_y = (rel_pos[2]*rel_vel[0] - rel_pos[0]*rel_vel[2]) / (range_to_target**2 + 1e-6)
                omega_z = (rel_pos[0]*rel_vel[1] - rel_pos[1]*rel_vel[0]) / (range_to_target**2 + 1e-6)
                
                acc_cmd_x = nav_coeff * v_close * (omega_y * rel_pos[2] - omega_z * rel_pos[1]) / (range_to_target + 1e-6)
                acc_cmd_y = nav_coeff * v_close * (omega_z * rel_pos[0] - omega_x * rel_pos[2]) / (range_to_target + 1e-6)
                acc_cmd_z = nav_coeff * v_close * (omega_x * rel_pos[1] - omega_y * rel_pos[0]) / (range_to_target + 1e-6)
            else:
                acc_cmd_x = acc_cmd_y = acc_cmd_z = 0.0

            # Note: `damping_ratio` is the autopilot's *dimensionless* damping ratio ζ
            # (a control-loop parameter), NOT a body-velocity friction coefficient.
            # Applying it as drag would catastrophically decelerate the missile
            # (e.g. with ζ=0.3 and v=960m/s → −288m/s² drag → ~40km miss).
            # We instead model it as a first-order autopilot lag that attenuates
            # the commanded acceleration response.  ζ ∈ [0.2, 1.0] gives lag ∈ [0, 0.5].
            autopilot_lag = max(0.0, min(0.5, 0.5 * (1.0 - damping)))
            ax = acc_cmd_x * (1.0 - autopilot_lag)
            ay = acc_cmd_y * (1.0 - autopilot_lag) - 9.81  # add gravity for realism
            az = acc_cmd_z * (1.0 - autopilot_lag)

            # Clip total commanded acceleration to a realistic missile limit (~30 g)
            a_mag = (ax**2 + ay**2 + az**2) ** 0.5
            a_max = 300.0
            if a_mag > a_max:
                scale = a_max / a_mag
                ax *= scale; ay *= scale; az *= scale

            control_energies.append(acc_cmd_x**2 + acc_cmd_y**2 + acc_cmd_z**2)

            x = x + vx * dt
            y = y + vy * dt
            z = z + vz * dt
            vx = vx + ax * dt
            vy = vy + ay * dt
            vz = vz + az * dt
            
            # 更新目标位置 (匀速直线运动，复杂机动可在后续细化)
            target_x += target_vx * dt
            target_y += target_vy * dt
            target_z += target_vz * dt

            # Early termination: if we are within hit radius, stop the integration.
            # This produces a proper "miss distance at closest approach" metric.
            if range_to_target < 5.0:
                break

        miss_distance = ((x - target_x) ** 2 + (y - target_y) ** 2 + (z - target_z) ** 2) ** 0.5
        control_energy = sum(control_energies) * dt

        positions = [t["y"] for t in trajectory]
        max_overshoot = max(abs(max(positions) - target_y), abs(min(positions))) if positions else 0

        settling_idx = len(trajectory) - 1
        for i, t in enumerate(trajectory):
            if abs(t["y"] - target_y) > 0.05:
                settling_idx = i
        settling_time = settling_idx * dt

        return SimulationResult(
            miss_distance=miss_distance,
            control_energy=control_energy,
            max_overshoot=max_overshoot,
            settling_time=settling_time,
            final_position=[x, y, z],
            final_velocity=[vx, vy, vz],
            trajectory=trajectory,
            execution_time=0,
            success=True,
        )

    async def run_external_simulation(
        self,
        script_path: str,
    ) -> SimulationResult:
        """使用 Octave 或 MATLAB 执行 .m 脚本"""

        def _failure_result(error_msg: str) -> SimulationResult:
            return SimulationResult(
                miss_distance=float('nan'),
                control_energy=float('nan'),
                max_overshoot=float('nan'),
                settling_time=float('nan'),
                final_position=[float('nan'), float('nan')],
                final_velocity=[float('nan'), float('nan')],
                trajectory=[],
                execution_time=0,
                success=False,
                error=error_msg,
            )

        if not os.path.exists(script_path):
            return _failure_result(f"Script not found: {script_path}")

        try:
            import subprocess

            # Use source() in Octave: it executes the file in the current
            # scope without requiring the basename to be a valid identifier
            # (which run() does and which mis-parses unusual filenames).
            script_path_fwd = str(script_path).replace("\\", "/")

            # ── Branch 1: in-process MATLAB Engine API for Python ───────
            # Skips the (5–15 s on Windows) subprocess startup cost and
            # keeps a warm MATLAB session for subsequent calls in the
            # same Python process — particularly valuable for RL training
            # and parameter studies.
            # Skip matlab_engine on Python 3.13+ (unsupported by MATLAB R2025a)
            import sys as _sys
            _use_engine = self.engine == "matlab_engine" and _sys.version_info < (3, 13)

            if _use_engine:
                backend = self.matlab_engine_backend
                if backend is not None:
                    # run_script() is synchronous (blocks on future.result()).
                    # Run it in a thread-pool executor so the asyncio event loop
                    # is NOT blocked while MATLAB executes the script.
                    _loop = asyncio.get_event_loop()
                    _tsc  = float(DEFAULT_SUBPROCESS_TIMEOUT_SEC)
                    stdout, stderr, ok = await _loop.run_in_executor(
                        None,
                        lambda: backend.run_script(
                            script_path,
                            call_name=None,
                            timeout_sec=_tsc,
                        ),
                    )
                    if not ok:
                        return _failure_result(
                            f"matlab.engine error: {(stderr or '')[:500]}"
                        )
                    return self._parse_external_output(stdout)
                # Backend failed to start — fall back to subprocess MATLAB
                logger.warning(
                    "matlab.engine backend unavailable; falling back to "
                    "MATLAB subprocess."
                )
                cmd = [self.matlab_path, "-batch", f"run('{script_path_fwd}')"]
                exe_name = "MATLAB"
            elif self.engine == "octave":
                # OCTAVE_BATCH_FLAGS/PREFIX disable GUI init + figure rendering
                # which is the dominant runtime cost for the KB template's
                # ~30 figure() calls per execution.
                cmd = [
                    self.octave_path, *OCTAVE_BATCH_FLAGS, "--eval",
                    build_octave_eval_string(script_path_fwd),
                ]
                exe_name = "Octave"
            elif self.engine in ("matlab", "matlab_engine"):
                # "matlab_engine" reaches here when engine API is skipped
                # (Python 3.13+) or backend failed to start.
                cmd = [self.matlab_path, "-batch", f"run('{script_path_fwd}')"]
                exe_name = "MATLAB"
            else:
                return _failure_result(f"Unsupported engine: {self.engine}")

            # Use Popen + async poll instead of blocking subprocess.run() so
            # KeyboardInterrupt can be delivered between polls.  Python 3.13
            # on Windows has a ProactorEventLoop bug with asyncio subprocess
            # pipes, so we stay with synchronous Popen.
            _poll_sec = 2.0
            from multi_agent.simulation.sim_timeout import get_matlab_timeout_sec

            _timeout = get_matlab_timeout_sec(None)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            import time as _time_gs
            _t0 = _time_gs.monotonic()
            while proc.poll() is None:
                if _time_gs.monotonic() - _t0 > _timeout:
                    proc.kill(); proc.wait(timeout=5)
                    return _failure_result(f"{exe_name} timed out after {_timeout}s")
                await asyncio.sleep(_poll_sec)
            _enc = locale.getpreferredencoding(False) or "utf-8"
            _stdout = (proc.stdout.read() or b"").decode(_enc, errors="replace")
            _stderr = (proc.stderr.read() or b"").decode(_enc, errors="replace")

            if proc.returncode != 0:
                return _failure_result(f"{exe_name} error: {_stderr[:500]}")

            # Parse output for metrics
            return self._parse_external_output(_stdout)

        except FileNotFoundError:
            logger.warning(f"{self.engine} not found, falling back to Python")
            return await self._python_simulation(GuidanceParameters(), 100.0, 0.01)
        except (asyncio.CancelledError, KeyboardInterrupt):
            try:
                proc.kill(); proc.wait(timeout=5)
            except Exception:
                pass
            raise
        except Exception as e:
            import traceback as _tb
            try:
                proc.kill(); proc.wait(timeout=5)
            except Exception:
                pass
            logger.error(
                f"{self.engine} execution failed: {type(e).__name__}: {e}\n"
                f"  traceback: {_tb.format_exc()[-800:]}"
            )
            return _failure_result(str(e))

    def _parse_external_output(self, stdout: str) -> SimulationResult:
        """解析 Octave/MATLAB 输出来提取指标"""
        miss_distance = float('nan')
        control_energy = float('nan')
        success = False

        # 使用正则表达式匹配多种格式
        miss_pattern = re.compile(r'miss[_\s]*distance\s*[=:]*\s*([-+]?\d*\.?\d+)', re.IGNORECASE)
        miss_min_pattern = re.compile(r'miss_min\s*[=:]\s*([-+]?\d*\.?\d+)', re.IGNORECASE)
        energy_pattern = re.compile(r'control[_\s]*energy\s*[=:]*\s*([-+]?\d*\.?\d+)', re.IGNORECASE)

        for line in stdout.split('\n'):
            # 尝试匹配 miss distance 或 miss_min
            if miss_distance is None or (isinstance(miss_distance, float) and math.isnan(miss_distance)):
                m = miss_pattern.search(line)
                if m:
                    try:
                        miss_distance = float(m.group(1))
                        success = True
                    except ValueError:
                        pass
                else:
                    mm = miss_min_pattern.search(line)
                    if mm:
                        try:
                            miss_distance = float(mm.group(1))
                            success = True
                        except ValueError:
                            pass

            # 尝试匹配 control energy
            if math.isnan(control_energy if isinstance(control_energy, float) else 0):
                e = energy_pattern.search(line)
                if e:
                    try:
                        control_energy = float(e.group(1))
                    except ValueError:
                        pass

        return SimulationResult(
            miss_distance=miss_distance,
            control_energy=control_energy,
            max_overshoot=0.0,
            settling_time=0.0,
            final_position=[0, 0],
            final_velocity=[0, 0],
            trajectory=[],
            execution_time=0.0,
            success=success,
        )


class GuidanceSimulator:
    """
    制导系统仿真器
    整合 SysML 模型生成、MATLAB 脚本生成、仿真执行
    """

    def __init__(
        self,
        output_dir: str = "./guidance_workspace",
        engine: str = "octave",  # "python" | "octave" | "matlab"
        octave_path: str = "octave",
        matlab_path: str = "matlab",
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.sysml_dir = os.path.join(output_dir, "sysml_models")
        self.matlab_dir = os.path.join(output_dir, "matlab_scripts")
        self.results_dir = os.path.join(output_dir, "results")

        for d in [self.sysml_dir, self.matlab_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)

        self.sysml_generator = SysMLModelGenerator(self.sysml_dir)
        self.matlab_generator = MATLABScriptGenerator(self.matlab_dir)
        self.executor = SimulationExecutor(
            engine=engine,
            octave_path=octave_path,
            matlab_path=matlab_path,
            output_dir=self.results_dir,
        )

    async def generate_and_simulate(
        self,
        params: GuidanceParameters,
        duration: float = 100.0,
        dt: float = 0.01,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        objectives: Optional[List[str]] = None,
        generate_sysml: bool = True,
        generate_matlab: bool = True,
    ) -> Dict[str, Any]:
        """
        完整的生成-仿真流程

        1. 生成 SysML 模型
        2. 生成 MATLAB 脚本
        3. 执行仿真
        4. 返回结果
        """

        results = {
            "timestamp": datetime.now().isoformat(),
            "parameters": params.to_dict(),
            "simulation_config": {"duration": duration, "dt": dt},
            "generated_files": {},
            "simulation_result": None,
        }

        if generate_sysml:
            sysml_files = self.sysml_generator.save_model(params)
            results["generated_files"]["sysml"] = sysml_files

        script_path = None
        if generate_matlab:
            matlab_files = self.matlab_generator.save_scripts(
                params, duration, dt, param_ranges, objectives
            )
            results["generated_files"]["matlab"] = matlab_files
            script_path = matlab_files.get("guidance_simulation.m")

        sim_result = await self.executor.run_simulation(params, duration, dt, script_path)
        results["simulation_result"] = {
            "miss_distance": sim_result.miss_distance,
            "control_energy": sim_result.control_energy,
            "max_overshoot": sim_result.max_overshoot,
            "settling_time": sim_result.settling_time,
            "final_position": sim_result.final_position,
            "execution_time": sim_result.execution_time,
            "success": sim_result.success,
            "error": sim_result.error,
        }

        results_file = os.path.join(
            self.results_dir, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Simulation complete. Results saved to: {results_file}")

        return results

    async def parameter_study(
        self,
        param_grid: Dict[str, List[float]],
        duration: float = 100.0,
        dt: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """
        参数研究：遍历参数网格，评估所有组合
        """
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        results = []
        for combo in combinations:
            params_dict = dict(zip(keys, combo))
            params = GuidanceParameters(**params_dict)

            result = await self.generate_and_simulate(params, duration, dt)
            results.append(
                {
                    "parameters": params_dict,
                    "metrics": result["simulation_result"],
                }
            )

        return results
