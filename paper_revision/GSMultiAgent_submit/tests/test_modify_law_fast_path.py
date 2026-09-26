"""Tests for T4 MODIFY_LAW fast-path gating."""

from multi_agent.config_loader import reload_config
from multi_agent.integration.modify_law_fast_path import should_use_modify_law_t4_fast_path


def test_fast_path_enabled_for_t4_modify_law_iter1():
    reload_config()
    assert should_use_modify_law_t4_fast_path(
        task_mode="MODIFY_LAW",
        iteration=1,
        mission_conditions_str="RUN_CASE='T';SUB_IDX=4;",
        task_prompt="T4 工况",
        enabled=True,
    )


def test_fast_path_disabled_iter2():
    reload_config()
    assert not should_use_modify_law_t4_fast_path(
        task_mode="MODIFY_LAW",
        iteration=2,
        mission_conditions_str="RUN_CASE='T';SUB_IDX=4;",
        task_prompt="T4",
        enabled=True,
    )


def test_fast_path_disabled_tune_params():
    reload_config()
    assert not should_use_modify_law_t4_fast_path(
        task_mode="TUNE_PARAMS",
        iteration=1,
        mission_conditions_str="RUN_CASE='T';SUB_IDX=4;",
        task_prompt="T4",
        enabled=True,
    )


def test_fast_path_disabled_when_workflow_flag_off():
    reload_config()
    assert not should_use_modify_law_t4_fast_path(
        task_mode="MODIFY_LAW",
        iteration=1,
        mission_conditions_str="RUN_CASE='T';SUB_IDX=4;",
        task_prompt="T4",
        enabled=False,
    )


def test_fast_path_disabled_under_correction_ablation():
    reload_config()
    from multi_agent.config_loader import get_config

    get_config().ablation.syntax_check = False
    assert not should_use_modify_law_t4_fast_path(
        task_mode="MODIFY_LAW",
        iteration=1,
        mission_conditions_str="RUN_CASE='T';SUB_IDX=4;",
        task_prompt="T4",
        enabled=True,
    )
    reload_config()
