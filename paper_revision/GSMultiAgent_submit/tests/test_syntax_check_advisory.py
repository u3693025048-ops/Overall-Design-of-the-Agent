"""Tests for syntax_check advisory vs blocking issues."""

from multi_agent.tools.syntax_check_tool import SyntaxCheckMATLABTool

_SAMPLE = """\
function demo()
rl_ny_lim = 20;
base=struct('N_pn',rl_N_pn);
end

function [N1,N2,G1,G2,gc]=gf()
% T4 minimal APN (deterministic template)
ny_lim=20;
N1=max(-ny_lim,min(ny_lim,N1));
end
"""


def test_strip_comments_removes_apn_false_call():
    stripped = SyntaxCheckMATLABTool._strip_matlab_comments(_SAMPLE)
    assert "APN (" not in stripped


def test_rl_ny_lim_is_advisory_when_gf_has_ny_lim():
    issue = "第 19 行: RL 参数 'rl_ny_lim' 已声明但未注入 base=struct()"
    assert SyntaxCheckMATLABTool._is_advisory_issue(issue, _SAMPLE)


def test_partition_issues_splits_advisory():
    issues = [
        "第 19 行: RL 参数 'rl_ny_lim' 已声明但未注入 base=struct()",
        "以下函数被调用但未在本文件中找到本地定义: ['APN']",
    ]
    blocking, advisory = SyntaxCheckMATLABTool._partition_issues(issues, _SAMPLE)
    assert len(advisory) == 1
    assert "rl_ny_lim" in advisory[0]
    # APN in comment should not appear after comment strip in real check;
    # partition still classifies unknown-call list as blocking.
    assert len(blocking) == 1


def test_interface_check_no_apn_from_comment():
    issues = SyntaxCheckMATLABTool._check_interface_consistency(_SAMPLE)
    assert not any("APN" in i for i in issues)
