#!/usr/bin/env python3
"""Tests for DeepSeek thinking-mode helpers used by Expert Tuning."""

import unittest
from types import SimpleNamespace

from multi_agent.llm_utils import (
    describe_chat_choice,
    extract_chat_message_text,
    thinking_extra_body,
    uses_deepseek_thinking,
)


def _resp(*, content=None, reasoning=None, finish="stop", usage=None):
    msg = SimpleNamespace(content=content, reasoning_content=reasoning)
    choice = SimpleNamespace(message=msg, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


class TestDeepSeekThinkingHelpers(unittest.TestCase):
    def test_uses_deepseek_thinking(self):
        self.assertTrue(uses_deepseek_thinking("deepseek-v4-pro"))
        self.assertTrue(uses_deepseek_thinking("deepseek-v4-flash"))
        self.assertFalse(uses_deepseek_thinking("qwen3.7-max"))

    def test_thinking_disabled_extra_body(self):
        self.assertEqual(
            thinking_extra_body("deepseek-v4-pro", enabled=False),
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(thinking_extra_body("kimi-k3"), {})

    def test_extract_prefers_content(self):
        resp = _resp(content='{"w1": 40}', reasoning='thought {"w1": 1}')
        self.assertEqual(extract_chat_message_text(resp), '{"w1": 40}')

    def test_extract_json_from_reasoning_when_content_empty(self):
        resp = _resp(
            content="",
            reasoning='I will tune BW.\n{"w1":45,"zeta1":0.7,"tao1":0.12}',
            finish="length",
        )
        self.assertEqual(
            extract_chat_message_text(resp),
            '{"w1":45,"zeta1":0.7,"tao1":0.12}',
        )

    def test_extract_empty_when_no_json(self):
        resp = _resp(content="  ", reasoning="still thinking about bandwidth")
        self.assertEqual(extract_chat_message_text(resp), "")

    def test_describe_includes_finish_reason(self):
        text = describe_chat_choice(
            _resp(content="", reasoning="x" * 80, finish="length")
        )
        self.assertIn("finish_reason='length'", text)
        self.assertIn("content_len=0", text)
        self.assertIn("reasoning_len=80", text)


if __name__ == "__main__":
    unittest.main()
