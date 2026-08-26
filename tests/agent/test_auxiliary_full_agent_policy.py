"""Fail-closed OpenRouter policy for auxiliary full-agent routes."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import background_review, curator, review_engine


def _parent_openrouter():
    agent = SimpleNamespace(
        provider="openrouter",
        model="openrouter/auto",
        client=SimpleNamespace(base_url="https://openrouter.ai/api/v1"),
        _credential_pool=None,
        request_overrides={},
        max_tokens=None,
        acp_command=None,
        acp_args=[],
    )
    agent._current_main_runtime = lambda: {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "probe",
        "api_mode": "chat_completions",
    }
    return agent


def test_background_review_marks_inherited_openrouter_route_blocked():
    with patch("agent.auxiliary_client._aux_openrouter_settings", return_value=(False, False, "openrouter/auto")):
        runtime = background_review._resolve_review_runtime(_parent_openrouter(), {})
    assert runtime["blocked"] is True
    assert runtime["provider"] == "openrouter"


def test_review_dispatch_denies_explicit_openrouter_before_delegate():
    parent = _parent_openrouter()
    with patch("agent.review_engine.snapshot_recent_messages", return_value=[{"role": "user", "content": "x"}]), \
         patch("agent.review_engine.collect_parent_loaded_skills", return_value=[]), \
         patch("agent.review_engine.build_review_task", return_value=("goal", "context")), \
         patch("agent.review_engine._load_review_credentials_cfg", return_value={
             "provider": "openrouter", "model": "openrouter/auto", "base_url": "",
         }), \
         patch("agent.auxiliary_client._aux_openrouter_settings", return_value=(False, False, "openrouter/auto")), \
         patch("tools.delegate_tool.delegate_task") as delegate:
        with pytest.raises(ValueError, match="OpenRouter auxiliary routes are disabled"):
            review_engine.start_review(parent, [{"role": "user", "content": "x"}])
    delegate.assert_not_called()


def test_curator_denies_openrouter_before_agent_construction():
    fake_agent = MagicMock()
    with patch("hermes_cli.config.load_config_readonly", return_value={
        "model": {"provider": "openrouter", "default": "openrouter/auto"},
    }), patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
        "provider": "openrouter",
        "model": "openrouter/auto",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "probe",
    }), patch("agent.auxiliary_client._aux_openrouter_settings", return_value=(False, False, "openrouter/auto")), \
         patch("run_agent.AIAgent", fake_agent):
        result = curator._run_llm_review("review")
    assert "OpenRouter auxiliary routes are disabled" in result["error"]
    fake_agent.assert_not_called()
