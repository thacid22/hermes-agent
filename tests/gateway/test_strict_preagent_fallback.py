"""Fail-closed policy for gateway fallback before AIAgent exists."""

from unittest.mock import patch

import pytest

from gateway import run as gateway_run
from hermes_cli.auth import AuthError


_STRICT_CHAIN = [{
    "provider": "openrouter",
    "model": "openrouter/auto",
    "activate_on": ["timeout", "overloaded", "server_error", "rate_limit"],
}]


def _fallback_runtime(**_kwargs):
    return {
        "api_key": "probe-key",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "requested_provider": "openrouter",
        "api_mode": "chat_completions",
        "credential_pool": None,
    }


def test_gateway_strict_preagent_fallback_denies_auth_error():
    calls = []

    def resolve_runtime_provider(**kwargs):
        calls.append(kwargs.get("requested"))
        if kwargs.get("requested") is None:
            raise AuthError("primary token rejected with status 401")
        return _fallback_runtime(**kwargs)

    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        side_effect=resolve_runtime_provider,
    ), patch(
        "gateway.run._load_gateway_runtime_config",
        return_value={"fallback_providers": _STRICT_CHAIN},
    ), patch(
        "hermes_cli.runtime_provider._get_model_config",
        return_value={},
    ):
        with pytest.raises(RuntimeError, match="401"):
            gateway_run._resolve_runtime_agent_kwargs()

    assert calls == [None]


def test_gateway_strict_preagent_fallback_allows_rate_limit():
    with patch(
        "gateway.run._load_gateway_runtime_config",
        return_value={"fallback_providers": _STRICT_CHAIN},
    ), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        side_effect=_fallback_runtime,
    ) as resolver:
        runtime = gateway_run._try_resolve_fallback_provider(reason="rate_limit")

    assert runtime is not None
    assert runtime["provider"] == "openrouter"
    resolver.assert_called_once()


def test_gateway_legacy_preagent_fallback_keeps_broad_auth_behavior():
    legacy = [{"provider": "openrouter", "model": "openrouter/auto"}]
    with patch(
        "gateway.run._load_gateway_runtime_config",
        return_value={"fallback_providers": legacy},
    ), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        side_effect=_fallback_runtime,
    ):
        runtime = gateway_run._try_resolve_fallback_provider(reason="auth")

    assert runtime is not None
    assert runtime["provider"] == "openrouter"
