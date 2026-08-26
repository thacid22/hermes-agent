"""Tests for hermes_cli/fallback_config.py — fallback entry API-key resolution."""

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.fallback_config import fallback_entry_allows_reason, resolve_entry_api_key


class TestResolveEntryApiKey:
    def test_inline_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"provider": "custom", "api_key": "inline-key", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "inline-key"


    def test_no_key_fields_returns_none(self):
        assert resolve_entry_api_key({"provider": "openrouter", "model": "glm"}) is None


    def test_whitespace_inline_key_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"api_key": "   ", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "env-key"

    def test_key_env_resolves_from_active_secret_scope_not_raw_env(self, monkeypatch):
        # Multiplexed gateway: os.environ holds another profile's key, but the
        # active per-turn secret scope holds this profile's key. The scoped
        # value must win — a raw os.getenv() would leak the other profile's
        # credential (issue #74311).
        monkeypatch.setenv("FB_KEY", "fake-other-profile-key")
        token = set_secret_scope({"FB_KEY": "fake-active-profile-key"})
        try:
            assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "fake-active-profile-key"
        finally:
            reset_secret_scope(token)

    def test_key_env_falls_back_to_env_when_no_active_scope(self, monkeypatch):
        # Non-multiplexed / single-profile behavior must be unchanged: with no
        # secret scope installed, resolution still reads os.environ.
        monkeypatch.setenv("FB_KEY", "env-key")
        assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "env-key"


class TestFallbackActivationPolicy:
    def test_legacy_entry_without_policy_allows_existing_behavior(self):
        entry = {"provider": "openrouter", "model": "openrouter/auto"}
        assert fallback_entry_allows_reason(entry, None) is True
        assert fallback_entry_allows_reason(entry, "format_error") is True

    def test_strict_policy_fails_closed_without_an_explicit_reason(self):
        entry = {
            "provider": "openrouter",
            "model": "openrouter/auto",
            "activate_on": ["timeout", "overloaded", "server_error", "rate_limit"],
        }
        assert fallback_entry_allows_reason(entry, None) is False

    def test_strict_policy_allows_only_named_availability_reasons(self):
        entry = {
            "provider": "openrouter",
            "model": "openrouter/auto",
            "activate_on": ["timeout", "overloaded", "server_error", "rate_limit"],
        }
        for reason in ("timeout", "overloaded", "server_error", "rate_limit"):
            assert fallback_entry_allows_reason(entry, reason) is True
        assert fallback_entry_allows_reason(entry, "upstream_rate_limit") is True
        for reason in (
            "format_error",
            "content_policy_blocked",
            "context_overflow",
            "model_not_found",
            "auth",
            "billing",
            "unknown",
        ):
            assert fallback_entry_allows_reason(entry, reason) is False
