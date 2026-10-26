"""#64 — failure copy must not name a vendor that never answered.

``provider_label("")`` answers ``"OpenRouter"``, which is right for ``hermes status`` ("what would
be used if nothing is configured") and wrong for a failed turn ("who answered this call"). The copy
layer reused it, so a turn routed to an endpoint with no OpenRouter key anywhere was reported as
"isn't available on OpenRouter" — pointing the user at the wrong service and hiding the routing
fault. Nobody has to touch ``provider_label`` for this: the distinction belongs to the caller.
"""

from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason
from agent.error_surface import _provider_label
from agent.turn_failure_copy import PROVIDER_UNKNOWN_LABEL, nonretryable_copy, provider_label_for


class TestProviderLabelFor:
    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_is_neutral(self, blank):
        assert provider_label_for(blank) == PROVIDER_UNKNOWN_LABEL

    def test_the_neutral_label_names_no_vendor(self):
        assert "openrouter" not in PROVIDER_UNKNOWN_LABEL.lower()

    @pytest.mark.parametrize("provider,expected", [("deepseek", "DeepSeek"), ("openrouter", "OpenRouter")])
    def test_a_recorded_provider_keeps_its_real_name(self, provider, expected):
        assert provider_label_for(provider) == expected

    def test_status_semantics_are_untouched(self):
        """``hermes status`` depends on this default; the fix must not migrate into it."""
        from hermes_cli.models import provider_label

        assert provider_label("") == "OpenRouter"


class TestErrorSurface:
    def test_blank_is_neutral(self):
        assert _provider_label("") == PROVIDER_UNKNOWN_LABEL

    def test_recorded_provider_is_labelled(self):
        assert _provider_label("deepseek") == "DeepSeek"


class TestFailureCopyEndToEnd:
    def _copy(self, provider):
        classified = SimpleNamespace(reason=FailoverReason.model_not_found, is_auth=False)
        return nonretryable_copy(classified, provider=provider, model="qwen38-27b", summary="nope")

    def test_unknown_provider_sentence_names_no_vendor(self):
        text = self._copy(None)
        assert "OpenRouter" not in text
        assert PROVIDER_UNKNOWN_LABEL in text

    def test_known_provider_sentence_still_names_it(self):
        text = self._copy("deepseek")
        assert "DeepSeek" in text
        assert "OpenRouter" not in text
