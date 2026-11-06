import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest  # noqa: E402
from inspect_ai.model import ModelUsage  # noqa: E402
from litellm import model_cost as litellm_model_cost  # noqa: E402

from agenteval.local_cost import (  # noqa: E402
    CUSTOM_PRICING,
    CUSTOM_PRICING_WITH_CACHE,
)
from agenteval.log import ModelUsageWithName, compute_model_cost  # noqa: E402

# Local supplements may add models the pinned LiteLLM snapshot lacks, but must
# not change a rate LiteLLM already supplies (that would shift existing costs).
NEW_CLAUDE_KEYS = [
    "claude-sonnet-5-5",
    "anthropic/claude-sonnet-5-5",
    "claude-haiku-5-5",
    "anthropic/claude-haiku-5-5",
]


@pytest.mark.parametrize("model", NEW_CLAUDE_KEYS)
def test_supplement_does_not_override_litellm(model):
    assert model not in litellm_model_cost


def test_supplement_tables_do_not_overlap():
    assert not set(CUSTOM_PRICING) & set(CUSTOM_PRICING_WITH_CACHE)


def _cost(model, **usage):
    usage.setdefault("output_tokens", 0)
    usage["total_tokens"] = sum(
        usage.get(k, 0)
        for k in (
            "input_tokens",
            "output_tokens",
            "input_tokens_cache_read",
            "input_tokens_cache_write",
        )
    )
    return compute_model_cost(
        [ModelUsageWithName(model=model, usage=ModelUsage(**usage))]
    )


def test_sonnet_5_5_cost():
    cost = _cost(
        "claude-sonnet-5-5",
        input_tokens=1_000,
        output_tokens=500,
        input_tokens_cache_read=10_000,
        input_tokens_cache_write=2_000,
    )
    assert cost == pytest.approx(
        1_000 * 2e-6 + 500 * 1e-5 + 10_000 * 2e-7 + 2_000 * 2.5e-6
    )


def test_haiku_5_5_short_prompt_uses_base_rates():
    cost = _cost(
        "claude-haiku-5-5",
        input_tokens=50_000,
        output_tokens=1_000,
        input_tokens_cache_read=50_000,
    )
    assert cost == pytest.approx(50_000 * 1e-7 + 1_000 * 5e-7 + 50_000 * 1e-8)


def test_haiku_5_5_long_prompt_uses_long_context_rates():
    # Prompt = input + cache read + cache write = 100_001 > 100K threshold.
    cost = _cost(
        "anthropic/claude-haiku-5-5",
        input_tokens=1,
        output_tokens=1_000,
        input_tokens_cache_read=90_000,
        input_tokens_cache_write=10_000,
    )
    assert cost == pytest.approx(
        1 * 5e-7 + 1_000 * 2.5e-6 + 90_000 * 5e-8 + 10_000 * 6.25e-7
    )
