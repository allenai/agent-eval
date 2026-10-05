"""Tests for compute_model_cost token-count conventions."""

import pytest
from inspect_ai.model import ModelUsage
from litellm import model_cost

from agenteval.log import (
    INPUT_EXCLUDES_CACHE_READ,
    ModelUsageWithName,
    compute_model_cost,
)

PROXY_MODEL = "osd-proxy/gpt-5.6-sol"


def _cost(model: str, **usage) -> float | None:
    return compute_model_cost(
        [ModelUsageWithName(model=model, usage=ModelUsage(**usage))]
    )


def _rates(model: str, suffix: str = "") -> tuple[float, float, float]:
    info = model_cost[model]
    return (
        info[f"input_cost_per_token{suffix}"],
        info[f"cache_read_input_token_cost{suffix}"],
        info[f"output_cost_per_token{suffix}"],
    )


def test_proxy_model_is_listed():
    assert PROXY_MODEL in INPUT_EXCLUDES_CACHE_READ


def test_input_excludes_cache_read_prices_cache_on_top():
    # Counts from a real osd-proxy sample: total == input + output, cache reads
    # reported separately and not included in input.
    input_rate, cache_rate, output_rate = _rates("gpt-5.6-sol")
    cost = _cost(
        PROXY_MODEL,
        input_tokens=9593,
        output_tokens=3785,
        total_tokens=13378,
        input_tokens_cache_read=56064,
    )
    expected = 9593 * input_rate + 56064 * cache_rate + 3785 * output_rate
    assert cost == pytest.approx(expected)
    assert cost > 0


def test_input_excludes_cache_read_uses_full_context_for_tier():
    # input + cache_read crosses 272k, so the above-272k rates apply.
    input_rate, cache_rate, output_rate = _rates("gpt-5.6-sol", "_above_272k_tokens")
    cost = _cost(
        PROXY_MODEL,
        input_tokens=10_000,
        output_tokens=1_000,
        total_tokens=11_000,
        input_tokens_cache_read=300_000,
    )
    expected = 10_000 * input_rate + 300_000 * cache_rate + 1_000 * output_rate
    assert cost == pytest.approx(expected)


def test_openai_style_input_includes_cache_read():
    input_rate, cache_rate, output_rate = _rates("gpt-5.6-sol")
    cost = _cost(
        "gpt-5.6-sol",
        input_tokens=60_000,
        output_tokens=1_000,
        total_tokens=61_000,
        input_tokens_cache_read=50_000,
    )
    expected = 10_000 * input_rate + 50_000 * cache_rate + 1_000 * output_rate
    assert cost == pytest.approx(expected)


def test_openai_style_cache_read_exceeding_input_blanks_cost():
    # Same counts as the proxy sample, but for an unlisted model: pricing
    # input - cache_read would go negative, so the cost is blanked instead.
    cost = _cost(
        "gpt-5.6-sol",
        input_tokens=9593,
        output_tokens=3785,
        total_tokens=13378,
        input_tokens_cache_read=56064,
    )
    assert cost is None


def test_anthropic_style_input_excludes_cache_read_and_write():
    model = "claude-sonnet-4-5"
    input_rate, cache_rate, output_rate = _rates(model)
    write_rate = model_cost[model]["cache_creation_input_token_cost"]
    cost = _cost(
        model,
        input_tokens=1_000,
        output_tokens=500,
        total_tokens=8_500,
        input_tokens_cache_read=5_000,
        input_tokens_cache_write=2_000,
    )
    expected = (
        1_000 * input_rate + 5_000 * cache_rate + 2_000 * write_rate + 500 * output_rate
    )
    assert cost == pytest.approx(expected)


def test_gemini_style_output_excludes_reasoning():
    model = "gemini/gemini-2.5-flash"
    input_rate, _, output_rate = _rates(model)
    cost = _cost(
        model,
        input_tokens=1_000,
        output_tokens=200,
        reasoning_tokens=300,
        total_tokens=1_500,
    )
    expected = 1_000 * input_rate + 500 * output_rate
    assert cost == pytest.approx(expected)
