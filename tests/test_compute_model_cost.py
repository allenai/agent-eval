"""Tests for compute_model_cost token-count conventions."""

import pytest
from inspect_ai.model import ModelUsage
from litellm import model_cost
from litellm.types.utils import Usage

import agenteval.log as log_module
from agenteval.log import (
    CostAccountingError,
    ModelUsageWithName,
    compute_model_cost,
)

PROXY_MODEL = "osd-proxy/gpt-5.6-sol"


def _cost(model: str, **usage) -> float:
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


def test_proxy_usage_with_cache_in_total_prices_cache_on_top():
    # The osd-proxy sample re-emitted per the Inspect convention: total
    # includes cache reads, so the Anthropic-style branch prices it.
    input_rate, cache_rate, output_rate = _rates("gpt-5.6-sol")
    cost = _cost(
        PROXY_MODEL,
        input_tokens=9593,
        output_tokens=3785,
        total_tokens=13378 + 56064,
        input_tokens_cache_read=56064,
    )
    expected = 9593 * input_rate + 56064 * cache_rate + 3785 * output_rate
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


@pytest.mark.parametrize("model", ["gpt-5.6-sol", PROXY_MODEL])
def test_cache_read_exceeding_input_fails_loudly(model):
    # The original osd-proxy counts: no model gets an exception list; the
    # ambiguity stops scoring instead of blanking or going negative.
    with pytest.raises(CostAccountingError, match="exceed input tokens"):
        _cost(
            model,
            input_tokens=9593,
            output_tokens=3785,
            total_tokens=13378,
            input_tokens_cache_read=56064,
        )


def test_invalid_model_usage_fails_aggregate_cost():
    valid = ModelUsageWithName(
        model="gpt-5.6-sol",
        usage=ModelUsage(input_tokens=1_000, output_tokens=500, total_tokens=1_500),
    )
    invalid = ModelUsageWithName(
        model="gpt-5.6-sol",
        usage=ModelUsage(
            input_tokens=1_000,
            output_tokens=500,
            total_tokens=1_500,
            input_tokens_cache_read=5_000,
        ),
    )
    assert compute_model_cost([valid]) > 0
    for usages in ([valid, invalid], [invalid, valid]):
        with pytest.raises(CostAccountingError):
            compute_model_cost(usages)


@pytest.mark.parametrize("cache_read,cache_write", [(800, 800), (0, 1001)])
def test_combined_cache_counts_exceeding_input_fail_loudly(cache_read, cache_write):
    with pytest.raises(CostAccountingError, match="exceed input tokens"):
        _cost(
            "gpt-5.6-sol",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            input_tokens_cache_read=cache_read,
            input_tokens_cache_write=cache_write,
        )


def test_unknown_model_price_fails_loudly():
    with pytest.raises(CostAccountingError, match="model unknown-price-model, usage"):
        _cost(
            "unknown-price-model",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
        )


def test_unrecognized_token_pattern_fails_loudly():
    with pytest.raises(CostAccountingError, match="don't follow expected pattern"):
        _cost(
            "gpt-5.6-sol",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1600,
        )


@pytest.mark.parametrize(
    "field",
    [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_tokens_cache_read",
        "input_tokens_cache_write",
        "reasoning_tokens",
    ],
)
def test_negative_token_counts_fail_loudly(field):
    usage = dict(input_tokens=1000, output_tokens=500, total_tokens=1500)
    usage[field] = -1
    with pytest.raises(CostAccountingError, match="Negative token counts"):
        _cost("gpt-5.6-sol", **usage)


@pytest.mark.parametrize("cost", [-1.0, float("nan"), float("inf")])
def test_invalid_price_result_fails_loudly(monkeypatch, cost):
    monkeypatch.setattr(log_module, "cost_per_token", lambda **_: (cost, 0.0))
    with pytest.raises(CostAccountingError):
        _cost("gpt-5.6-sol", input_tokens=1000, output_tokens=500, total_tokens=1500)


def test_gemini_cache_categories_cannot_exceed_input():
    with pytest.raises(CostAccountingError, match="Cache categories exceed"):
        _cost(
            "gemini/gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            reasoning_tokens=400,
            total_tokens=1900,
            input_tokens_cache_read=800,
            input_tokens_cache_write=800,
        )


def test_reasoning_cannot_exceed_inclusive_output():
    with pytest.raises(CostAccountingError, match="Reasoning tokens exceed"):
        _cost(
            "gpt-5.6-sol",
            input_tokens=1000,
            output_tokens=500,
            reasoning_tokens=600,
            total_tokens=1500,
        )


def test_custom_cached_model_cannot_have_negative_uncached_input():
    with pytest.raises(CostAccountingError, match="Derived uncached input"):
        _cost(
            "moonshotai/kimi-k2.5-0127",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            input_tokens_cache_read=1001,
        )


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


def test_gemini_style_output_excludes_reasoning(monkeypatch):
    captured_usage = []
    original_cost_per_token = log_module.cost_per_token

    def capture_cost_per_token(model: str, usage_object: Usage) -> tuple[float, float]:
        captured_usage.append(usage_object)
        return original_cost_per_token(model=model, usage_object=usage_object)

    monkeypatch.setattr(log_module, "cost_per_token", capture_cost_per_token)
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
    assert len(captured_usage) == 1
    assert captured_usage[0].prompt_tokens == 1_000
    assert captured_usage[0].completion_tokens == 500
    assert captured_usage[0].total_tokens == 1_500
    assert captured_usage[0].completion_tokens_details.reasoning_tokens == 300
