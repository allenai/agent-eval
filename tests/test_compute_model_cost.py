"""Tests for compute_model_cost token-count conventions."""

import pytest
from inspect_ai.log import (  # type: ignore[attr-defined]
    ModelEvent,
    SpanBeginEvent,
    SpanEndEvent,
)
from inspect_ai.model import GenerateConfig, ModelOutput, ModelUsage
from litellm import model_cost
from litellm.types.utils import Usage

import agenteval.log as log_module
from agenteval.log import (
    ModelUsageWithName,
    collect_model_usage,
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


def test_proxy_name_survives_log_collection_and_is_priced():
    event = ModelEvent(
        model=PROXY_MODEL,
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=ModelOutput(
            model=PROXY_MODEL,
            usage=ModelUsage(
                input_tokens=9593,
                output_tokens=3785,
                total_tokens=13378,
                input_tokens_cache_read=56064,
            ),
        ),
    )
    events = [
        SpanBeginEvent(id="model-call", parent_id=None, name="generate"),
        ModelEvent.model_validate_json(event.model_dump_json()),
        SpanEndEvent(id="model-call"),
    ]
    usages = collect_model_usage(events)
    assert len(usages) == 1
    assert usages[0].model == PROXY_MODEL
    assert compute_model_cost(usages) == pytest.approx(0.189547)


@pytest.mark.parametrize(
    "cache_write_tokens,suffix",
    [(2_000, ""), (271_000, "_above_272k_tokens")],
)
def test_proxy_cache_writes_are_priced_and_count_toward_context_tier(
    cache_write_tokens, suffix
):
    input_rate, cache_rate, output_rate = _rates("gpt-5.6-sol", suffix)
    write_rate = model_cost["gpt-5.6-sol"][f"cache_creation_input_token_cost{suffix}"]
    cost = _cost(
        PROXY_MODEL,
        input_tokens=1_000,
        output_tokens=500,
        total_tokens=1_500,
        input_tokens_cache_read=1_000,
        input_tokens_cache_write=cache_write_tokens,
    )
    expected = (
        1_000 * input_rate
        + 1_000 * cache_rate
        + cache_write_tokens * write_rate
        + 500 * output_rate
    )
    assert cost == pytest.approx(expected)


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


def test_openai_style_cache_read_exceeding_input_blanks_cost(caplog):
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
    assert (
        "Problem calculating cost for model gpt-5.6-sol: Cache read tokens (56064) "
        "exceed input tokens (9593)"
    ) in caplog.text
    assert "Add the model to INPUT_EXCLUDES_CACHE_READ" in caplog.text
    assert caplog.records[-1].levelname == "WARNING"


def test_invalid_model_usage_blanks_aggregate_cost():
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
    assert compute_model_cost([valid, invalid]) is None
    assert compute_model_cost([invalid, valid]) is None


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
