"""Tests for collect_model_usage function."""

from inspect_ai.log import ModelEvent, ScoreEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.model import GenerateConfig, ModelOutput, ModelUsage
from inspect_ai.scorer import Score

from agenteval.log import collect_model_usage


def _make_model_event() -> ModelEvent:
    return ModelEvent(
        model="test_model",
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=ModelOutput(model="test_model", usage=ModelUsage(input_tokens=10)),
    )


def test_collect_model_usage_no_scorer_spans():
    """Test collecting model usage when there are no scorer spans."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None, name="test_span"),
        _make_model_event(),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 1
    assert result[0].model == "test_model"
    assert result[0].usage.input_tokens == 10


def test_collect_model_usage_with_scorer_span():
    """Test that model events within scorer spans are excluded."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None, name="test_span"),
        _make_model_event(),
        ScoreEvent(score=Score(value="test_score")),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 0


def test_collect_model_usage_nested_spans():
    """Test model usage collection with nested spans."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None, name="test_span"),
        _make_model_event(),
        SpanBeginEvent(id="span2", parent_id="span1", name="test_span2"),
        ScoreEvent(score=Score(value="test_score")),
        _make_model_event(),
        SpanEndEvent(id="span2"),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 0


def test_collect_model_usage_mixed_spans():
    """Test with both scorer and non-scorer spans."""
    events = [
        # Non-scorer span
        SpanBeginEvent(id="span1", parent_id=None, name="test_span1"),
        _make_model_event(),
        SpanEndEvent(id="span1"),
        # Scorer span
        SpanBeginEvent(id="span2", parent_id=None, name="test_span2"),
        _make_model_event(),
        ScoreEvent(score=Score(value="test_score")),
        SpanEndEvent(id="span2"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 1
    assert result[0].model == "test_model"
    assert result[0].usage.input_tokens == 10


def test_collect_model_usage_model_after_score_event():
    """Test that model events after ScoreEvent in same span are excluded."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None, name="test_span1"),
        ScoreEvent(score=Score(value="test_score")),
        _make_model_event(),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 0


def test_collect_model_usage_empty_events():
    """Test with empty events list."""
    result = collect_model_usage([])
    assert len(result) == 0
