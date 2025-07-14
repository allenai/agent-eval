"""Tests for log.py module."""

import pytest
from inspect_ai.log import (
    Event,
    ModelEvent,
    ScoreEvent,
    SpanBeginEvent,
    SpanEndEvent,
    StepEvent,
)
from inspect_ai.model import ModelOutput, ModelUsage

from agenteval.log import collect_model_usage


def test_collect_model_usage_no_scorer_spans():
    """Test collecting model usage when there are no scorer spans."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None),
        ModelEvent(
            output=ModelOutput(
                model="gpt-4",
                usage=ModelUsage(
                    input_tokens=10,
                    output_tokens=20,
                    total_tokens=30,
                ),
                completion="test",
            )
        ),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 1
    assert result[0].model == "gpt-4"
    assert result[0].usage.input_tokens == 10
    assert result[0].usage.output_tokens == 20


def test_collect_model_usage_with_scorer_span():
    """Test that model events within scorer spans are excluded."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None),
        ModelEvent(
            output=ModelOutput(
                model="gpt-4",
                usage=ModelUsage(
                    input_tokens=10,
                    output_tokens=20,
                    total_tokens=30,
                ),
                completion="test",
            )
        ),
        ScoreEvent(score=0.5, value="test"),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 0


def test_collect_model_usage_nested_spans():
    """Test model usage collection with nested spans."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None),
        ModelEvent(
            output=ModelOutput(
                model="gpt-4",
                usage=ModelUsage(
                    input_tokens=10,
                    output_tokens=20,
                    total_tokens=30,
                ),
                completion="test1",
            )
        ),
        SpanBeginEvent(id="span2", parent_id="span1"),
        ScoreEvent(score=0.5, value="test"),
        ModelEvent(
            output=ModelOutput(
                model="gpt-4",
                usage=ModelUsage(
                    input_tokens=15,
                    output_tokens=25,
                    total_tokens=40,
                ),
                completion="test2",
            )
        ),
        SpanEndEvent(id="span2"),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 0


def test_collect_model_usage_mixed_spans():
    """Test with both scorer and non-scorer spans."""
    events = [
        # Non-scorer span
        SpanBeginEvent(id="span1", parent_id=None),
        ModelEvent(
            output=ModelOutput(
                model="gpt-4",
                usage=ModelUsage(
                    input_tokens=10,
                    output_tokens=20,
                    total_tokens=30,
                ),
                completion="included",
            )
        ),
        SpanEndEvent(id="span1"),
        # Scorer span
        SpanBeginEvent(id="span2", parent_id=None),
        ModelEvent(
            output=ModelOutput(
                model="gpt-4",
                usage=ModelUsage(
                    input_tokens=15,
                    output_tokens=25,
                    total_tokens=40,
                ),
                completion="excluded",
            )
        ),
        ScoreEvent(score=0.5, value="test"),
        SpanEndEvent(id="span2"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 1
    assert result[0].usage.input_tokens == 10


def test_collect_model_usage_model_after_score_event():
    """Test that model events after ScoreEvent in same span are excluded."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None),
        ScoreEvent(score=0.5, value="test"),
        ModelEvent(
            output=ModelOutput(
                model="gpt-4",
                usage=ModelUsage(
                    input_tokens=10,
                    output_tokens=20,
                    total_tokens=30,
                ),
                completion="test",
            )
        ),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 0


def test_collect_model_usage_empty_events():
    """Test with empty events list."""
    result = collect_model_usage([])
    assert len(result) == 0
