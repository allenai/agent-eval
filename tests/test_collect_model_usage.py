"""Tests for collect_model_usage function."""

from collections.abc import Generator
from random import Random

import inspect_ai
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import ModelEvent, ScoreEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.model import (
    ChatMessageSystem,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from agenteval.log import collect_model_usage


def _make_model_event(model_name: str) -> ModelEvent:
    return ModelEvent(
        model=model_name,
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=ModelOutput(model=model_name, usage=ModelUsage(input_tokens=10)),
    )


def test_collect_model_usage_no_scorer_spans():
    """Test collecting model usage when there are no scorer spans."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None, name="test_span"),
        _make_model_event("included"),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 1
    assert result[0].model == "included"


def test_collect_model_usage_with_scorer_span():
    """Test that model events within scorer spans are excluded."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None, name="test_span"),
        _make_model_event("excluded"),
        ScoreEvent(score=Score(value="test_score")),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 0


def test_collect_model_usage_nested_spans():
    """Test model usage collection with nested spans."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None, name="test_span"),
        _make_model_event("included"),
        SpanBeginEvent(id="span2", parent_id="span1", name="test_span2"),
        ScoreEvent(score=Score(value="test_score")),
        _make_model_event("excluded"),
        SpanEndEvent(id="span2"),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 1
    assert result[0].model == "included"


def test_collect_model_usage_mixed_spans():
    """Test with both scorer and non-scorer spans."""
    events = [
        # Non-scorer span
        SpanBeginEvent(id="span1", parent_id=None, name="test_span1"),
        _make_model_event("included"),
        SpanEndEvent(id="span1"),
        # Scorer span
        SpanBeginEvent(id="span2", parent_id=None, name="test_span2"),
        _make_model_event("excluded"),
        ScoreEvent(score=Score(value="test_score")),
        SpanEndEvent(id="span2"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 1
    assert result[0].model == "included"


def test_collect_model_usage_model_after_score_event():
    """Test that model events after ScoreEvent in same span are excluded."""
    events = [
        SpanBeginEvent(id="span1", parent_id=None, name="test_span1"),
        ScoreEvent(score=Score(value="test_score")),
        _make_model_event("excluded"),
        SpanEndEvent(id="span1"),
    ]

    result = collect_model_usage(events)
    assert len(result) == 0


def test_collect_model_usage_empty_events():
    """Test with empty events list."""
    result = collect_model_usage([])
    assert len(result) == 0


def test_collect_model_usage_e2e():
    def infinite_random_generator(
        input_tokens: int,
    ) -> Generator[ModelOutput, None, None]:
        rng = Random(0)
        while True:
            output = ModelOutput.from_content(
                model="mockllm/model", content=str(rng.random())
            )
            output.usage = ModelUsage(input_tokens=input_tokens)
            yield output

    @solver
    def test_solver() -> Solver:
        """Simple solver that just runs llm with a given system prompt"""

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            model = get_model(
                "mockllm/model", custom_outputs=infinite_random_generator(10)
            )
            state.output = await model.generate(input="test")
            return state

        return solve

    @scorer(metrics=[accuracy()])
    def test_scorer() -> Scorer:
        """A test scorer"""

        model = get_model("mockllm/model", custom_outputs=infinite_random_generator(42))

        async def score(state: TaskState, target: Target) -> Score:
            output = await model.generate(input="test")
            content = output.choices[0].message.content
            if isinstance(content, str):
                value = float(content)
            else:
                value = 1.0
            return Score(
                value=value,
                explanation="Test scorer completed.",
            )

        return score

    @task
    def test_task() -> Task:
        """A task that uses the exception_solver to test error handling."""

        return Task(
            dataset=MemoryDataset([Sample(id="1", input="Test input")]),
            solver=test_solver(),
            scorer=test_scorer(),
        )

    eval_logs = inspect_ai.eval(
        test_task(),
        model="mockllm/model",
        display="plain",
        log_level="info",
    )

    assert len(eval_logs) > 0
    assert len(eval_logs[0].samples) > 0
    assert len(eval_logs[0].samples[0].events) > 0

    events = eval_logs[0].samples[0].events
    result = collect_model_usage(events)
    assert len(result) > 0
    # Actual model usage is 10, scorer's model usage is 42. Exclude scorer's model usage.
    for r in result:
        assert r.usage.input_tokens == 10
