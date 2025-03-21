"""Utilities for computing model usages and costs from Inspect eval logs."""

from logging import getLogger

from inspect_ai.log import EvalSample, ModelEvent, StepEvent
from inspect_ai.model import ModelUsage
from litellm import cost_per_token

logger = getLogger(__name__)


def compute_model_usage(sample: EvalSample) -> dict[str, ModelUsage]:
    """
    Compute model usage for a single sample, excluding scorer model calls.
    """
    usage = dict()
    for event in sample.events:
        if isinstance(event, StepEvent) and event.type == "scorer":
            break
        if isinstance(event, ModelEvent) and event.output and event.output.usage:
            if event.output.model not in usage:
                usage[event.output.model] = event.output.usage
            else:
                usage[
                    event.output.model
                ].input_tokens += event.output.usage.input_tokens
                usage[
                    event.output.model
                ].output_tokens += event.output.usage.output_tokens
                usage[
                    event.output.model
                ].total_tokens += event.output.usage.total_tokens
    return usage


def compute_model_cost(model_usage: dict[str, ModelUsage]) -> float:
    """
    Compute aggregate cost for a given dictionary of ModelUsage.
    """
    total_cost = 0.0
    for model, usage in model_usage.items():
        try:
            prompt_cost, completion_cost = cost_per_token(
                model=model,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            )
            total_cost += prompt_cost + completion_cost
        except Exception as e:
            total_cost = float("nan")
            logger.warning(f"Problem calculating cost for model {model}: {e}")
            break
    return total_cost
