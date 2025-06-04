"""Utilities for computing model usages and costs from Inspect eval logs."""

from logging import getLogger

from inspect_ai.log import EvalSample, ModelEvent, StepEvent
from inspect_ai.model import ModelUsage
from litellm import cost_per_token
from pydantic import BaseModel

logger = getLogger(__name__)


class ModelUsageWithName(BaseModel):
    """ModelUsage with model name information."""

    model: str
    usage: ModelUsage


def collect_model_usage(sample: EvalSample) -> list[ModelUsageWithName]:
    """
    Collect model usage for a single sample, excluding scorer model calls.
    Returns a list of ModelUsageWithName objects.
    """
    usages = []
    for event in sample.events:
        if isinstance(event, StepEvent) and event.type == "scorer":
            break
        if isinstance(event, ModelEvent) and event.output and event.output.usage:
            usages.append(
                ModelUsageWithName(model=event.output.model, usage=event.output.usage)
            )

    return usages


def compute_model_cost(model_usages: list[ModelUsageWithName]) -> float:
    """
    Compute aggregate cost for a list of ModelUsageWithName objects.
    Takes into account cached input tokens through cost_per_token's cache_read_input_tokens parameter.
    """
    total_cost = 0.0
    for model_usage in model_usages:
        try:
            prompt_cost, completion_cost = cost_per_token(
                model=model_usage.model,
                prompt_tokens=model_usage.usage.input_tokens,
                completion_tokens=model_usage.usage.output_tokens,
                cache_read_input_tokens=model_usage.usage.input_tokens_cache_read,
            )
            total_cost += prompt_cost + completion_cost
        except Exception as e:
            total_cost = None
            logger.warning(
                f"Problem calculating cost for model {model_usage.model}: {e}"
            )
            break
    return total_cost
