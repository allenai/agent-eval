"""Utilities for computing model usages and costs from Inspect eval logs."""

from logging import getLogger

from inspect_ai.log import EvalSample, ModelEvent, StepEvent
from inspect_ai.model import ModelUsage
from litellm import cost_per_token
from litellm.types.utils import PromptTokensDetails, PromptTokensDetailsWrapper, Usage
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
    Handles cached tokens via litellm Usage object.
    """
    total_cost = 0.0
    for model_usage in model_usages:
        input_tokens = model_usage.usage.input_tokens
        output_tokens = model_usage.usage.output_tokens

        cache_read_input_tokens = model_usage.usage.input_tokens_cache_read or 0
        cache_write_input_tokens = model_usage.usage.input_tokens_cache_write or 0

        try:
            # input tokens count includes any cached tokens
            if (
                input_tokens
                == model_usage.usage.total_tokens - model_usage.usage.output_tokens
            ):
                text_tokens = input_tokens - cache_read_input_tokens
                prompt_tokens = input_tokens

            # (anthropic) input tokens count excludes cache read and cache write tokens
            elif (
                input_tokens
                == model_usage.usage.total_tokens
                - output_tokens
                - cache_read_input_tokens
                - cache_write_input_tokens
            ):
                text_tokens = input_tokens
                prompt_tokens = (
                    input_tokens + cache_read_input_tokens + cache_write_input_tokens
                )

            else:
                raise ValueError(
                    f"Model usage token counts don't follow expected pattern."
                )

            prompt_tokens_wrapper = PromptTokensDetailsWrapper(
                cached_tokens=cache_read_input_tokens, text_tokens=text_tokens
            )

            litellm_usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=output_tokens,
                total_tokens=model_usage.usage.total_tokens,
                reasoning_tokens=model_usage.usage.reasoning_tokens,
                prompt_tokens_details=prompt_tokens_wrapper,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_write_input_tokens,
            )

            prompt_cost, completion_cost = cost_per_token(
                model=model_usage.model,
                usage_object=litellm_usage,
            )

            total_cost += prompt_cost + completion_cost
        except Exception as e:
            total_cost = None
            logger.warning(
                f"Problem calculating cost for model {model_usage.model}: {e}"
            )
            break
    return total_cost
