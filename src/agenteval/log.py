"""Utilities for computing model usages and costs from Inspect eval logs."""

from logging import getLogger

from inspect_ai.log import (
    Event,
    ModelEvent,
    ScoreEvent,
    SpanBeginEvent,
    SpanEndEvent,
    StepEvent,
)
from inspect_ai.model import ModelUsage
from litellm import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage
from pydantic import BaseModel

from .local_cost import CUSTOM_PRICING

logger = getLogger(__name__)

MODEL_TRANSLATIONS = {
    "google:gemini2flash-default": "gemini/gemini-2.0-flash",
    "models/gemini-2.5-flash-preview-05-20": "gemini/gemini-2.5-flash",
    "models/gemini-2.5-pro-preview-06-05": "gemini/gemini-2.5-pro",
    "mistral-large-2411": "vertex_ai/mistral-large-2411",
    "sonar-deep-research": "perplexity/sonar-deep-research",
}


class ModelUsageWithName(BaseModel):
    """ModelUsage with model name information."""

    model: str
    usage: ModelUsage


def collect_model_usage(events: list[Event]) -> list[ModelUsageWithName]:
    """
    Collect model usage for a single sample, excluding scorer model calls.

    Model usage is an event and events are grouped by span ID.
    We want to exclude ModelEvents that are in the same immediate span as ScoreEvent.

    Returns a list of ModelUsageWithName objects.
    """
    # First pass: identify immediate spans that contain ScoreEvents
    active_spans = []  # Stack of currently active span IDs
    scorer_spans = set()  # Set of span IDs that contain score events

    for event in events:
        if isinstance(event, SpanBeginEvent):
            active_spans.append(event.id)
        elif isinstance(event, SpanEndEvent):
            if active_spans and active_spans[-1] == event.id:
                active_spans.pop()
        elif isinstance(event, ScoreEvent) or (
            isinstance(event, StepEvent) and event.type == "scorer"
        ):
            # Mark all currently active spans as scorer spans
            scorer_spans.add(active_spans[-1])

    # Second pass: collect model usage, excluding those in scorer spans
    usages = []
    active_spans = []

    for event in events:
        if isinstance(event, SpanBeginEvent):
            active_spans.append(event.id)
        elif isinstance(event, SpanEndEvent):
            if active_spans and active_spans[-1] == event.id:
                active_spans.pop()
        elif isinstance(event, ModelEvent) and event.output and event.output.usage:
            # Only include if none of the active spans are scorer spans
            if active_spans[-1] not in scorer_spans:
                usages.append(
                    ModelUsageWithName(
                        model=event.output.model, usage=event.output.usage
                    )
                )

    return usages


def adapt_model_name(model: str) -> str:
    """
    Translate provider/model name from inspect logs
    to provider/model name in litellm cost lookup
    """
    if model in MODEL_TRANSLATIONS.keys():
        return MODEL_TRANSLATIONS[model]
    else:
        return model


def compute_model_cost(model_usages: list[ModelUsageWithName]) -> float | None:
    """
    Compute aggregate cost for a list of ModelUsageWithName objects.
    Handles cached tokens via litellm Usage object.
    """
    total_cost: float | None = 0.0
    for model_usage in model_usages:
        input_tokens = model_usage.usage.input_tokens
        output_tokens = model_usage.usage.output_tokens

        try:
            if model_usage.model in CUSTOM_PRICING.keys():

                prompt_cost, completion_cost = cost_per_token(
                    model=model_usage.model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    custom_cost_per_token=CUSTOM_PRICING[model_usage.model],
                )

            else:
                total_tokens = model_usage.usage.total_tokens

                cache_read_input_tokens = model_usage.usage.input_tokens_cache_read or 0
                cache_write_input_tokens = (
                    model_usage.usage.input_tokens_cache_write or 0
                )
                reasoning_tokens = model_usage.usage.reasoning_tokens or 0

                if input_tokens == total_tokens - output_tokens:
                    text_tokens = input_tokens - cache_read_input_tokens
                    prompt_tokens = input_tokens
                    completion_tokens = output_tokens

                # (gemini) output tokens count excludes reasoning tokens
                elif (
                    input_tokens
                    == model_usage.usage.total_tokens - output_tokens - reasoning_tokens
                ):
                    text_tokens = input_tokens
                    prompt_tokens = input_tokens
                    completion_tokens = output_tokens + reasoning_tokens

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
                        input_tokens
                        + cache_read_input_tokens
                        + cache_write_input_tokens
                    )
                    completion_tokens = output_tokens

                else:
                    raise ValueError(
                        f"Model usage token counts don't follow expected pattern."
                    )

                prompt_tokens_wrapper = PromptTokensDetailsWrapper(
                    cached_tokens=cache_read_input_tokens, text_tokens=text_tokens
                )

                litellm_usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=model_usage.usage.total_tokens,
                    reasoning_tokens=model_usage.usage.reasoning_tokens,
                    prompt_tokens_details=prompt_tokens_wrapper,
                    cache_read_input_tokens=cache_read_input_tokens,
                    cache_creation_input_tokens=cache_write_input_tokens,
                )

                prompt_cost, completion_cost = cost_per_token(
                    model=adapt_model_name(model_usage.model),
                    usage_object=litellm_usage,
                )

            if total_cost is not None:
                total_cost += prompt_cost + completion_cost
        except Exception as e:
            total_cost = None
            logger.warning(
                f"Problem calculating cost for model {model_usage.model}: {e}"
            )
            break
    return total_cost
