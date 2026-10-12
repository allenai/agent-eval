"""Utilities for computing model usages and costs from Inspect eval logs."""

import re
from logging import getLogger
from math import isfinite
from typing import Any

import litellm
from inspect_ai.log import (  # type: ignore[attr-defined]
    Event,
    ModelEvent,
    ScoreEvent,
    SpanBeginEvent,
    SpanEndEvent,
    StepEvent,
)
from inspect_ai.model import ModelUsage
from litellm import cost_per_token, get_model_info
from litellm.types.utils import PromptTokensDetailsWrapper, Usage
from pydantic import BaseModel

from .local_cost import CUSTOM_PRICING, CUSTOM_PRICING_WITH_CACHE

logger = getLogger(__name__)

MODEL_TRANSLATIONS = {
    "google:gemini2flash-default": "gemini/gemini-2.0-flash",
    "models/gemini-2.5-flash-preview-05-20": "gemini/gemini-2.5-flash",
    "models/gemini-2.5-pro-preview-06-05": "gemini/gemini-2.5-pro",
    "mistral-large-2411": "vertex_ai/mistral-large-2411",
    "osd-proxy/gpt-5.6-sol": "gpt-5.6-sol",
    "sonar-deep-research": "perplexity/sonar-deep-research",
}


class CostAccountingError(ValueError):
    """Usage cannot be priced reliably; scoring must stop."""


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


def _validate_price(rate: Any, field: str) -> float:
    if rate is None:
        raise ValueError(f"Missing required price: {field}")
    rate = float(rate)
    if rate < 0 or not isfinite(rate):
        raise ValueError(f"Invalid price {field}: {rate}")
    return rate


def compute_model_cost(model_usages: list[ModelUsageWithName]) -> float:
    """
    Compute aggregate cost for a list of ModelUsageWithName objects.
    Handles cached tokens via litellm Usage object.
    Raises CostAccountingError on invalid usage or pricing; callers must propagate it.
    """
    total_cost = 0.0
    for model_usage in model_usages:
        input_tokens = model_usage.usage.input_tokens
        output_tokens = model_usage.usage.output_tokens

        try:
            counts = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": model_usage.usage.total_tokens,
                "input_tokens_cache_read": model_usage.usage.input_tokens_cache_read
                or 0,
                "input_tokens_cache_write": model_usage.usage.input_tokens_cache_write
                or 0,
                "reasoning_tokens": model_usage.usage.reasoning_tokens or 0,
            }
            if any(count < 0 for count in counts.values()):
                raise ValueError(f"Negative token counts: {counts}")

            total_tokens = model_usage.usage.total_tokens

            cache_read_input_tokens = model_usage.usage.input_tokens_cache_read or 0
            cache_write_input_tokens = model_usage.usage.input_tokens_cache_write or 0
            reasoning_tokens = model_usage.usage.reasoning_tokens or 0

            if input_tokens == total_tokens - output_tokens:
                # Legacy inclusive input counts contain both cache categories.
                if cache_read_input_tokens + cache_write_input_tokens > input_tokens:
                    raise ValueError(
                        f"Model {model_usage.model}: cache read/write tokens "
                        f"({cache_read_input_tokens} + {cache_write_input_tokens}) "
                        f"exceed input tokens "
                        f"({input_tokens}) but total_tokens ({total_tokens}) "
                        f"excludes them, so the usage convention is ambiguous. "
                        f"Re-emit usage with total_tokens including cache "
                        f"reads/writes (Inspect ModelUsage convention)."
                    )
                prompt_tokens = input_tokens
                completion_tokens = output_tokens

            # (gemini) output tokens count excludes reasoning tokens
            elif (
                input_tokens
                == model_usage.usage.total_tokens - output_tokens - reasoning_tokens
            ):
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
                prompt_tokens = (
                    input_tokens + cache_read_input_tokens + cache_write_input_tokens
                )
                completion_tokens = output_tokens

            else:
                raise ValueError(
                    "Model usage token counts don't follow expected pattern."
                )

            if cache_read_input_tokens + cache_write_input_tokens > prompt_tokens:
                raise ValueError("Cache categories exceed total prompt tokens")
            if reasoning_tokens > completion_tokens:
                raise ValueError("Reasoning tokens exceed total completion tokens")
            text_tokens = (
                prompt_tokens - cache_read_input_tokens - cache_write_input_tokens
            )

            if model_usage.model in CUSTOM_PRICING.keys():
                pricing_without_cache = CUSTOM_PRICING[model_usage.model]
                for count, field in (
                    (cache_read_input_tokens, "cache_read_input_token_cost"),
                    (cache_write_input_tokens, "cache_creation_input_token_cost"),
                ):
                    if count:
                        raise ValueError(f"Missing required price: {field}")
                for count, field in (
                    (prompt_tokens, "input_cost_per_token"),
                    (completion_tokens, "output_cost_per_token"),
                ):
                    if count:
                        _validate_price(pricing_without_cache.get(field), field)
                prompt_cost, completion_cost = cost_per_token(
                    model=model_usage.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    custom_cost_per_token=pricing_without_cache,
                )

            elif model_usage.model in CUSTOM_PRICING_WITH_CACHE.keys():

                pricing = CUSTOM_PRICING_WITH_CACHE[model_usage.model]
                prompt_cost = 0.0
                for count, field in (
                    (text_tokens, "input_cost_per_token"),
                    (cache_read_input_tokens, "cache_read_input_token_cost"),
                    (cache_write_input_tokens, "cache_write_input_token_cost"),
                ):
                    if count:
                        prompt_cost += count * _validate_price(
                            getattr(pricing, field), field
                        )
                completion_cost = (
                    completion_tokens
                    * _validate_price(
                        pricing.output_cost_per_token, "output_cost_per_token"
                    )
                    if completion_tokens
                    else 0.0
                )

            else:
                price_key = get_model_info(adapt_model_name(model_usage.model))["key"]
                # get_model_info() replaces missing base prices with zero.
                # Check the source entry so absence cannot become free usage.
                price_info = litellm.model_cost[price_key]
                for count, price_field in (
                    (text_tokens, "input_cost_per_token"),
                    (completion_tokens, "output_cost_per_token"),
                    (cache_read_input_tokens, "cache_read_input_token_cost"),
                    (cache_write_input_tokens, "cache_creation_input_token_cost"),
                ):
                    if count:
                        _validate_price(price_info.get(price_field), price_field)
                        # Reject malformed context overrides before LiteLLM can
                        # silently fall back to a base rate or mask a bad component.
                        for field, rate in price_info.items():
                            match = re.fullmatch(
                                re.escape(price_field) + r"_above_(\d+)(k?)_tokens",
                                field,
                            )
                            if match:
                                threshold = int(match[1]) * (1000 if match[2] else 1)
                                if prompt_tokens > threshold:
                                    _validate_price(rate, field)
                if reasoning_tokens and "output_cost_per_reasoning_token" in price_info:
                    _validate_price(
                        price_info["output_cost_per_reasoning_token"],
                        "output_cost_per_reasoning_token",
                    )

                prompt_tokens_wrapper = PromptTokensDetailsWrapper(
                    cached_tokens=cache_read_input_tokens, text_tokens=text_tokens
                )

                litellm_usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    reasoning_tokens=model_usage.usage.reasoning_tokens,
                    prompt_tokens_details=prompt_tokens_wrapper,
                    cache_read_input_tokens=cache_read_input_tokens,
                    cache_creation_input_tokens=cache_write_input_tokens,
                )

                prompt_cost, completion_cost = cost_per_token(
                    model=adapt_model_name(model_usage.model),
                    usage_object=litellm_usage,
                )

            if prompt_cost < 0 or completion_cost < 0:
                raise ValueError(
                    f"Model {model_usage.model}: negative cost "
                    f"(prompt {prompt_cost}, completion {completion_cost})."
                )
            if not isfinite(prompt_cost) or not isfinite(completion_cost):
                raise ValueError(
                    f"Non-finite cost (prompt {prompt_cost}, completion {completion_cost})"
                )
            total_cost += prompt_cost + completion_cost
            if not isfinite(total_cost):
                raise ValueError("Non-finite aggregate cost")
        except CostAccountingError:
            raise
        except Exception as e:
            raise CostAccountingError(
                f"Problem calculating cost for model {model_usage.model}, "
                f"usage {model_usage.usage.model_dump(exclude_none=True)}: {e}"
            ) from e
    return total_cost
