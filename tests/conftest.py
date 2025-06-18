import pytest
from inspect_ai.model import ModelUsage

from agenteval.log import ModelUsageWithName


@pytest.fixture
def sample_model_costs():
    """Sample cost rates for different models and token types."""
    return {
        "o3-2025-04-16": {
            "input_tokens": 0.00003,
            "output_tokens": 0.00006,
            "input_tokens_cache_write": 0.000075,
            "input_tokens_cache_read": 0.000015,
            "reasoning_tokens": 0.00012,
        },
        "claude-sonnet-4-20250514": {
            "input_tokens": 0.000015,
            "output_tokens": 0.000075,
            "input_tokens_cache_write": 0.0000375,
            "input_tokens_cache_read": 0.0000075,
            "reasoning_tokens": 0.00015,
        },
    }


@pytest.fixture
def sample_model_usages() -> list[list[ModelUsageWithName]]:
    """Create a list of model usages grouped by problem."""
    problem1_usages = [
        ModelUsageWithName(
            model="claude-sonnet-4-20250514",
            usage=ModelUsage(
                input_tokens=7,
                output_tokens=104,
                total_tokens=18994,
                input_tokens_cache_write=4388,
                input_tokens_cache_read=14495,
                reasoning_tokens=None,
            ),
        ),
        ModelUsageWithName(
            model="claude-sonnet-4-20250514",
            usage=ModelUsage(
                input_tokens=7,
                output_tokens=95,
                total_tokens=24357,
                input_tokens_cache_write=5372,
                input_tokens_cache_read=18883,
                reasoning_tokens=None,
            ),
        ),
        ModelUsageWithName(
            model="o3-2025-04-16",
            usage=ModelUsage(
                input_tokens=613,
                output_tokens=307,
                total_tokens=920,
                input_tokens_cache_write=None,
                input_tokens_cache_read=0,
                reasoning_tokens=256,
            ),
        ),
        ModelUsageWithName(
            model="o3-2025-04-16",
            usage=ModelUsage(
                input_tokens=970,
                output_tokens=408,
                total_tokens=1378,
                input_tokens_cache_write=None,
                input_tokens_cache_read=890,
                reasoning_tokens=0,
            ),
        ),
    ]

    problem2_usages = [
        ModelUsageWithName(
            model="o3-2025-04-16",
            usage=ModelUsage(
                input_tokens=542,
                output_tokens=427,
                total_tokens=969,
                input_tokens_cache_write=None,
                input_tokens_cache_read=0,
                reasoning_tokens=384,
            ),
        ),
        ModelUsageWithName(
            model="o3-2025-04-16",
            usage=ModelUsage(
                input_tokens=1033,
                output_tokens=36,
                total_tokens=1069,
                input_tokens_cache_write=None,
                input_tokens_cache_read=890,
                reasoning_tokens=0,
            ),
        ),
        ModelUsageWithName(
            model="claude-sonnet-4-20250514",
            usage=ModelUsage(
                input_tokens=7,
                output_tokens=90,
                total_tokens=15362,
                input_tokens_cache_write=5484,
                input_tokens_cache_read=9781,
                reasoning_tokens=None,
            ),
        ),
    ]

    return [problem1_usages, problem2_usages]
