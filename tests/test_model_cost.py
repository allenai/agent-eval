import pytest
from inspect_ai.model import ModelUsage

from agenteval.log import ModelUsageWithName, compute_model_cost


def test_deepseek_v4_pro_preview_cost():
    usages = [
        ModelUsageWithName(
            model="deepseek/deepseek-v4pro-preview",
            usage=ModelUsage(
                input_tokens=79_800_000,
                output_tokens=1_300_000,
                total_tokens=81_100_000,
                input_tokens_cache_write=0,
                input_tokens_cache_read=0,
            ),
        )
    ]

    assert compute_model_cost(usages) == pytest.approx(35.844)


def test_deepseek_v4_pro_preview_cost_with_cache():
    usages = [
        ModelUsageWithName(
            model="deepseek/deepseek-v4pro-preview",
            usage=ModelUsage(
                input_tokens=1_000_000,
                output_tokens=4_000_000,
                total_tokens=10_000_000,
                input_tokens_cache_write=3_000_000,
                input_tokens_cache_read=2_000_000,
            ),
        )
    ]

    assert compute_model_cost(usages) == pytest.approx(3.92225)
