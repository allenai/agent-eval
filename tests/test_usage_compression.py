import pytest
from inspect_ai.model import ModelUsage

from agenteval.leaderboard.upload import compress_usages_by_problem
from agenteval.log import ModelUsageWithName


def calculate_model_usages_cost(
    model_usages: list[list[ModelUsageWithName]],
    model_costs: dict[str, dict[str, float]],
) -> float:
    total_cost = 0.0

    for problem_usages in model_usages:
        for usage_with_name in problem_usages:
            model_name = usage_with_name.model
            usage = usage_with_name.usage

            if model_name in model_costs:
                costs = model_costs[model_name]

                usage_cost = 0.0
                usage_cost += usage.input_tokens * costs.get("input_tokens", 0.0)
                usage_cost += usage.output_tokens * costs.get("output_tokens", 0.0)

                if usage.input_tokens_cache_write is not None:
                    usage_cost += usage.input_tokens_cache_write * costs.get(
                        "input_tokens_cache_write", 0.0
                    )

                if usage.input_tokens_cache_read is not None:
                    usage_cost += usage.input_tokens_cache_read * costs.get(
                        "input_tokens_cache_read", 0.0
                    )

                if usage.reasoning_tokens is not None:
                    usage_cost += usage.reasoning_tokens * costs.get(
                        "reasoning_tokens", 0.0
                    )

                total_cost += usage_cost

    return total_cost


def test_compression_preserves_total_cost(sample_model_usages, sample_model_costs):
    """Test that compression preserves the total cost calculation."""
    # Calculate cost before compression
    original_cost = calculate_model_usages_cost(sample_model_usages, sample_model_costs)

    # Compress each problem's usages
    compressed_model_usages = [
        compress_usages_by_problem(problem_usages)
        for problem_usages in sample_model_usages
    ]

    # Calculate cost after compression
    compressed_cost = calculate_model_usages_cost(
        compressed_model_usages, sample_model_costs
    )

    # Costs should be identical
    assert (
        abs(original_cost - compressed_cost) < 1e-10
    ), f"Original cost: {original_cost}, Compressed cost: {compressed_cost}"
