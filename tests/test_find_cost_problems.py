import pytest
from inspect_ai.model import ModelUsage

from agenteval.log import ModelUsageWithName
from agenteval.score import TaskResult, find_cost_problems


def _result(costs, sample_count=1):
    return TaskResult(
        task_name="t",
        metrics=[],
        sample_count=sample_count,
        model_costs=costs,
    )


def test_valid_costs_pass():
    assert find_cost_problems([_result([0.1, 0.0], sample_count=2)]) == []


def test_negative_cost_rejected_even_when_missing_allowed():
    problems = find_cost_problems(
        [_result([0.1, -0.2], sample_count=2)], allow_missing=True
    )
    assert problems == ["t: negative cost at samples [2]"]


def test_missing_costs_rejected_by_default():
    assert find_cost_problems([_result([0.1, None], sample_count=2)]) == [
        "t: missing cost at samples [2]"
    ]
    assert find_cost_problems([_result(None)]) == ["t: no model costs"]


def test_missing_costs_allowed_explicitly():
    assert (
        find_cost_problems([_result(None), _result([None])], allow_missing=True) == []
    )


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_cost_rejected_even_when_missing_allowed(cost):
    assert any(
        "non-finite cost" in problem
        for problem in find_cost_problems([_result([cost])], allow_missing=True)
    )


def test_empty_costs_and_results_rejected():
    assert find_cost_problems([_result([])]) == [
        "t: 0 costs for 1 samples",
        "t: no model costs",
    ]
    assert find_cost_problems([]) == ["No task results"]
    assert find_cost_problems(None, allow_missing=True) == ["No task results"]


def test_missing_cost_override_cannot_hide_reported_usage():
    result = _result([None])
    result.model_usages = [
        [
            ModelUsageWithName(
                model="gpt-5.6-sol",
                usage=ModelUsage(
                    input_tokens=1000, output_tokens=500, total_tokens=1500
                ),
            )
        ]
    ]
    assert find_cost_problems([result], allow_missing=True) == [
        "t: missing cost at samples [1]"
    ]


def test_truncated_costs_rejected():
    result = _result([0.1])
    result.sample_count = None
    result.model_usages = [[], []]
    assert find_cost_problems([result]) == ["t: 1 costs for 2 samples"]


def test_empty_costs_cannot_bypass_sample_coverage_check():
    result = _result([])
    result.sample_count = None
    result.model_usages = [[], []]
    assert find_cost_problems([result], allow_missing=True) == [
        "t: 0 costs for 2 samples"
    ]


@pytest.mark.parametrize("costs", [[], [0.1], [0.1, 0.2, 0.3]])
def test_cost_coverage_checked_without_usage_records(costs):
    result = _result(costs)
    result.sample_count = 2
    assert result.model_usages is None
    assert f"t: {len(costs)} costs for 2 samples" in find_cost_problems(
        [result], allow_missing=True
    )


def test_legacy_costs_without_sample_count_or_usages_require_rescoring():
    result = _result([0.1])
    result.sample_count = None
    assert any(
        "cannot verify cost coverage" in problem
        for problem in find_cost_problems([result], allow_missing=True)
    )


def test_sample_count_must_agree_with_usage_records():
    result = _result([0.1, 0.2], sample_count=2)
    result.model_usages = [[]]
    assert find_cost_problems([result]) == ["t: 1 sample usages for 2 samples"]


def test_recorded_empty_task_cannot_publish_even_with_missing_cost_allowance():
    result = _result(None)
    result.sample_count = 0
    assert find_cost_problems([result], allow_missing=True) == ["t: no samples"]


def test_legacy_intentionally_unavailable_costs_remain_allowed():
    result = _result(None)
    result.sample_count = None
    assert find_cost_problems([result], allow_missing=True) == []


def test_sample_count_survives_json_round_trip_without_usage():
    result = _result([0.1, 0.2], sample_count=2)
    saved = TaskResult.model_validate_json(result.model_dump_json())
    assert saved.sample_count == 2
    assert saved.model_usages is None
    assert find_cost_problems([saved]) == []


@pytest.mark.parametrize("allow_missing", [False, True])
def test_positive_saved_cost_cannot_hide_ambiguous_cache_usage(allow_missing):
    result = _result([0.1])
    result.model_usages = [
        [
            ModelUsageWithName(
                model="gpt-5.6-sol",
                usage=ModelUsage(
                    input_tokens=1000,
                    output_tokens=500,
                    total_tokens=1500,
                    input_tokens_cache_read=200,
                ),
            )
        ]
    ]
    saved = TaskResult.model_validate_json(result.model_dump_json())
    problems = find_cost_problems([saved], allow_missing=allow_missing)
    assert any(
        "sample 1" in problem and "Ambiguous cache accounting" in problem
        for problem in problems
    )
