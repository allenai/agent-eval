from agenteval.score import TaskResult, find_cost_problems


def _result(costs):
    return TaskResult(task_name="t", metrics=[], model_costs=costs)


def test_valid_costs_pass():
    assert find_cost_problems([_result([0.1, 0.0])]) == []


def test_negative_cost_rejected_even_when_missing_allowed():
    problems = find_cost_problems([_result([0.1, -0.2])], allow_missing=True)
    assert problems == ["t: negative cost at samples [1]"]


def test_missing_costs_rejected_by_default():
    assert find_cost_problems([_result([0.1, None])]) == [
        "t: missing cost at samples [1]"
    ]
    assert find_cost_problems([_result(None)]) == ["t: no model costs"]


def test_missing_costs_allowed_explicitly():
    assert (
        find_cost_problems([_result(None), _result([None])], allow_missing=True) == []
    )
