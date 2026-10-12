from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from inspect_ai.model import ModelUsage

import agenteval.cli as cli_module
import agenteval.score as score_module
from agenteval.config import Split, SuiteConfig, Task
from agenteval.log import ModelUsageWithName
from agenteval.models import EvalConfig
from agenteval.score import EvalSpec, Metric


@pytest.mark.parametrize("prior_error", [False, True])
@pytest.mark.parametrize(
    "model,total_tokens,cache_read,reasoning_tokens",
    [
        ("unknown-price-model", 1500, 0, 0),
        ("gpt-5.6-sol", 1600, 0, 0),
        ("gpt-5.6-sol", 1500, 5000, 0),
        ("grok-3", 1600, 0, 0),
        ("grok-3", 1700, 200, 0),
        ("moonshotai/kimi-k2.5-0127", 1600, 0, 0),
        ("deepseek/deepseek-v4pro-preview", 1500, 5000, 0),
        ("gpt-5.6-sol", 1500, 200, 0),
        ("gpt-5.6-sol", 1800, 200, 300),
        ("deepseek/deepseek-v4pro-preview", 1500, 200, 0),
        ("gpt-5.6-sol", 1700, 200, 200),
        ("deepseek/deepseek-v4pro-preview", 1700, 200, 200),
    ],
)
def test_scoring_failure_identifies_sample_and_preserves_artifacts(
    monkeypatch,
    tmp_path,
    model,
    total_tokens,
    cache_read,
    reasoning_tokens,
    prior_error,
):
    task = Task(name="task_test", path="task.py", primary_metric="accuracy")
    missing_metrics_task = Task(
        name="missing_metrics", path="missing.py", primary_metric="accuracy"
    )
    config = EvalConfig(
        suite_config=SuiteConfig(
            name="test",
            version="1.0.0",
            splits=[Split(name="test", tasks=[missing_metrics_task, task])],
        ),
        split="test",
        inspect_command=["inspect", "eval", "task.py"],
    )
    (tmp_path / cli_module.EVAL_CONFIG_FILENAME).write_text(config.model_dump_json())
    artifacts = [
        tmp_path / cli_module.SCORES_FILENAME,
        tmp_path / cli_module.SUMMARY_FILENAME,
    ]
    for artifact in artifacts:
        artifact.write_text("previous result")

    log = SimpleNamespace(
        eval=SimpleNamespace(task="task_test", task_registry_name="task.py")
    )
    prior_log = SimpleNamespace(
        eval=SimpleNamespace(task="missing_metrics", task_registry_name="missing.py")
    )
    monkeypatch.setattr(
        score_module,
        "list_eval_logs",
        lambda _: [
            SimpleNamespace(name=name)
            for name in (["prior", "log"] if prior_error else ["log"])
        ],
    )
    monkeypatch.setattr(
        score_module,
        "read_eval_log",
        lambda name, **__: prior_log if name == "prior" else log,
    )
    monkeypatch.setattr(
        score_module,
        "get_metrics",
        lambda loaded: (
            [] if loaded is prior_log else [Metric(name="accuracy", value=1.0)]
        ),
    )
    monkeypatch.setattr(EvalSpec, "from_eval_log", lambda _: EvalSpec(model=model))
    valid_usage = ModelUsageWithName(
        model="gpt-5.6-sol",
        usage=ModelUsage(input_tokens=1000, output_tokens=500, total_tokens=1500),
    )
    invalid_usage = ModelUsageWithName(
        model=model,
        usage=ModelUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=total_tokens,
            input_tokens_cache_read=cache_read,
            reasoning_tokens=reasoning_tokens,
        ),
    )
    monkeypatch.setattr(
        score_module, "get_model_usages", lambda _: [[valid_usage], [invalid_usage]]
    )
    monkeypatch.setattr(cli_module, "prep_litellm_cost_map", lambda: "fixed-price-map")

    result = CliRunner().invoke(cli_module.cli, ["score", str(tmp_path)])

    assert result.exit_code != 0
    assert result.exit_code == 1
    assert "Error: Task task_test, sample 2:" in result.output
    assert model in result.output
    assert "usage {" in result.output
    assert "Traceback" not in result.output
    if prior_error:
        assert "No metrics found for task missing_metrics." in result.output
    assert all(artifact.read_text() == "previous result" for artifact in artifacts)
    for artifact in artifacts:
        artifact.unlink()
    result = CliRunner().invoke(cli_module.cli, ["score", str(tmp_path)])
    assert result.exit_code != 0
    assert not any(artifact.exists() for artifact in artifacts)


@pytest.mark.parametrize("has_usage", [True, False])
def test_scoring_records_sample_count_even_without_usage(monkeypatch, has_usage):
    log = SimpleNamespace(
        eval=SimpleNamespace(task="task_test", task_registry_name="task.py")
    )
    monkeypatch.setattr(
        score_module, "list_eval_logs", lambda _: [SimpleNamespace(name="log")]
    )
    monkeypatch.setattr(score_module, "read_eval_log", lambda *_, **__: log)
    monkeypatch.setattr(
        score_module, "get_metrics", lambda _: [Metric(name="accuracy", value=1.0)]
    )
    monkeypatch.setattr(EvalSpec, "from_eval_log", lambda _: EvalSpec(model="test"))
    usage = ModelUsageWithName(
        model="gpt-5.6-sol",
        usage=ModelUsage(input_tokens=1000, output_tokens=500, total_tokens=1500),
    )
    monkeypatch.setattr(
        score_module,
        "get_model_usages",
        lambda _: [[usage], []] if has_usage else [[], []],
    )
    task = Task(name="task_test", path="task.py", primary_metric="accuracy")

    processed = score_module.process_eval_logs("logs", [task])
    saved = score_module.TaskResults.model_validate_json(
        score_module.TaskResults(results=processed.results).model_dump_json()
    ).results[0]

    assert saved.sample_count == 2
    assert (saved.model_usages is not None) == has_usage
    if has_usage:
        assert saved.model_costs is not None
        assert len(saved.model_costs) == 2
    else:
        assert saved.model_costs is None
