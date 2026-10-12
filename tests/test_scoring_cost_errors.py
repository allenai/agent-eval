from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from inspect_ai.model import ModelUsage

import agenteval.cli as cli_module
import agenteval.score as score_module
from agenteval.config import Split, SuiteConfig, Task
from agenteval.log import CostAccountingError, ModelUsageWithName
from agenteval.models import EvalConfig
from agenteval.score import EvalSpec, Metric


@pytest.mark.parametrize(
    "model,total_tokens,cache_read",
    [
        ("unknown-price-model", 1500, 0),
        ("gpt-5.6-sol", 1600, 0),
        ("gpt-5.6-sol", 1500, 5000),
    ],
)
def test_scoring_failure_identifies_sample_and_preserves_artifacts(
    monkeypatch, tmp_path, model, total_tokens, cache_read
):
    task = Task(name="task_test", path="task.py", primary_metric="accuracy")
    config = EvalConfig(
        suite_config=SuiteConfig(
            name="test", version="1.0.0", splits=[Split(name="test", tasks=[task])]
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
    monkeypatch.setattr(
        score_module, "list_eval_logs", lambda _: [SimpleNamespace(name="log")]
    )
    monkeypatch.setattr(score_module, "read_eval_log", lambda *_, **__: log)
    monkeypatch.setattr(
        score_module, "get_metrics", lambda _: [Metric(name="accuracy", value=1.0)]
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
        ),
    )
    monkeypatch.setattr(
        score_module, "get_model_usages", lambda _: [[valid_usage], [invalid_usage]]
    )
    monkeypatch.setattr(cli_module, "prep_litellm_cost_map", lambda: "fixed-price-map")

    result = CliRunner().invoke(cli_module.cli, ["score", str(tmp_path)])

    assert result.exit_code != 0
    assert isinstance(result.exception, CostAccountingError)
    assert "Task task_test, sample 2:" in str(result.exception)
    assert model in str(result.exception)
    assert all(artifact.read_text() == "previous result" for artifact in artifacts)
    for artifact in artifacts:
        artifact.unlink()
    result = CliRunner().invoke(cli_module.cli, ["score", str(tmp_path)])
    assert result.exit_code != 0
    assert not any(artifact.exists() for artifact in artifacts)
