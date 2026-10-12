from pathlib import Path
from unittest.mock import Mock

import huggingface_hub
import pytest
from click.testing import CliRunner
from inspect_ai.model import ModelUsage

import agenteval.cli as cli_module
from agenteval.config import Split, SuiteConfig, Task
from agenteval.leaderboard.models import Readme
from agenteval.leaderboard.schema_generator import load_dataset_features
from agenteval.log import ModelUsageWithName
from agenteval.models import EvalConfig
from agenteval.score import Metric, TaskResult, TaskResults


@pytest.mark.parametrize(
    "allowance",
    [
        "hf://datasets/example/submissions/typo",
        "hf://datasets/example/submissions/good/",
    ],
)
def test_unmatched_cost_allowance_fails_before_network_access(monkeypatch, allowance):
    def no_api():
        pytest.fail("No network access is needed to reject an unmatched URL")

    monkeypatch.setattr(huggingface_hub, "HfApi", no_api)
    result = CliRunner().invoke(
        cli_module.cli,
        [
            "lb",
            "publish",
            "hf://datasets/example/submissions/good",
            "--allow-missing-costs",
            allowance,
        ],
    )
    assert result.exit_code != 0
    assert "URLs must match a submission URL exactly" in result.output


@pytest.mark.parametrize(
    "costs,allow_missing,has_usage,expected_success",
    [
        ([-0.1], False, False, False),
        ([-0.1], True, False, False),
        (None, False, False, False),
        ([], False, False, False),
        ([None], False, False, False),
        ([None], True, True, False),
        ([0.1], False, False, True),
        (None, True, False, True),
    ],
)
def test_publish_validates_whole_batch_before_upload(
    monkeypatch, costs, allow_missing, has_usage, expected_success
):
    config = EvalConfig(
        suite_config=SuiteConfig(
            name="test",
            version="1.0.0",
            splits=[
                Split(
                    name="test",
                    tasks=[
                        Task(
                            name="task_test", path="task.py", primary_metric="accuracy"
                        )
                    ],
                )
            ],
        ),
        split="test",
        inspect_command=["inspect", "eval", "task.py"],
    )
    good_path = "1.0.0/test/good"
    bad_path = "1.0.0/test/second"
    urls = [
        f"hf://datasets/example/submissions/{path}" for path in [good_path, bad_path]
    ]
    api = Mock()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    features = load_dataset_features()
    monkeypatch.setattr(
        Readme,
        "download_and_parse",
        lambda _: Readme(
            text_content="", configs={"1.0.0": ["test"]}, features=features
        ),
    )

    def download(*, local_dir, **_):
        for path in [good_path, bad_path]:
            directory = Path(local_dir) / path
            directory.mkdir(parents=True)
            (directory / cli_module.EVAL_CONFIG_FILENAME).write_text(
                config.model_dump_json()
            )
            (directory / cli_module.SUBMISSION_METADATA_FILENAME).write_text("{}")
            result = TaskResult(
                task_name="task_test",
                metrics=[Metric(name="accuracy", value=1.0)],
                model_costs=[0.1] if path == good_path else costs,
            )
            if path == bad_path and has_usage:
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
            summary_dir = Path(local_dir) / cli_module.SUMMARIES_PREFIX / path
            summary_dir.mkdir(parents=True)
            (summary_dir / cli_module.SCORES_FILENAME).write_text(
                TaskResults(results=[result]).model_dump_json()
            )

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    args = ["lb", "publish", *urls, "--repo-id", "example/results"]
    if allow_missing:
        args.extend(["--allow-missing-costs", urls[1]])

    result = CliRunner().invoke(cli_module.cli, args)

    if expected_success:
        assert result.exit_code == 0, result.output
        api.upload_folder.assert_called_once()
    else:
        assert result.exit_code != 0
        assert "Refusing to publish 1.0.0/test/second" in result.output
        api.upload_folder.assert_not_called()
