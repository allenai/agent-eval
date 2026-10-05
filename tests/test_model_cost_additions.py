import copy
import hashlib
import importlib.resources
import json
from types import SimpleNamespace
from unittest.mock import Mock

import click
import litellm
import pytest
from click.testing import CliRunner
from inspect_ai.model import ModelUsage

from agenteval import cli as cli_module
from agenteval.log import ModelUsageWithName, compute_model_cost
from agenteval.score import EvalLogProcessingResult, TaskResults


@pytest.fixture
def additions():
    return json.loads(
        importlib.resources.files("agenteval")
        .joinpath("model_cost_additions.json")
        .read_text(encoding="utf-8")
    )


@pytest.fixture
def registered_additions(additions):
    original = copy.deepcopy(litellm.model_cost)
    try:
        litellm.register_model(additions["models"])
        yield
    finally:
        litellm.model_cost.clear()
        litellm.model_cost.update(original)


@pytest.mark.parametrize("version", ["1.67.4.post1", "1.98.0"])
def test_score_rejects_unsupported_litellm_before_processing(
    monkeypatch, tmp_path, version
):
    from agenteval import score as score_module

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(cli_module.importlib.metadata, "version", lambda name: version)
    fetch = Mock()
    register = Mock()
    process = Mock()
    monkeypatch.setattr(cli_module.httpx, "get", fetch)
    monkeypatch.setattr(cli_module, "register_model", register)
    monkeypatch.setattr(score_module, "process_eval_logs", process)

    result = CliRunner().invoke(cli_module.score_command, [str(tmp_path)])

    assert result.exit_code == 1
    assert "Scoring requires LiteLLM 1.97.0" in result.output
    assert f"found {version}" in result.output
    assert "pip install 'agent-eval[scoring]'" in result.output
    fetch.assert_not_called()
    register.assert_not_called()
    process.assert_not_called()
    assert not (tmp_path / cli_module.SCORES_FILENAME).exists()


def test_frozen_table_preserved_and_combined_hash(monkeypatch, capsys, additions):
    base = {"existing-model": {"input_cost_per_token": 0.123}}
    registered = {}
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        cli_module.httpx,
        "get",
        lambda url, timeout: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: copy.deepcopy(base)
        ),
    )
    monkeypatch.setattr(
        cli_module, "register_model", lambda model_cost: registered.update(model_cost)
    )
    metadata = cli_module.prep_litellm_cost_map()
    assert metadata["cost_map_url"] == cli_module.FROZEN_MODEL_COST_MAP_URL
    assert registered == {**base, **additions["models"]}
    expected_hash = hashlib.sha256(
        json.dumps(
            registered, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    assert metadata["frozen_cost_map_sha256"] == expected_hash
    assert metadata["cost_map_additions_source"] == additions["source"]
    output = capsys.readouterr().out
    assert f"Frozen model costs hash {expected_hash}." in output
    assert additions["source"] in output


def test_addition_cannot_override_base(monkeypatch, additions):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        cli_module.httpx,
        "get",
        lambda url, timeout: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: additions["models"]
        ),
    )
    with pytest.raises(click.ClickException, match="Remove cost additions"):
        cli_module.prep_litellm_cost_map()


@pytest.mark.parametrize(
    "cache,reasoning,total,expected",
    [
        (0, 0, 1200, 0.0015),
        (0, 300, 1500, 0.002625),
        (400, 0, 1200, 0.00123),
        (400, 300, 1500, 0.002355),
    ],
)
@pytest.mark.parametrize("model", ["gemini-3.7-flash", "gemini/gemini-3.7-flash"])
def test_gemini_standard_token_costs(
    registered_additions, cache, reasoning, total, expected, model
):
    usage = ModelUsage(
        input_tokens=1000,
        output_tokens=200,
        total_tokens=total,
        input_tokens_cache_read=cache,
        reasoning_tokens=reasoning,
    )
    assert compute_model_cost(
        [ModelUsageWithName(model=model, usage=usage)]
    ) == pytest.approx(expected)


def test_gemini_inconsistent_usage_returns_blank(registered_additions):
    usage = ModelUsage(input_tokens=1000, output_tokens=200, total_tokens=999)
    assert (
        compute_model_cost([ModelUsageWithName(model="gemini-3.7-flash", usage=usage)])
        is None
    )


@pytest.mark.parametrize("target", ["local", "hf"])
def test_score_persists_cost_map_metadata(monkeypatch, tmp_path, additions, target):
    from agenteval import score as score_module
    from agenteval import summary as summary_module

    metadata = {
        "cost_map_url": cli_module.FROZEN_MODEL_COST_MAP_URL,
        "cost_map_additions_source": additions["source"],
        "frozen_cost_map_sha256": "a" * 64,
    }
    monkeypatch.setattr(cli_module, "prep_litellm_cost_map", lambda: metadata)
    config = {
        "suite_config": {
            "name": "test",
            "version": "1.0.0",
            "splits": [{"name": "test", "tasks": []}],
        },
        "split": "test",
    }
    (tmp_path / cli_module.EVAL_CONFIG_FILENAME).write_text(json.dumps(config))
    monkeypatch.setattr(
        score_module,
        "process_eval_logs",
        lambda *args, **kwargs: EvalLogProcessingResult(results=[], errors=[]),
    )
    monkeypatch.setattr(
        summary_module,
        "compute_summary_statistics",
        lambda *args: SimpleNamespace(
            model_dump=lambda **kwargs: {}, model_dump_json=lambda **kwargs: "{}"
        ),
    )
    import huggingface_hub

    api = Mock()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path.parent)
    )
    log_dir = str(tmp_path) if target == "local" else f"hf://test/repo/{tmp_path.name}"
    result = CliRunner().invoke(cli_module.score_command, [log_dir])
    assert result.exit_code == 0, result.output
    if target == "local":
        scored = (tmp_path / cli_module.SCORES_FILENAME).read_text()
        api.upload_file.assert_not_called()
    else:
        scored = api.upload_file.call_args_list[0].kwargs["path_or_fileobj"].getvalue()
    assert {key: json.loads(scored)[key] for key in metadata} == metadata
    assert TaskResults.model_validate_json(scored).frozen_cost_map_sha256 == "a" * 64


def test_legacy_scores_without_additions_metadata():
    scores = TaskResults.model_validate_json('{"results":[],"cost_map_url":"old"}')
    assert scores.cost_map_url == "old"
    assert scores.cost_map_additions_source is None
    assert scores.frozen_cost_map_sha256 is None


@pytest.mark.parametrize(
    "invalid",
    [
        "{",
        "[]",
        "{}",
        '{"source":"https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/model_prices_and_context_window_backup.json","models":{}}',
        '{"models": []}',
        '{"models": {"model": {"input_cost_per_token": 1, "output_cost_per_token": 1}}}',
        '{"models": {"model": {"litellm_provider": "gemini"}}}',
        '{"models": {"model": {"litellm_provider": "gemini", "input_cost_per_token": -1, "output_cost_per_token": 1}}}',
        '{"models": {"model": {"litellm_provider": "gemini", "input_cost_per_token": 1, "output_cost_per_token": NaN}}}',
    ],
)
def test_invalid_additions_fail_before_registration(monkeypatch, additions, invalid):
    if invalid.startswith('{"models"'):
        invalid = json.dumps({"source": additions["source"], **json.loads(invalid)})
    resource = SimpleNamespace(
        joinpath=lambda name: resource, read_text=lambda **kwargs: invalid
    )
    monkeypatch.setattr(
        cli_module.importlib.resources, "files", lambda package: resource
    )
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        cli_module.httpx,
        "get",
        lambda url, timeout: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {}
        ),
    )
    register = Mock()
    monkeypatch.setattr(cli_module, "register_model", register)
    with pytest.raises(click.ClickException, match="Invalid model cost additions"):
        cli_module.prep_litellm_cost_map()
    register.assert_not_called()
