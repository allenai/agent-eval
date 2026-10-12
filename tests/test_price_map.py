import json
from copy import deepcopy
from importlib import resources

import httpx
import litellm
import pytest
from click.testing import CliRunner

import agenteval.cli as cli_module


@pytest.mark.parametrize("setting", [None, "False", "true"])
def test_score_requires_frozen_prices_before_reading_logs(
    monkeypatch, tmp_path, setting
):
    if setting is None:
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    else:
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", setting)

    def no_fetch(*args, **kwargs):
        pytest.fail("Scoring must reject the setting before fetching prices")

    monkeypatch.setattr(cli_module.httpx, "get", no_fetch)
    result = CliRunner().invoke(cli_module.cli, ["score", str(tmp_path)])
    assert result.exit_code != 0
    assert "LITELLM_LOCAL_MODEL_COST_MAP" in result.output
    assert not (tmp_path / cli_module.SCORES_FILENAME).exists()


def test_cli_restores_the_same_release_prices_used_by_cost_tests(monkeypatch):
    bundled_map = json.loads(
        resources.files("litellm")
        .joinpath("model_prices_and_context_window_backup.json")
        .read_text()
    )
    isolated_map = deepcopy(cli_module.litellm_model_cost)
    monkeypatch.setattr(cli_module, "litellm_model_cost", isolated_map)
    monkeypatch.setattr(litellm, "model_cost", isolated_map)
    models = ["gpt-5.6-sol", "claude-sonnet-4-5", "gemini/gemini-2.5-flash"]
    for model in models:
        assert cli_module.litellm_model_cost[model] == bundled_map[model]
        monkeypatch.setitem(
            cli_module.litellm_model_cost,
            model,
            {**bundled_map[model], "input_cost_per_token": 999.0},
        )

    def fetch(url, timeout):
        # The upstream v1.97.0 tag resolves to this commit.
        assert url == (
            "https://raw.githubusercontent.com/BerriAI/litellm/"
            "ef84494d52c6708e4e9f4a54ce551a265995ad8f/"
            "litellm/model_prices_and_context_window_backup.json"
        )
        return httpx.Response(200, json=bundled_map, request=httpx.Request("GET", url))

    monkeypatch.setattr(cli_module.httpx, "get", fetch)
    cli_module.prep_litellm_cost_map()
    for model in models:
        for field, value in bundled_map[model].items():
            assert cli_module.litellm_model_cost[model][field] == value
