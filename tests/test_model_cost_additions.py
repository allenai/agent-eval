import copy
import hashlib
import importlib.resources
import json
from types import SimpleNamespace

import click
import litellm
import pytest
from inspect_ai.model import ModelUsage

from agenteval import cli as cli_module
from agenteval.log import ModelUsageWithName, compute_model_cost


@pytest.fixture
def additions():
    return json.loads(
        importlib.resources.files("agenteval")
        .joinpath("model_cost_additions.json")
        .read_text(encoding="utf-8")
    )


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
    url = cli_module.prep_litellm_cost_map()
    assert "ef84494d52c6708e4e9f4a54ce551a265995ad8f" in url
    assert registered == {**base, **additions["models"]}
    expected_hash = hashlib.sha256(
        json.dumps(registered, sort_keys=True).encode()
    ).hexdigest()
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
    additions, cache, reasoning, total, expected, model
):
    original = copy.deepcopy(litellm.model_cost)
    try:
        litellm.register_model(additions["models"])
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
    finally:
        litellm.model_cost.clear()
        litellm.model_cost.update(original)


def test_gemini_inconsistent_usage_returns_blank(additions):
    litellm.register_model(additions["models"])
    usage = ModelUsage(input_tokens=1000, output_tokens=200, total_tokens=999)
    assert (
        compute_model_cost([ModelUsageWithName(model="gemini-3.7-flash", usage=usage)])
        is None
    )
