import pytest
import yaml
from pydantic import ValidationError

from agenteval.config import SuiteConfig, Task


def test_task_simple_tags():
    """Test task with simple string tags."""
    task_data = {
        "name": "test_task",
        "path": "tasks/test",
        "primary_metric": "accuracy/mean",
        "tags": ["tag1", "tag2"],
    }
    task = Task.model_validate(task_data)

    assert task.tags is not None
    assert len(task.tags) == 2
    assert task.tags[0] == "tag1"
    assert task.tags[1] == "tag2"

    # Test helper method
    assert task.get_tag_names() == ["tag1", "tag2"]


def test_task_no_tags():
    """Test task with no tags."""
    task_data = {
        "name": "test_task",
        "path": "tasks/test",
        "primary_metric": "accuracy/mean",
    }
    task = Task.model_validate(task_data)

    assert task.tags is None
    assert task.get_tag_names() == []


def test_split_weight_adjustments():
    """Test split with macro_average_weight_adjustments."""
    split_data = {
        "name": "test_split",
        "tasks": [
            {
                "name": "task1",
                "path": "tasks/task1",
                "primary_metric": "accuracy",
                "tags": ["category1"],
            },
            {
                "name": "task2",
                "path": "tasks/task2",
                "primary_metric": "accuracy",
                "tags": ["category1"],
            },
        ],
        "macro_average_weight_adjustments": [
            {"tag": "category1", "task": "task1", "weight": 0.5},
            {"tag": "category1", "task": "task2", "weight": 2.0},
        ],
    }

    from agenteval.config import Split

    split = Split.model_validate(split_data)

    # Test weight lookup
    assert split.get_macro_average_weight("category1", "task1") == 0.5
    assert split.get_macro_average_weight("category1", "task2") == 2.0

    # Test default weight for tasks not in adjustments
    assert split.get_macro_average_weight("category1", "task3") == 1.0

    # Test default weight for tags not in adjustments
    assert split.get_macro_average_weight("other_tag", "task1") == 1.0


def test_split_no_weight_adjustments():
    """Test split without weight adjustments - should default to 1.0."""
    split_data = {
        "name": "test_split",
        "tasks": [
            {
                "name": "task1",
                "path": "tasks/task1",
                "primary_metric": "accuracy",
                "tags": ["category1"],
            }
        ],
    }

    from agenteval.config import Split

    split = Split.model_validate(split_data)

    # Should default to 1.0
    assert split.get_macro_average_weight("category1", "task1") == 1.0
    assert split.get_macro_average_weight("any_tag", "any_task") == 1.0


def test_suite_config_loading():
    """Test loading a complete suite config."""
    config_data = {
        "name": "test_suite",
        "version": "1.0.0",
        "splits": [
            {
                "name": "test",
                "tasks": [
                    {
                        "name": "task1",
                        "path": "tasks/task1",
                        "primary_metric": "accuracy",
                        "tags": ["category1"],
                    }
                ],
                "macro_average_weight_adjustments": [
                    {"tag": "category1", "task": "task1", "weight": 0.5}
                ],
            }
        ],
    }

    config = SuiteConfig.model_validate(config_data)
    assert config.name == "test_suite"
    assert len(config.splits) == 1

    # Test getting tasks and splits
    tasks = config.get_tasks("test")
    assert len(tasks) == 1
    assert tasks[0].name == "task1"

    split = config.get_split("test")
    assert split.get_macro_average_weight("category1", "task1") == 0.5


def test_yaml_loading():
    """Test loading from YAML format."""
    yaml_content = """
name: test_suite
version: "1.0.0"
splits:
  - name: test
    tasks:
      - name: task1
        path: tasks/task1
        primary_metric: accuracy
        tags:
          - category1
    macro_average_weight_adjustments:
      - tag: category1
        task: task1
        weight: 0.5
"""

    config_data = yaml.safe_load(yaml_content)
    config = SuiteConfig.model_validate(config_data)

    assert config.name == "test_suite"
    split = config.get_split("test")
    assert split.get_macro_average_weight("category1", "task1") == 0.5


def test_invalid_split_name():
    """Test error when requesting invalid split."""
    config_data = {"name": "test_suite", "splits": [{"name": "valid", "tasks": []}]}

    config = SuiteConfig.model_validate(config_data)

    with pytest.raises(ValueError, match="Split 'invalid' not found"):
        config.get_tasks("invalid")

    with pytest.raises(ValueError, match="Split 'invalid' not found"):
        config.get_split("invalid")
