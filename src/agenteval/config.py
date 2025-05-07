"""
Configuration management for agent evaluation.
"""

import yaml
from pydantic import BaseModel, ValidationError


class Task(BaseModel):
    name: str
    """Canonical task name (used by the leaderboard)."""

    path: str
    """Path to the task definition (used by Inspect)."""

    primary_metric: str
    """Primary metric for the task, used for summary scores."""

    tags: list[str] | None = None
    """List of tags, used for computing summary scores for task groups."""


class Split(BaseModel):
    name: str
    """Name of the split."""

    tasks: list[Task]
    """List of tasks associated with the split."""


class SuiteConfig(BaseModel):
    name: str
    """Name of the suite."""

    version: str | None = None
    """Version of the suite, e.g. '1.0.0.dev1'."""

    splits: list[Split]
    """List of splits in the suite."""

    def get_tasks(self, split_name: str) -> list[Task]:
        """
        Get the tasks for a specific split.

        Args:
            split_name: Name of the split to retrieve tasks from

        Returns:
            List of Task objects for the specified split

        Raises:
            ValueError: If the split is not found
        """
        for split in self.splits:
            if split.name == split_name:
                return split.tasks

        available_splits = ", ".join(split.name for split in self.splits)
        raise ValueError(
            f"Split '{split_name}' not found. Available splits: {available_splits}"
        )


def load_suite_config(file_path: str) -> SuiteConfig:
    """
    Load the suite configuration from the specified YAML file.

    Args:
        file_path: Path to the YAML file containing the suite/tasks configuration

    Returns:
        A validated SuiteConfig object
    """
    try:
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Task configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file: {e}")

    try:
        return SuiteConfig.model_validate(config_data)
    except ValidationError as e:
        raise ValueError(
            f"Invalid task configuration: {e}\nPlease refer to the config spec."
        )
