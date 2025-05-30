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

    extras: list[str] | None = None
    """Extra dependencies required to run the task. Should """

    primary_metric: str
    """Primary metric for the task, used for summary scores."""

    tags: list[str] | None = None
    """List of tags, used for computing summary scores for task groups."""


class TaskSet(BaseModel):
    name: str
    """Name of the split."""

    extras: list[str] | None = None

    tasks: list[Task]
    """List of tasks associated with the split."""


class Solver(BaseModel):
    path: str
    args: list[str] | None = None
    extras: list[str] | None = None

class SuiteConfig(BaseModel):
    name: str
    """Name of the suite."""

    model: str | None
    """Name of model exposed by Inspect AI."""

    tasks: list[Task]
    """Tasks to run"""

    solver: Solver
    """Solver to run the tasks."""

    output: str
    """Log directory for output"""

    @staticmethod
    def load(file_path: str) -> "SuiteConfig":
        """
        Load the suite configuration from the specified YAML file.

        Args:
            file_path: Path to the YAML file containing the suite/tasks configuration

        Returns:
            A validated SuiteConfig object
        """
        return load_suite_config(file_path)


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
