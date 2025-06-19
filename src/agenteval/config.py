"""
Configuration management for agent evaluation.
"""

import yaml
from pydantic import BaseModel, ValidationError


class WeightAdjustment(BaseModel):
    """Weight adjustment for a specific tag-task combination."""

    tag: str
    task: str
    weight: float


class Task(BaseModel):
    name: str
    """Canonical task name (used by the leaderboard)."""

    path: str
    """Path to the task definition (used by Inspect)."""

    primary_metric: str
    """Primary metric for the task, used for summary scores."""

    tags: list[str] | None = None
    """List of tags, used for computing summary scores for task groups."""

    def get_tag_names(self) -> list[str]:
        """Get list of tag names."""
        return self.tags or []


class Split(BaseModel):
    name: str
    """Name of the split."""

    tasks: list[Task]
    """List of tasks associated with the split."""

    macro_average_weight_adjustments: list[WeightAdjustment] | None = None
    """Weight adjustments for macro averaging."""

    def get_macro_average_weight(self, tag_name: str, task_name: str) -> float:
        """Get weight for a specific tag-task combination in macro averaging."""
        if self.macro_average_weight_adjustments:
            for adjustment in self.macro_average_weight_adjustments:
                if adjustment.tag == tag_name and adjustment.task == task_name:
                    return adjustment.weight
        return 1.0  # Default weight


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
        return self.get_split(split_name).tasks

    def get_split(self, split_name: str) -> Split:
        """Get a specific split by name."""
        for split in self.splits:
            if split.name == split_name:
                return split
        available_splits = [split.name for split in self.splits]
        raise ValueError(
            f"Split '{split_name}' not found. Available splits: {available_splits}"
        )


def load_suite_config(file_path: str) -> SuiteConfig:
    """
    Load configuration from a YAML file.

    Args:
        file_path: Path to the YAML file containing the suite/tasks configuration

    Returns:
        SuiteConfig object

    Raises:
        FileNotFoundError: If the file is not found
        ValidationError: If the configuration is invalid
    """
    try:
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Task configuration file not found: {file_path}")

    try:
        return SuiteConfig.model_validate(config_data)
    except ValidationError as e:
        raise ValidationError(
            f"Invalid task configuration: {e}\nPlease refer to the config spec."
        )
