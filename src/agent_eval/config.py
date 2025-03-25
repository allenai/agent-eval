"""
Configuration management for agent evaluation.
"""

import os
from importlib import resources
from typing import List

import yaml
from pydantic import BaseModel


class Task(BaseModel):
    task_name: str
    """Canonical task name (used by the leaderboard)."""

    task_path: str
    """Path to the task definition (used by Inspect)."""


def get_tasks(taskset_name: str) -> List[Task]:
    """
    Load tasks from the specified taskset configuration.

    Args:
        taskset_name: Name of the taskset configuration file (without extension)
                     or a custom path to a YAML file containing task definitions

    Returns:
        List of Task objects
    """
    # Check if taskset_name is a path to a YAML file
    if os.path.isfile(taskset_name) and (
        taskset_name.endswith(".yml") or taskset_name.endswith(".yaml")
    ):
        with open(taskset_name, "r") as f:
            tasks_data = yaml.safe_load(f)
    else:
        # Load from package resources
        try:
            with (
                resources.files(__package__)
                .joinpath(f"config/{taskset_name}.yml")
                .open("r") as f
            ):
                tasks_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ValueError(f"Taskset configuration not found: {taskset_name}")

    return [Task(**task_data) for task_data in tasks_data]
