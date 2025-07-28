from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, Union

import datasets
from pydantic import BaseModel, Field

from .config import SuiteConfig
from .io import atomic_write_file
from .score import TaskResult


class EvalConfig(BaseModel):
    suite_config: SuiteConfig
    """Task configuration for the results."""

    split: str
    """Split used for the results."""


class SubmissionMetadata(BaseModel):
    """Metadata for Hugging Face submission."""

    submit_time: datetime | None = None
    username: str | None = None
    agent_name: str | None = None
    agent_description: str | None = None
    agent_url: str | None = None
    logs_url: str | None = None
    logs_url_public: str | None = None
    summary_url: str | None = None
    openness: str | None = None
    tool_usage: str | None = None


class EvalResult(EvalConfig):
    results: list[TaskResult] | None = None
    submission: SubmissionMetadata = Field(default_factory=SubmissionMetadata)

    def find_missing_tasks(self) -> list[str]:
        try:
            tasks = self.suite_config.get_tasks(self.split)
            result_task_names = (
                {result.task_name for result in self.results} if self.results else set()
            )
            return [task.name for task in tasks if task.name not in result_task_names]
        except ValueError:
            return []

    def is_scored(self) -> bool:
        """
        Check if the evaluation result is scored.

        Returns:
            bool: True if the evaluation result is scored, False otherwise.
        """
        return self.results is not None and len(self.results) > 0

    def save_json(
        self,
        path: Union[str, Path],
        indent: int = 2,
        **model_dump_kwargs,
    ) -> None:
        """
        Atomically write this EvalResult to JSON at the given path.

        The motivation for using an atomic write is to avoid data loss of the
        original config file, if something goes wrong during the write.
        """
        content = self.dump_json_bytes(
            indent=indent,
            **model_dump_kwargs,
        ).decode("utf-8")
        atomic_write_file(path, content, encoding="utf-8")

    def dump_json_bytes(
        self,
        indent: int | None = 2,
        **model_dump_kwargs,
    ) -> bytes:
        """
        Return the JSON representation of this EvalResult as bytes.
        """
        return self.model_dump_json(
            indent=indent,
            exclude_none=False,
            exclude_defaults=False,
            **model_dump_kwargs,
        ).encode("utf-8")

    def check_result_primary_metrics_against_provided_suite_config(self, provided_suite_config: SuiteConfig) -> Dict[str, Set[str]]:
        # prep for suite config info
        tasks_from_suite_config = provided_suite_config.get_tasks(self.split)
        primary_metric_from_suite_config_by_task_name: Dict[str, str] = {}
        for task in tasks_from_suite_config:
            task_name = task.name
            assert task_name not in primary_metric_from_suite_config_by_task_name
            primary_metric_from_suite_config_by_task_name[task_name] = task.primary_metric

        # prep for result info
        task_metric_names_from_results_by_task_name: Dict[str, Set[str]] = {}
        for result in self.results:
            task_name = result.task_name
            if task_name not in task_metric_names_from_results_by_task_name:
                task_metric_names_from_results_by_task_name[task_name] = set([])
            for metric in result.metrics:
                task_metric_names_from_results_by_task_name[task_name].add(metric.name)

        # check metrics
        available_metrics_for_tasks_missing_primary_metric_by_task_name: Dict[str, Tuple[str, Set[str]]] = {}
        for task_name, primary_metric in primary_metric_from_suite_config_by_task_name.items():
            result_metric_names = task_metric_names_from_results_by_task_name.get(task_name)
            if result_metric_names is not None:
                if primary_metric not in result_metric_names:
                    available_metrics_for_tasks_missing_primary_metric_by_task_name[task_name] = (primary_metric, result_metric_names)

        return available_metrics_for_tasks_missing_primary_metric_by_task_name

    def check_result_primary_metrics_against_provided_suite_config(self) -> Dict[str, Tuple[str, Set[str]]]:
        return check_results_against_provided_suite_config(self.suite_config)

    # TODO: should we use this in view.py too? Probably?
    @staticmethod
    def fetch_first_result_from_result_repo(repo_id: str, huggingface_config: str, split: str) -> Optional["EvalResult"]:
        ds = datasets.load_dataset(repo_id, name=huggingface_config).get(split)
        if ds:
            return EvalResult.model_validate(ds[0])
        else:
            return None
