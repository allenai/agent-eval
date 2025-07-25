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


class SuiteConfigResultsComparisonInfo(BaseModel):
    tasks_in_only_suite_config: Set[str]
    tasks_in_only_results: Set[str]
    available_metrics_for_tasks_missing_primary_metric_in_results: Dict[str, Set[str]]

    def tasks_expected_by_suite_config_are_missing(self) -> bool:
        return len(self.tasks_in_only_suite_config) > 0

    def tasks_missing_primary_metric(self) -> bool:
        return len(self.available_metrics_for_tasks_missing_primary_metric_in_results) > 0

    def warning_for_missing_tasks_expected_by_suite_config(self, which_suite_config: str) -> str:
        return (
            f"Warning: Tasks in the {which_suite_config}'s suite config that are missing "
            f"from results: {', '.join(self.tasks_in_only_suite_config)}"
        )

    def warnings_for_tasks_missing_primary_metric(self, which_suite_config: str) -> List[str]:
        warnings = []
        for task_name, available_metrics in self.available_metrics_for_tasks_missing_primary_metric_in_results.items():
            warnings.append(
                (
                    f"Warning: the {task_name} task does not have the primary metric "
                    f"expected based on the {which_suite_config}'s suite config. "
                    f"Available metrics in results for this task: {', '.join(available_metrics)}"
                )
            )
        return warnings


class EvalResult(EvalConfig):
    results: list[TaskResult] | None = None
    submission: SubmissionMetadata = Field(default_factory=SubmissionMetadata)

    # TODO: do we still need this?
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

    def check_results_against_provided_suite_config(self, provided_suite_config: SuiteConfig) -> SuiteConfigResultsComparisonInfo:
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

        # check tasks
        task_names_from_suite_config = set(tasks_from_suite_config_by_task_name.keys())
        task_names_from_results = set(task_metric_names_from_results_by_task_name.keys())
        tasks_in_suite_config_but_not_in_results = task_names_from_suite_config.difference(task_names_from_results)
        tasks_in_results_but_not_in_suite_config = task_names_from_results.difference(task_names_from_suite_config)

        # check metrics
        available_metrics_for_tasks_missing_primary_metric_by_task_name = {}
        for task_name, primary_metric in primary_metric_from_suite_config_by_task_name.items():
            result_metric_names = task_metric_names_from_results_by_task_name.get(task_name)
            if primary_metric not in result_metric_names:
                available_metrics_for_tasks_missing_primary_metric_by_task_name[task_name] = result_metric_names

        return SuiteConfigResultsComparisonInfo(
            tasks_in_only_suite_config=tasks_in_suite_config_but_not_in_results,
            tasks_in_only_results=tasks_in_results_but_not_in_suite_config,
            available_metrics_for_tasks_missing_primary_metric_in_results=available_metrics_for_tasks_missing_primary_metric_by_task_name,
        )

    def check_results_against_my_suite_config(self) -> SuiteConfigResultsComparisonInfo:
        return check_results_against_provided_suite_config(self.suite_config)

    # TODO: should we use this in view.py too? Probably?
    @staticmethod
    def fetch_first_result_from_result_repo(repo_id: str, huggingface_config: str, split: str) -> Optional["EvalResult"]:
        ds = datasets.load_dataset(repo_id, name=huggingface_config).get(split)
        if ds:
            return EvalResult.model_validate(ds[0])
        else:
            return None
