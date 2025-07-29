from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Dict, Set, Tuple

from pydantic import BaseModel, Field

from .config import Task, SuiteConfig
from .io import atomic_write_file
from .score import TaskResult


class EvalConfig(BaseModel):
    suite_config: SuiteConfig
    """Task configuration for the results."""

    split: str
    """Split used for the results."""

    @cached_property
    def task_names(self) -> set[str]:
        """
        Get the names of all tasks in the suite for the specified split.

        Returns:
            List of task names.
        """
        return set(task.name for task in self.get_tasks())

    def get_tasks(self) -> Set[Task]:
        return self.suite_config.get_tasks(self.split)


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


class TaskResults(BaseModel):
    """Scores for all tasks in the suite"""

    results: list[TaskResult] | None = None

    @cached_property
    def agent_specs(self) -> set[str]:
        specs = set()
        for task_result in self.results:
            if task_result.eval_spec:
                agent_spec = task_result.eval_spec.model_dump_json(
                    include={"solver", "solver_args", "model", "model_args"}
                )
                specs.add(agent_spec)
        return specs

    @cached_property
    def code_specs(self) -> set[str]:
        specs = set()
        for task_result in self.results:
            if task_result.eval_spec:
                code_spec = task_result.eval_spec.model_dump_json(
                    include={"revision", "packages"}
                )
                specs.add(code_spec)
        return specs

    @cached_property
    def tasks_with_args(self) -> list[str]:
        tasks_with_args = []
        for task_result in self.results:
            if task_result.eval_spec and task_result.eval_spec.task_args_passed:
                tasks_with_args.append(task_result.task_name)
        return tasks_with_args

    @cached_property
    def task_names(self) -> set[str]:
        """
        Get the names of all tasks in the results.

        Returns:
            List of task names.
        """
        return set(result.task_name for result in self.results)

    def check_primary_metrics_against_provided_eval_config(self, provided_eval_config: EvalConfig) -> Dict[str, Set[str]]:
        # prep for eval config info
        tasks_from_eval_config = provided_eval_config.get_tasks()
        primary_metric_from_eval_config_by_task_name: Dict[str, str] = {}
        for task in tasks_from_eval_config:
            task_name = task.name
            assert task_name not in primary_metric_from_eval_config_by_task_name
            primary_metric_from_eval_config_by_task_name[task_name] = task.primary_metric

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
        for task_name, primary_metric in primary_metric_from_eval_config_by_task_name.items():
            result_metric_names = task_metric_names_from_results_by_task_name.get(task_name)
            if result_metric_names is not None:
                if primary_metric not in result_metric_names:
                    available_metrics_for_tasks_missing_primary_metric_by_task_name[task_name] = (primary_metric, result_metric_names)

        return available_metrics_for_tasks_missing_primary_metric_by_task_name

    # # TODO: should we use this in view.py too? Probably?
    # @staticmethod
    # def fetch_first_result_from_result_repo(repo_id: str, huggingface_config: str, split: str) -> Optional["EvalResult"]:
    #     ds = datasets.load_dataset(repo_id, name=huggingface_config).get(split)
    #     if ds:
    #         return EvalResult.model_validate(ds[0])
    #     else:
    #         return None
