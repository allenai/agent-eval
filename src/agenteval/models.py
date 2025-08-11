from datetime import datetime
from functools import cached_property

from pydantic import BaseModel

from .config import SuiteConfig
from .score import TaskResult


class EvalConfig(BaseModel):
    suite_config: SuiteConfig
    """Task configuration for the results."""

    split: str
    """Split used for the results."""

    inspect_command: list[str] | None = None
    """InspectAI command line invoked to run the evaluation."""

    @cached_property
    def task_names(self) -> set[str]:
        """
        Get the names of all tasks in the suite for the specified split.

        Returns:
            List of task names.
        """
        return set(task.name for task in self.suite_config.get_tasks(self.split))


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

    results: list[TaskResult]

    @cached_property
    def agent_specs(self) -> set[str]:
        specs: set[str] = set()
        for task_result in self.results or []:
            if task_result.eval_spec:
                agent_spec = task_result.eval_spec.model_dump_json(
                    include={"solver", "solver_args", "model", "model_args"}
                )
                specs.add(agent_spec)
        return specs

    @cached_property
    def code_specs(self) -> set[str]:
        specs: set[str] = set()
        for task_result in self.results or []:
            if task_result.eval_spec:
                code_spec = task_result.eval_spec.model_dump_json(
                    include={"revision", "packages"}
                )
                specs.add(code_spec)
        return specs

    @cached_property
    def tasks_with_args(self) -> list[str]:
        tasks_with_args: list[str] = []
        for task_result in self.results or []:
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
        return (
            set(result.task_name for result in self.results) if self.results else set()
        )
