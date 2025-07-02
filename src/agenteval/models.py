from datetime import datetime
from pathlib import Path
from typing import Union

from pydantic import BaseModel, Field

from .config import SuiteConfig
from .io import atomic_write_file
from .score import EvalSpec, TaskResult


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
    eval_specs: list[EvalSpec] | None = Field(default=None, exclude=True)
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
        indent: int = 2,
        **model_dump_kwargs,
    ) -> bytes:
        """
        Return the JSON representation of this EvalResult as bytes,
        always excluding `eval_specs` and null/default values.
        """
        return self.model_dump_json(
            indent=indent,
            exclude_none=False,
            exclude_defaults=False,
            **model_dump_kwargs,
        ).encode("utf-8")
