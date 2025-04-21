import json
from datetime import datetime
from pathlib import Path
from typing import Union

from pydantic import BaseModel, Field, field_serializer, field_validator

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


JSON_SERIALIZED_FIELDS = ["suite_config"]


class EvalResult(EvalConfig):
    eval_specs: list[EvalSpec] | None = None
    results: list[TaskResult] | None = None
    submission: SubmissionMetadata = Field(default_factory=SubmissionMetadata)

    @field_serializer(*JSON_SERIALIZED_FIELDS)
    def _serialize_fields(self, v, _info):
        if v is None:
            return None
        if isinstance(v, list):
            data = [
                item.model_dump() if hasattr(item, "model_dump") else item for item in v
            ]
        else:
            data = v.model_dump() if hasattr(v, "model_dump") else v
        return json.dumps(data)

    @field_validator(*JSON_SERIALIZED_FIELDS, mode="before")
    @classmethod
    def validate_json_fields(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

    def find_missing_tasks(self) -> list[str]:
        try:
            tasks = self.suite_config.get_tasks(self.split)
            result_task_names = (
                {result.task_name for result in self.results} if self.results else set()
            )
            return [task.name for task in tasks if task.name not in result_task_names]
        except ValueError:
            return []

    def save_json(
        self,
        path: Union[str, Path],
        indent: int = 2,
        exclude_none: bool = True,
        exclude_defaults: bool = True,
    ) -> None:
        """
        Atomically write this EvalResult to JSON at the given path.

        Uses model_dump_json with indent/exclude flags and
        atomic file replace.
        """
        content = self.model_dump_json(
            indent=indent,
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults,
            exclude={"eval_specs"},
        )
        atomic_write_file(path, content, encoding="utf-8")

    def dump_json_bytes(
        self,
        indent: int = 2,
        exclude_none: bool = True,
        exclude_defaults: bool = True,
    ) -> bytes:
        """
        Return the JSON representation of this EvalResult as bytes,
        using default indent and exclusion settings.
        """
        return self.model_dump_json(
            indent=indent,
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults,
            exclude={"eval_specs"},
        ).encode("utf-8")
