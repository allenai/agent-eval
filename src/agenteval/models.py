import importlib
from datetime import datetime
from functools import cached_property

from pydantic import BaseModel

from .config import SuiteConfig


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


# These inspect-bound types historically lived in this module. Keep lazy
# compatibility aliases so existing scoring consumers continue to work while
# importing EvalConfig remains inspect-free for solve environments.
_LAZY_ATTRS = {
    "TaskResult": "agenteval.score",
    "TaskResults": "agenteval.score",
}


def __getattr__(name: str):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTRS))
