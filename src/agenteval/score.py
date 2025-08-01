"""Scoring utilities for the NoraBench suite."""

import logging
from typing import Any

from inspect_ai.log import (
    EvalLog,
    EvalRevision,
    list_eval_logs,
    read_eval_log,
    read_eval_log_samples,
)
from pydantic import BaseModel, Field, field_serializer, field_validator

from .log import ModelUsageWithName, collect_model_usage, compute_model_cost

logger = logging.getLogger(__name__)

# Fields with dict type that need JSON serialization for Arrow/Parquet and HuggingFace datasets compatibility
# These systems cannot handle dict types so we serialize to JSON strings
_EVALSPEC_JSON_FIELDS = ["solver_args", "model_args", "task_args", "packages"]


class Metric(BaseModel):
    """A metric for a task."""

    name: str
    value: float


class EvalSpec(BaseModel):
    """Combined solver and model specification for agent evaluation."""

    solver: str | None = None
    solver_args: dict[str, Any] | None = None
    model: str
    model_args: dict[str, Any] | None = None
    task_args: dict[str, Any] | None = None
    task_args_passed: dict[str, Any] | None = Field(default=None, exclude=True)
    revision: EvalRevision | None = None
    packages: dict[str, str] | None = None

    @classmethod
    def from_eval_log(cls, log: EvalLog) -> "EvalSpec":
        return cls(
            solver=log.eval.solver,
            solver_args=log.eval.solver_args,
            model=log.eval.model,
            model_args=log.eval.model_args,
            task_args=log.eval.task_args,
            task_args_passed=log.eval.task_args_passed,
            revision=log.eval.revision,
            packages=log.eval.packages,
        )

    @field_validator(*_EVALSPEC_JSON_FIELDS, mode="before")
    @classmethod
    def deserialize_json_fields(cls, v):
        """Deserialize JSON strings back to Python objects. Raises on JSON errors."""
        import json

        if not isinstance(v, str):
            return v  # Already deserialized or None
        return json.loads(v)

    @field_serializer(*_EVALSPEC_JSON_FIELDS)
    def serialize_json_fields(self, v):
        """Serialize Python objects to JSON strings. Logs errors and returns fallback."""
        import json

        if v is None:
            return None
        try:
            return json.dumps(v, default=str, sort_keys=True)
        except (TypeError, ValueError) as e:
            logger.warning(
                f"Failed to serialize field to JSON: {e}, returning error indicator"
            )
            return json.dumps({"__serialization_error__": str(e)})


class TaskResult(BaseModel):
    """Results for a single task."""

    task_name: str
    """Name of the task. Derived from Inspect `EvalLog.eval.task`."""

    eval_spec: EvalSpec | None = None
    """Evaluation specification used for this task. Derived from Inspect `EvalLog.eval`."""

    metrics: list[Metric]
    """List of metrics. Derived from Inspect `EvalLog.results.scores`."""

    model_usages: list[list[ModelUsageWithName]] | None = None
    """List of model usage lists per sample. Derived from Inspect `EvalLog.samples`."""

    model_costs: list[float | None] | None = None
    """List of model costs per sample. Computed from `model_usages`."""


def get_metrics(log: EvalLog) -> list[Metric]:
    """Extract metrics from an evaluation log."""
    metrics_list = []
    seen_metric_names = set()
    if not log.results or not log.results.scores:
        raise ValueError("No scores available in the evaluation log.")
    for score in log.results.scores:
        for metric in score.metrics.values():
            metric_name = f"{score.name}/{metric.name}"
            # Check for duplicates using a set for efficiency
            if metric_name in seen_metric_names:
                raise ValueError(
                    f"Duplicate metric key {metric_name} in task {log.eval.task}"
                )
            seen_metric_names.add(metric_name)
            metrics_list.append(Metric(name=metric_name, value=metric.value))
    return metrics_list


def get_model_usages(log: EvalLog) -> list[list[ModelUsageWithName]]:
    """Extract model usages of all samples in an evaluation log."""
    model_usages = []
    # Don't assume eval log has more than the header
    for sample in read_eval_log_samples(log.location, all_samples_required=True):
        model_usages.append(collect_model_usage(sample.events))
    return model_usages


def get_normalized_task_name(log: EvalLog) -> str:
    """
    Normalize task name from eval log.

    Removes namespace from tasks that were run eg as inspect_evals/task_name

    """
    return log.eval.task.split("/")[-1]


class EvalLogProcessingResult(BaseModel):
    results: list[TaskResult]
    errors: list[str]


def process_eval_logs(log_dir: str) -> EvalLogProcessingResult:
    """
    Process evaluation logs from a directory and return task results.

    Args:
        log_dir: Directory containing evaluation logs

    Returns:
        A tuple containing a list of task results and whether there were errors
    """
    # Read evaluation logs
    logs = {}
    errors = []
    for loginfo in list_eval_logs(log_dir):
        log = read_eval_log(loginfo.name, header_only=True)
        task_name = get_normalized_task_name(log)
        if task_name in logs:
            raise ValueError(f"Task {task_name} already read.")
        logs[task_name] = log

    if not logs:
        raise ValueError("No valid evaluation logs found.")

    results = []
    for task_name, log in logs.items():
        eval_spec = EvalSpec.from_eval_log(log)
        metrics = get_metrics(log)
        if len(metrics) == 0:
            errors.append(f"No metrics found for task {task_name}.")
            continue
        model_usages = get_model_usages(log)
        model_costs: list[float | None] = [
            compute_model_cost(usages) for usages in model_usages
        ]
        has_model_usages = any(len(usages) > 0 for usages in model_usages)
        results.append(
            TaskResult(
                task_name=task_name,
                eval_spec=eval_spec,
                metrics=metrics,
                # Set to None to avoid incorrect pyarrow model usage type inference
                model_usages=model_usages if has_model_usages else None,
                model_costs=model_costs if has_model_usages else None,
            )
        )

    return EvalLogProcessingResult(results=results, errors=errors)
