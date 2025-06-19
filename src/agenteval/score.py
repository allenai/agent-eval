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
from pydantic import BaseModel, Field

from .log import ModelUsageWithName, collect_model_usage, compute_model_cost

logger = logging.getLogger(__name__)


class Metric(BaseModel):
    """A metric for a task."""

    name: str
    value: float


class EvalSpec(BaseModel):
    """Combined solver and model specification for agent evaluation."""

    solver: str | None = None
    solver_args: dict[str, Any] | None = None
    model: str
    model_args: dict[str, Any] = Field(default_factory=dict)
    revision: EvalRevision | None = None

    @classmethod
    def from_eval_log(cls, log: EvalLog) -> "EvalSpec":
        return cls(
            solver=log.eval.solver,
            solver_args=log.eval.solver_args,
            model=log.eval.model,
            model_args=log.eval.model_args,
            revision=log.eval.revision,
        )


class TaskResult(BaseModel):
    """Results for a single task."""

    task_name: str
    """Name of the task."""

    metrics: list[Metric]
    """List of metrics."""

    model_usages: list[list[ModelUsageWithName]] | None = None
    """List of model usage lists per sample."""

    model_costs: list[float | None] | None = None
    """List of model costs per sample."""


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
        model_usages.append(collect_model_usage(sample))
    return model_usages


def get_normalized_task_name(log: EvalLog) -> str:
    """
    Normalize task name from eval log.

    Removes namespace from tasks that were run eg as inspect_evals/task_name

    """
    return log.eval.task.split("/")[-1]


def process_eval_logs(log_dir: str) -> tuple[list[TaskResult], list[EvalSpec], bool]:
    """
    Process evaluation logs from a directory and return task results and eval specs.

    Args:
        log_dir: Directory containing evaluation logs

    Returns:
        A tuple containing a list of task results and a list of eval specs
    """
    # Read evaluation logs
    logs = {}
    had_errors = False
    for loginfo in list_eval_logs(log_dir):
        log = read_eval_log(loginfo.name, header_only=True)
        task_name = get_normalized_task_name(log)
        if task_name in logs:
            raise ValueError(f"Task {task_name} already read.")
        logs[task_name] = log

    if not logs:
        raise ValueError("No valid evaluation logs found.")

    # Collect eval specs
    eval_specs = []
    seen_specs = set()
    for log in logs.values():
        next_eval_spec = EvalSpec.from_eval_log(log)
        # Use the hash of the serialized spec to check for duplicates
        spec_hash = hash(next_eval_spec.model_dump_json())
        if spec_hash not in seen_specs:
            seen_specs.add(spec_hash)
            eval_specs.append(next_eval_spec)

    if not eval_specs:
        raise ValueError("Eval specification is required.")

    results = []
    for task_name, log in logs.items():
        try:
            metrics = get_metrics(log)
            if len(metrics) == 0:
                raise ValueError(f"No metrics found for task {task_name}.")
            model_usages = get_model_usages(log)
            model_costs = [compute_model_cost(usages) for usages in model_usages]
            has_model_usages = any(len(usages) > 0 for usages in model_usages)
            results.append(
                TaskResult(
                    task_name=task_name,
                    metrics=metrics,
                    # Set to None to avoid incorrect pyarrow model usage type inference
                    model_usages=model_usages if has_model_usages else None,
                    model_costs=model_costs if has_model_usages else None,
                )
            )
        except ValueError as error:
            had_errors = True
            logger.exception(f"No metrics for {task_name}:")

    return results, eval_specs, had_errors
