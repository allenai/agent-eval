"""Scoring utilities for the NoraBench suite."""

from typing import Any

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log, read_eval_log_samples
from inspect_ai.model import ModelUsage
from inspect_ai.solver import SolverSpec
from pydantic import BaseModel

from .log import compute_model_cost, compute_model_usage


class SolverConfig(BaseModel):
    solver: SolverSpec
    model: str
    model_args: dict[str, Any]

    @classmethod
    def from_eval_log(cls, log: EvalLog) -> "SolverConfig":
        return cls(
            solver=SolverSpec(log.eval.solver or "", log.eval.solver_args or {}),
            model=log.eval.model,
            model_args=log.eval.model_args,
        )


class TaskResult(BaseModel):
    eval_log: EvalLog
    """Evaluation log."""

    metrics: dict[str, float] = {}
    """Flat mapping of metric names to values."""

    model_usages: list[dict[str, ModelUsage]] = []
    """List of model usages per sample."""

    model_costs: list[float] = []
    """List of model costs per sample."""


class ResultSet(BaseModel):
    results: dict[str, TaskResult]
    """Mapping of task to TaskResult."""

    solver_config: SolverConfig
    """Solver configuration for the results."""

    submission_name: str | None = None
    """Leaderboard submission name for the results."""


def get_metrics(log: EvalLog) -> dict[str, float]:
    """
    Extract metrics from an evaluation log
    """
    metrics = dict()
    if not log.results or not log.results.scores:
        raise ValueError("No scores available in the evaluation log.")
    for score in log.results.scores:
        for metric in score.metrics.values():
            metric_name = f"{score.name}/{metric.name}"
            if metric_name in metrics:
                raise ValueError(
                    f"Duplicate metric key {metric_name} in task {log.eval.task}"
                )
            metrics[metric_name] = metric.value
    return metrics


def get_model_usages(log: EvalLog) -> list[dict[str, ModelUsage]]:
    """
    Extract model usages of all samples in an evaluation log
    """
    model_usages = []
    # Don't assume eval log has more than the header
    for sample in read_eval_log_samples(log.location, all_samples_required=True):
        model_usages.append(compute_model_usage(sample))
    return model_usages


def read_eval_log_dir(log_dir: str) -> dict[str, EvalLog]:
    """
    Reads evaluation log files in a directory
    and collects them into a dictionary keyed by task name
    """
    results = dict()
    for loginfo in list_eval_logs(log_dir):
        log = read_eval_log(loginfo.name, header_only=True)
        task = log.eval.task
        if task in results:
            raise ValueError(f"Task {task} already processed.")
        results[task] = log
    if not results:
        raise ValueError("No valid evaluation logs found.")
    return results


def get_result_set(log_dir: str) -> ResultSet:
    logs = read_eval_log_dir(log_dir)

    # Validate single solver configuration
    solver_config: SolverConfig | None = None
    for t in logs:
        next_solver_config = SolverConfig.from_eval_log(logs[t])
        if not solver_config:
            solver_config = next_solver_config
        elif solver_config != next_solver_config:
            raise ValueError(
                f"All tasks must have the same solver configuration, but found different configurations: {solver_config} and {next_solver_config}"
            )
    if not solver_config:
        raise ValueError("Solver configuration is required.")

    results = dict()
    for t in logs:
        metrics = get_metrics(logs[t])
        model_usages = get_model_usages(logs[t])
        model_costs = [compute_model_cost(usage) for usage in model_usages]
        results[t] = TaskResult(
            eval_log=logs[t],
            metrics=metrics,
            model_usages=model_usages,
            model_costs=model_costs,
        )

    return ResultSet(
        results={t: results[t] for t in results},
        solver_config=solver_config,
    )
