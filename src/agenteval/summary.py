import math
from collections.abc import Sequence
from statistics import mean, stdev

from pydantic import BaseModel

from .config import SuiteConfig
from .score import TaskResult


class SummaryStat(BaseModel):
    score: float | None
    score_stderr: float | None
    cost: float | None
    cost_stderr: float | None


def _mean(
    vals: Sequence[float], weights: Sequence[float] | None = None
) -> float | None:
    """Compute mean, optionally weighted."""
    if weights is None:
        return mean(vals)

    if len(vals) != len(weights):
        raise ValueError(
            f"Length mismatch: values ({len(vals)}) and weights "
            f"({len(weights)}) must have the same length"
        )

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight is zero, cannot compute weighted mean")

    weighted_sum = sum(v * w for v, w in zip(vals, weights))
    return weighted_sum / total_weight


def _safe_mean(
    xs: Sequence[float | None],
    is_score: bool = False,
    weights: Sequence[float] | None = None,
) -> float | None:
    """Compute mean, treating None as 0 for scores, otherwise returning None if any Nones."""
    if not xs:
        return None
    if is_score:
        vals = [x if x is not None else 0.0 for x in xs]
        return _mean(vals, weights)
    vals = [x for x in xs if x is not None]
    return _mean(vals, weights) if vals and len(vals) == len(xs) else None


def _safe_stderr(xs: Sequence[float | None]) -> float | None:
    """Compute the standard error of the mean of a list of numbers, returning None if any Nones."""
    vals = [x for x in xs if x is not None]
    if vals and len(vals) == len(xs) and len(vals) > 1:
        return stdev(vals) / math.sqrt(len(vals))
    else:
        return None


def compute_summary_statistics(
    suite_config: SuiteConfig,
    split: str,
    results: list[TaskResult],
) -> dict[str, SummaryStat]:
    """
    Compute summary statistics for a set of task results.
    """
    tasks = suite_config.get_tasks(split)

    # build per-task stats
    tasks_summary: dict[str, SummaryStat] = {}
    for task in tasks:
        res = next((r for r in results if r.task_name == task.name), None)
        # initialize variables with explicit types
        score: float | None = None
        stderr: float | None = None
        cost: float | None = None
        cost_stderr: float | None = None
        if res:
            try:
                m = next(m for m in res.metrics if m.name == task.primary_metric)
            except StopIteration:
                raise ValueError(
                    f"Task {task.name} does not have a metric named {task.primary_metric}."
                    f" Available metrics: {', '.join(m.name for m in res.metrics)}"
                )
            score = m.value

            stderr = next(
                (
                    m.value
                    for m in res.metrics
                    if m.name.startswith(f"{task.primary_metric.split('/')[0]}/")
                    and "stderr" in m.name
                ),
                None,
            )

            task_costs = res.model_costs or []
            cost = _safe_mean(task_costs)
            cost_stderr = _safe_stderr(task_costs)
        tasks_summary[task.name] = SummaryStat(
            score=score,
            score_stderr=stderr,
            cost=cost,
            cost_stderr=cost_stderr,
        )

    # per-tag summary with weighted averaging
    split_obj = suite_config.get_split(split)
    tag_to_tasks: dict[str, list] = {}
    for task in tasks:
        for tag in task.tags or []:
            tag_to_tasks.setdefault(tag, []).append(task)

    tags_summary: dict[str, SummaryStat] = {}
    for tag_name, tagged_tasks in tag_to_tasks.items():
        tag_scores = []
        tag_costs = []
        weights = []

        for task in tagged_tasks:
            task_weight = split_obj.get_macro_average_weight(tag_name, task.name)
            task_summary = tasks_summary[task.name]

            tag_scores.append(task_summary.score)
            tag_costs.append(task_summary.cost)
            weights.append(task_weight)

        tags_summary[tag_name] = SummaryStat(
            score=_safe_mean(tag_scores, is_score=True, weights=weights),
            score_stderr=None,
            cost=_safe_mean(tag_costs, weights=weights),
            cost_stderr=None,
        )

    # overall summary
    all_scores = [s.score for s in tasks_summary.values()]
    all_costs = [s.cost for s in tasks_summary.values()]
    overall = SummaryStat(
        score=_safe_mean(all_scores, is_score=True),
        score_stderr=None,
        cost=_safe_mean(all_costs),
        cost_stderr=None,
    )

    # flattened stats
    stats: dict[str, SummaryStat] = {"overall": overall}
    for tag, stat in tags_summary.items():
        stats[f"tag/{tag}"] = stat
    for task_name, stat in tasks_summary.items():
        stats[f"task/{task_name}"] = stat
    return stats
