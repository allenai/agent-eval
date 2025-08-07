import logging
import math
from collections.abc import Sequence
from statistics import mean, stdev

from pydantic import BaseModel

from .config import SuiteConfig
from .score import TaskResult

logger = logging.getLogger(__name__)


class SummaryStat(BaseModel):
    score: float | None
    score_stderr: float | None
    cost: float | None
    cost_stderr: float | None


class SummaryStats(BaseModel):
    stats: dict[str, SummaryStat]


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
) -> SummaryStats:
    """
    Compute summary statistics for a set of task results.
    """
    tasks = suite_config.get_tasks(split)

    # build per-task stats
    tasks_summary: dict[str, SummaryStat] = {}
    for task in tasks:
        tasks_summary[task.name] = SummaryStat(
            score=None,
            score_stderr=None,
            cost=None,
            cost_stderr=None,
        )

        res = next((r for r in results if r.task_name == task.name), None)
        if not res:
            # logger.warning(f"Task {task.name} has no results.")
            continue

        metrics_by_name = {m.name: m for m in res.metrics}
        if task.primary_metric not in metrics_by_name:
            # We don't have a value for the primary metric.
            logger.warning(
                f"Task {task.name} does not have a metric named {task.primary_metric}."
                f" Available metrics: {', '.join(m.name for m in res.metrics)}"
            )
            continue

        tasks_summary[task.name].score = metrics_by_name[task.primary_metric].value

        expected_stderr_name = f"{task.primary_metric.rpartition('/')[0]}/stderr"
        stderr_metric = metrics_by_name.get(expected_stderr_name, None)
        tasks_summary[task.name].score_stderr = (
            stderr_metric.value if stderr_metric else None
        )

        if tasks_summary[task.name].score_stderr is None:
            logger.warning(
                f"Task {task.name} does not have a metric named {expected_stderr_name}."
            )

        task_costs = res.model_costs or []
        tasks_summary[task.name].cost = _safe_mean(task_costs)
        tasks_summary[task.name].cost_stderr = _safe_stderr(task_costs)

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

    # overall summary statistics are a macro-average over tag scores
    all_scores = [s.score for s in tags_summary.values()]
    all_costs = [s.cost for s in tags_summary.values()]
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
    return SummaryStats(stats=stats)
