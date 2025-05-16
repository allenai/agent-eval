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


def _safe_mean(xs: Sequence[float | None]) -> float | None:
    """Compute the mean of a list of numbers, returning None if any Nones."""
    vals = [x for x in xs if x is not None]
    if vals and len(vals) == len(xs):
        return mean(vals)
    else:
        return None


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

    # per-tag summary
    all_tags = {t for task in tasks for t in (task.tags or [])}
    tags_summary: dict[str, SummaryStat] = {}
    for tag in all_tags:
        tag_scores = [
            tasks_summary[t.name].score for t in tasks if tag in (t.tags or [])
        ]
        tag_costs = [tasks_summary[t.name].cost for t in tasks if tag in (t.tags or [])]
        tags_summary[tag] = SummaryStat(
            score=_safe_mean(tag_scores),
            score_stderr=None,
            cost=_safe_mean(tag_costs),
            cost_stderr=None,
        )

    # overall summary
    all_scores = [s.score for s in tasks_summary.values()]
    all_costs = [s.cost for s in tasks_summary.values()]
    overall = SummaryStat(
        score=_safe_mean(all_scores),
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
