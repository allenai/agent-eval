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


def compute_summary_statistics(
    suite_config: SuiteConfig,
    split: str,
    results: list[TaskResult],
) -> dict[str, SummaryStat]:
    """
    Compute summary statistics for a set of task results.
    """
    tasks = suite_config.get_tasks(split)

    def safe_mean(xs: Sequence[float | None]) -> float | None:
        vals = [x for x in xs if x is not None]
        return mean(vals) if vals and len(vals) == len(xs) else None

    def safe_stderr(xs: Sequence[float | None]) -> float | None:
        vals = [x for x in xs if x is not None]
        return stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None

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
            m = next(m for m in res.metrics if m.name == task.primary_metric)
            score = m.value
            if task.primary_metric_stderr:
                stderr = next(
                    (
                        metric.value
                        for metric in res.metrics
                        if metric.name == task.primary_metric_stderr
                    ),
                    None,
                )
            task_costs = res.model_costs or []
            cost = safe_mean(task_costs)
            cost_stderr = safe_stderr(task_costs)
        tasks_summary[task.name] = SummaryStat(
            score=score,
            score_stderr=stderr,
            cost=cost,
            cost_stderr=cost_stderr,
        )

    # per-category summary
    all_tags = {t for task in tasks for t in (task.tags or [])}
    categories_summary: dict[str, SummaryStat] = {}
    for tag in all_tags:
        category_scores = [
            tasks_summary[t.name].score for t in tasks if tag in (t.tags or [])
        ]
        category_costs = [
            tasks_summary[t.name].cost for t in tasks if tag in (t.tags or [])
        ]
        categories_summary[tag] = SummaryStat(
            score=safe_mean(category_scores),
            score_stderr=None,
            cost=safe_mean(category_costs),
            cost_stderr=None,
        )

    # overall summary
    all_scores = [s.score for s in tasks_summary.values()]
    all_costs = [s.cost for s in tasks_summary.values()]
    overall = SummaryStat(
        score=safe_mean(all_scores),
        score_stderr=None,
        cost=safe_mean(all_costs),
        cost_stderr=None,
    )

    # flattened stats
    stats: dict[str, SummaryStat] = {"overall": overall}
    for tag, stat in categories_summary.items():
        stats[f"category/{tag}"] = stat
    for task_name, stat in tasks_summary.items():
        stats[f"task/{task_name}"] = stat
    return stats
