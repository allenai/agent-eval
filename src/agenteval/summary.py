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
    weights: Sequence[float] | None = None,
    replace_none: float | int | None = None,
) -> float | None:
    """Compute mean, optionally replacing None values with a specified value."""
    if not xs:
        return None

    if replace_none is not None:
        # Replace None values with the specified value (e.g., 0.0 for scores)
        vals = [x if x is not None else replace_none for x in xs]
        return _mean(vals, weights)
    else:
        # Preserve None values - only aggregate non-None values
        vals = [x for x in xs if x is not None]
        return _mean(vals, weights) if vals and len(vals) == len(xs) else None


def _safe_stderr(xs: Sequence[float | None]) -> float | None:
    """Compute the standard error of the mean of a list of numbers, returning None if any Nones."""
    vals = [x for x in xs if x is not None]
    if vals and len(vals) == len(xs) and len(vals) > 1:
        return stdev(vals) / math.sqrt(len(vals))
    else:
        return None


def _aggregate_with_stderr(
    values: Sequence[float | None],
    stderrs: Sequence[float | None],
    weights: Sequence[float] | None = None,
    none_is_zero: bool = False,
) -> tuple[float | None, float | None]:
    """
    Compute weighted mean and propagate standard errors using analytical formula.

    Args:
        values: Values to aggregate
        stderrs: Standard errors for each value
        weights: Weights for each value (default: equal weights)
        none_is_zero: If True, treat None values as 0 (for missing tasks).
                      This assumes the task is consistently missing/failed with value=0.

    Returns:
        Tuple of (aggregated_mean, propagated_stderr)

    Behavior:
    - If none_is_zero=True and value is None: both value and stderr become 0
      (stderr must have been None too, else ValueError)
        - The logic is that None represents an entirely missing task; if a task
          is excluded by design, it will be 0 no matter how many times we run
          the experiment, thus stderr=0.
    - If none_is_zero=False and value is None: return (None, None)
    - If stderr is None but value is not None: aggregate values but stderr becomes None (with warning)
    """
    if not values or not stderrs:
        return None, None

    if len(values) != len(stderrs):
        raise ValueError(
            f"Length mismatch: values ({len(values)}) and stderrs "
            f"({len(stderrs)}) must have the same length"
        )

    # Early validation of weights length
    if weights is not None and len(weights) != len(values):
        raise ValueError(
            f"Length mismatch: weights ({len(weights)}) must match "
            f"values ({len(values)}) length"
        )

    processed_vals = []
    processed_stderrs = []
    stderr_is_none = False

    for i, (v, se) in enumerate(zip(values, stderrs)):
        if v is None:
            if none_is_zero:
                if se is not None:
                    raise ValueError(
                        f"Value at index {i} is None (to be treated as 0), but "
                        f"stderr is {se}. When treating None as 0, stderr must also be None."
                    )
                processed_vals.append(0.0)
                processed_stderrs.append(0.0)
            else:
                # Cannot aggregate with None values when not treating as zero
                return None, None
        else:
            processed_vals.append(v)
            if se is None:
                # Value exists but stderr is missing - we can still compute mean but not stderr
                stderr_is_none = True
            elif se < 0:
                raise ValueError(
                    f"Negative standard error found: {se} at index {i}. "
                    "Standard errors must be non-negative."
                )
            else:
                processed_stderrs.append(se)

    # Use provided weights or default to equal weights
    processed_weights = weights if weights is not None else [1.0] * len(processed_vals)

    if any(w < 0 for w in processed_weights):
        raise ValueError(
            f"Negative weights found: {[w for w in processed_weights if w < 0]}. "
            "Weights must be non-negative."
        )

    total_weight = sum(processed_weights)
    if total_weight == 0:
        raise ValueError("Total weight is zero, cannot compute weighted mean")

    # Compute weighted mean
    weighted_sum = sum(v * w for v, w in zip(processed_vals, processed_weights))
    mean_val = weighted_sum / total_weight

    # Compute stderr if possible
    if stderr_is_none:
        logger.warning(
            "Some standard errors are None while values are not. "
            "Computing mean but returning None for aggregated stderr."
        )
        return mean_val, None

    # Propagate stderr using analytical formula
    # SE(weighted_mean) = sqrt(sum(w_i^2 * SE_i^2)) / sum(w_i)
    weighted_variance_sum = sum(
        (w * se) ** 2 for w, se in zip(processed_weights, processed_stderrs)
    )
    stderr_val = math.sqrt(weighted_variance_sum) / total_weight

    return mean_val, stderr_val


def compute_summary_statistics(
    suite_config: SuiteConfig,
    split: str,
    results: list[TaskResult],
    preserve_none_scores: bool = False,
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
        tag_score_stderrs = []
        tag_costs = []
        tag_cost_stderrs = []
        weights = []

        for task in tagged_tasks:
            task_weight = split_obj.get_macro_average_weight(tag_name, task.name)
            task_summary = tasks_summary[task.name]

            tag_scores.append(task_summary.score)
            tag_score_stderrs.append(task_summary.score_stderr)
            tag_costs.append(task_summary.cost)
            tag_cost_stderrs.append(task_summary.cost_stderr)
            weights.append(task_weight)

        tag_score, tag_score_stderr = _aggregate_with_stderr(
            tag_scores,
            tag_score_stderrs,
            weights=weights,
            none_is_zero=not preserve_none_scores,
        )

        tag_cost, tag_cost_stderr = _aggregate_with_stderr(
            tag_costs, tag_cost_stderrs, weights=weights
        )

        tags_summary[tag_name] = SummaryStat(
            score=tag_score,
            score_stderr=tag_score_stderr,
            cost=tag_cost,
            cost_stderr=tag_cost_stderr,
        )

    # overall summary statistics are a macro-average over tag scores
    all_scores = [s.score for s in tags_summary.values()]
    all_score_stderrs = [s.score_stderr for s in tags_summary.values()]
    all_costs = [s.cost for s in tags_summary.values()]
    all_cost_stderrs = [s.cost_stderr for s in tags_summary.values()]

    overall_score, overall_score_stderr = _aggregate_with_stderr(
        all_scores,
        all_score_stderrs,
        weights=None,
        none_is_zero=not preserve_none_scores,
    )

    overall_cost, overall_cost_stderr = _aggregate_with_stderr(
        all_costs, all_cost_stderrs, weights=None
    )

    overall = SummaryStat(
        score=overall_score,
        score_stderr=overall_score_stderr,
        cost=overall_cost,
        cost_stderr=overall_cost_stderr,
    )

    # flattened stats
    stats: dict[str, SummaryStat] = {"overall": overall}
    for tag, stat in tags_summary.items():
        stats[f"tag/{tag}"] = stat
    for task_name, stat in tasks_summary.items():
        stats[f"task/{task_name}"] = stat
    return SummaryStats(stats=stats)
