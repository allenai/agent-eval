"""
View and plot leaderboard results.
"""

import logging
from zoneinfo import ZoneInfo

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .. import compute_summary_statistics
from ..config import SuiteConfig
from .model_name_mapping import LB_MODEL_NAME_MAPPING
from .models import LeaderboardSubmission

logger = logging.getLogger(__name__)


class LeaderboardViewer:
    """
    Load and visualize leaderboard for a given HF dataset split.
    """

    def __init__(
        self, repo_id: str, config: str, split: str, is_internal: bool = False
    ):
        self._repo_id = repo_id
        self._config = config
        self._split = split
        self._internal = is_internal

        # build suite_config and mapping from tags to tasks from the first result
        # TODO: Verify the sort order
        ds = datasets.load_dataset(repo_id, name=config).get(split)
        if not ds:
            raise ValueError(f"Split '{split}' not found in dataset results")
        suite = LeaderboardSubmission.model_validate(ds[0]).suite_config
        self._cfg = suite
        self.tag_map: dict[str, list[str]] = {}
        for task in suite.get_tasks(split):
            for t in task.tags or []:
                self.tag_map.setdefault(t, []).append(task.name)

    def _load(self):
        results = datasets.load_dataset(self._repo_id, name=self._config)
        overview = _get_dataframe(
            eval_results=results,
            split=self._split,
            is_internal=self._internal,
            suite_config=self._cfg,
        )
        return overview, self.tag_map

    def view(
        self, tag: str | None = None, with_plots: bool = False
    ) -> tuple[pd.DataFrame, dict[str, plt.Figure]]:
        """
        If tag is None, primary="Overall" and group=all tags.
        Otherwise primary=tag and group=tasks under that tag.
        """
        data, tag_map = self._load()
        cols = [
            "id",
            "Agent",
            "Agent description",
            "User/organization",
            "Submission date",
            "Logs",
            "Source",
            "Openness",
            "Agent tooling",
            "LLM base",
        ]

        # choose primary metric and its sub‐group
        if tag is None:
            primary = "Overall"
            group = list(tag_map.keys())
        else:
            primary = f"{tag} score"
            group = tag_map.get(tag, [])

        # Check if the primary column exists before sorting
        if primary not in data.columns:
            raise KeyError(
                f"Column '{primary}' not found. Available columns: {list(data.columns)}"
            )

        data = data.sort_values(primary, ascending=False)

        # build full metric list: primary + its cost + each member and its cost
        if tag is None:
            metrics = [primary, f"{primary} cost"] + [
                m for t in group for m in (f"{t} score", f"{t} cost")
            ]
        else:
            metrics = [primary, f"{tag} cost"] + [
                m for t in group for m in (f"{t} score", f"{t} cost")
            ]

        # filter to relevant columns
        ci_cols = [f"{m} 95% CI" for m in metrics if f"{m} 95% CI" in data.columns]
        df = data.loc[
            :,
            cols + [c for c in metrics if c in data.columns] + ci_cols,
        ].reset_index(drop=True)

        plots: dict[str, plt.Figure] = {}
        if with_plots:
            avail = [c for c in metrics if c in df.columns]
            plots["bar"] = _plot_hbar(df, agent_col="Agent", metrics=avail)
            plot_metrics = [primary] + [f"{t} score" for t in group]
            for m in plot_metrics:
                cost_col = f"{m.replace(' score', '')} cost"
                if cost_col in df.columns and m in df.columns:
                    plots[f"scatter_{m.replace(' score', '')}"] = _plot_scatter(
                        df, x=cost_col, y=m, agent_col="Agent"
                    )

        return df, plots


def _get_dataframe(
    eval_results: datasets.DatasetDict,
    split: str,
    is_internal: bool,
    suite_config: SuiteConfig,
    timezone: str = "US/Pacific",
) -> pd.DataFrame:
    """
    Load leaderboard results from the given dataset split and return a DataFrame.
    """
    ds = eval_results.get(split)
    if not ds:
        cols = ["agent_name", "agent_description", "username", "submit_time"]
        pretty = [_pretty_column_name(c) for c in cols]
        empty = pd.DataFrame({c: ["No data"] for c in pretty})
        return empty

    cfg = suite_config

    rows = []
    for itm in ds:
        ev = LeaderboardSubmission.model_validate(itm)

        model_token_counts: dict[str, int] = {}
        if ev.results:
            for task_result in ev.results:
                if task_result.model_usages:
                    for usage_list in task_result.model_usages:
                        for model_usage in usage_list:
                            model_name = model_usage.model
                            total_tokens = model_usage.usage.total_tokens

                            if model_name in model_token_counts:
                                model_token_counts[model_name] += total_tokens
                            else:
                                model_token_counts[model_name] = total_tokens

        # Sort by cumulative token count (descending - most used first)
        sorted_raw_names = sorted(
            model_token_counts.keys(), key=lambda x: model_token_counts[x], reverse=True
        )

        model_names = [
            LB_MODEL_NAME_MAPPING.get(name, name) for name in sorted_raw_names
        ]

        sub = ev.submission
        # only format if submit_time present, else leave as None
        ts = sub.submit_time
        if ts is not None:
            date = ts.astimezone(ZoneInfo(timezone)).strftime("%Y-%m-%d")
        else:
            date = None

        if not ev.results:
            logger.warning(
                f"Skipping submission {sub.agent_name} ({sub.username}) "
                f"({sub.submit_time}) with no results"
            )
            continue
        stats = compute_summary_statistics(
            suite_config=cfg, split=split, results=ev.results
        )
        flat = {}
        for key, s in stats.stats.items():
            parts = key.split("/")
            if parts[0] == "overall":
                flat["overall/score"], flat["overall/cost"] = s.score, s.cost
            elif parts[0] == "tag":
                flat[f"tag/{parts[1]}/score"], flat[f"tag/{parts[1]}/cost"] = (
                    s.score,
                    s.cost,
                )
            else:  # task
                t0 = parts[1]
                # compute 95% CI half-width from stderr
                flat.update(
                    {
                        f"task/{t0}/score": s.score,
                        f"task/{t0}/score_ci": (
                            (s.score_stderr * 1.96)
                            if s.score_stderr is not None
                            else np.nan
                        ),
                        f"task/{t0}/cost": s.cost,
                        f"task/{t0}/cost_ci": (
                            (s.cost_stderr * 1.96)
                            if s.cost_stderr is not None
                            else np.nan
                        ),
                    }
                )

        # extract git revision source code URL with SHA
        # only show source URL if all eval specs have the same revision
        source_url = None
        if ev.results:
            task_revisions = [
                tr.eval_spec.revision
                for tr in ev.results
                if tr.eval_spec and tr.eval_spec.revision
            ]
            if task_revisions and all(
                rev == task_revisions[0] for rev in task_revisions
            ):
                revision = task_revisions[0]

                # Only handle git revisions with complete info
                if (
                    revision
                    and revision.type == "git"
                    and revision.origin
                    and revision.commit
                ):
                    origin = revision.origin
                    commit = revision.commit

                    # Convert SSH URLs to HTTPS URLs
                    if origin.startswith("git@"):
                        # Convert git@github.com:user/repo.git to https://github.com/user/repo
                        origin = origin.replace("git@", "https://").replace(":", "/", 1)

                    # Remove .git suffix if present
                    if origin.endswith(".git"):
                        origin = origin[:-4]

                    # Only create URL if it looks like a valid HTTP(S) URL
                    if origin.startswith(("http://", "https://")):
                        source_url = f"{origin}/tree/{commit}"

        rows.append(
            {
                "id": sub.submit_time,
                "agent_name": sub.agent_name,
                "agent_description": sub.agent_description or "",
                "username": sub.username or "",
                "submit_time": date,
                "openness": sub.openness,
                "tool_usage": sub.tool_usage,
                "base_models": model_names,
                **flat,
                "logs_url": sub.logs_url if is_internal else sub.logs_url_public,
                "source_url": source_url,
            }
        )

    df = pd.DataFrame(rows)

    # prepare pretty column mapping
    pretty_cols = {c: _pretty_column_name(c) for c in df.columns}

    # construct overview table with human-friendly names
    overview = df.rename(columns=pretty_cols)

    return overview


def _pretty_column_name(col: str) -> str:
    """Map raw column name to display name."""
    # fixed mappings
    mapping = {
        "submit_time": "Submission date",
        "agent_name": "Agent",
        "agent_description": "Agent description",
        "username": "User/organization",
        "openness": "Openness",
        "tool_usage": "Agent tooling",
        "base_models": "LLM base",
        "logs_url": "Logs",
        "source_url": "Source",
        "overall/score": "Overall",
        "overall/cost": "Overall cost",
    }
    if col in mapping:
        return mapping[col]
    # dynamic: task/{name}/{metric} or tag/{name}/{metric}
    parts = col.split("/")
    if len(parts) == 3:
        _, name, metric = parts
        if metric == "score":
            return f"{name} score"
        if metric == "cost":
            return f"{name} cost"
        if metric == "score_ci":
            return f"{name} 95% CI"
        if metric == "cost_ci":
            return f"{name} cost 95% CI"
    # fallback to last segment
    return parts[-1]


def _plot_hbar(
    data: pd.DataFrame,
    agent_col: str,
    metrics: list[str],
) -> plt.Figure:
    """Horizontal bar chart of metrics, one row per agent."""
    n = len(metrics)
    # color each metric pair the same
    group_count = (n + 1) // 2
    palette = sns.color_palette(n_colors=group_count)

    # Set minimum width per subplot for readable x-axis labels, let height auto-size
    min_width_per_subplot = 4
    fig_width = n * min_width_per_subplot
    fig, axes = plt.subplots(
        ncols=n, sharey=True, figsize=(fig_width, plt.rcParams["figure.figsize"][1])
    )

    if n == 1:
        axes = [axes]

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        color = palette[idx // 2]

        sns.barplot(data=data, y=agent_col, x=metric, ax=ax, color=color)
        ci = data.get(f"{metric} 95% CI")
        if ci is not None and not ci.isna().all():
            y_positions = range(len(data))
            ax.errorbar(
                x=data[metric],
                y=y_positions,
                xerr=ci,
                fmt="none",
                ecolor="gray",
                capsize=3,
            )
        ax.set_xlabel(metric)
        ax.set_xlim(left=0)

    plt.tight_layout()
    return fig


def _plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    agent_col: str,
) -> plt.Figure:
    """Scatter plot of agent results, for showing score vs cost."""
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x=x, y=y, hue=agent_col, s=100, ax=ax)
    # attach error bars if available
    x_ci = f"{x} 95% CI"
    y_ci = f"{y} 95% CI"
    if x_ci in data.columns or y_ci in data.columns:
        ax.errorbar(
            x=data[x],
            y=data[y],
            xerr=data.get(x_ci),
            yerr=data.get(y_ci),
            fmt="none",
            ecolor="gray",
            alpha=0.5,
            capsize=3,
        )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    return fig


__all__ = ["LeaderboardViewer"]
