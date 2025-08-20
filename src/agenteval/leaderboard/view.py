"""
View and plot leaderboard results.
"""

import logging
from dataclasses import dataclass
from typing import Literal
from zoneinfo import ZoneInfo

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import EvalRevision
from matplotlib.figure import Figure

from .. import compute_summary_statistics
from ..config import SuiteConfig
from .model_name_mapping import LB_MODEL_NAME_MAPPING
from .models import LeaderboardSubmission

logger = logging.getLogger(__name__)

# Font size constants for scatter plots
SCATTER_SUBPLOT_TITLE_FONTSIZE = 11
SCATTER_AXIS_LABEL_FONTSIZE = 9
SCATTER_TICK_LABEL_FONTSIZE = 8
SCATTER_LEGEND_FONTSIZE = 7


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

    def _load(
        self,
        apply_pretty_names: bool = True,
        preserve_none_scores: bool = False,
    ):
        results = datasets.load_dataset(self._repo_id, name=self._config)
        overview = _get_dataframe(
            eval_results=results,
            split=self._split,
            is_internal=self._internal,
            suite_config=self._cfg,
            apply_pretty_names=apply_pretty_names,
            preserve_none_scores=preserve_none_scores,
        )
        return overview, self.tag_map

    def view(
        self,
        tag: str | None = None,
        with_plots: bool = False,
        preserve_none_scores: bool = False,
        exclude_primary_metric: bool = False,
        duplicate_handling: Literal["latest", "index"] = "index",
        exclude_agent_patterns: list[str] | None = None,
        include_task_patterns: list[str] | None = None,
        scatter_show_missing_cost: bool = False,
        scatter_legend_max_width: int | None = None,
        scatter_figure_width: float | None = None,
        scatter_subplot_height: float | None = None,
        scatter_subplot_spacing: float | None = None,
        scatter_x_log_scale: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Figure]]:
        """
        If tag is None, primary="Overall" and group=all tags.
        Otherwise primary=tag and group=tasks under that tag.
        """
        # Load raw data for internal processing
        raw_data, tag_map = self._load(
            apply_pretty_names=False, preserve_none_scores=preserve_none_scores
        )

        # Create combined agent names with model information
        if not raw_data.empty:

            def create_display_name(row):
                agent_name = row["agent_name"]
                base_models = row["base_models"]
                if base_models and len(base_models) > 0:
                    models_str = ", ".join(base_models)
                    return f"{agent_name} ({models_str})"
                return agent_name

            raw_data["display_name"] = raw_data.apply(create_display_name, axis=1)

        # Handle duplicate agents based on specified strategy
        if not raw_data.empty:
            if duplicate_handling == "latest":
                raw_data = (
                    raw_data.sort_values("submit_time", ascending=False)
                    .drop_duplicates(subset=["display_name"], keep="first")
                    .reset_index(drop=True)
                )
            elif duplicate_handling == "index":
                # Sort by time, then add sequential numbers to duplicate display names
                raw_data = raw_data.sort_values("submit_time", ascending=True)
                raw_data["display_name"] = (
                    raw_data["display_name"]
                    + "_"
                    + (raw_data.groupby("display_name").cumcount() + 1).astype(str)
                )
                raw_data["display_name"] = raw_data["display_name"].str.replace(
                    "_1$", "", regex=True
                )

        # Raw column names (will be converted to pretty names for final display)
        raw_cols = [
            "id",
            "agent_name",
            "display_name",
            "agent_description",
            "username",
            "submit_time",
            "logs_url",
            "source_url",
            "openness",
            "tool_usage",
            "base_models",
        ]

        # choose primary metric and its sub‐group (using raw column names)
        if tag is None:
            primary = "overall/score"
            group = list(tag_map.keys())
        else:
            primary = f"tag/{tag}/score"
            group = tag_map.get(tag, [])

        # Check if the primary column exists before sorting
        if primary not in raw_data.columns:
            raise KeyError(
                f"Column '{primary}' not found. Available columns: {list(raw_data.columns)}"
            )

        # Filter tasks/sub-benchmarks by include patterns (only applies when viewing a specific tag)
        # Preserve the order of include_task_patterns arguments
        if tag is not None and include_task_patterns and group:
            import re

            filtered_group = []
            for pattern in include_task_patterns:
                for task_name in group:
                    if (
                        re.match(pattern, task_name, re.IGNORECASE)
                        and task_name not in filtered_group
                    ):
                        filtered_group.append(task_name)
            group = filtered_group

        raw_data = raw_data.sort_values(primary, ascending=False)

        # build full metric list: primary + its cost + each member and its cost (using raw names)
        if tag is None:
            # For overall view, group contains tag names
            metrics = [primary, "overall/cost"] + [
                m for t in group for m in (f"tag/{t}/score", f"tag/{t}/cost")
            ]
        else:
            # For tag view, group contains task names
            metrics = [primary, f"tag/{tag}/cost"] + [
                m for t in group for m in (f"task/{t}/score", f"task/{t}/cost")
            ]

        # Get CI columns for error bar plotting (only available for task-level metrics)
        ci_cols = []
        for m in metrics:
            if m.startswith("task/") and (m.endswith("/score") or m.endswith("/cost")):
                ci_col = f"{m}_ci"
                if ci_col in raw_data.columns:
                    ci_cols.append(ci_col)

        # Keep raw column names for internal processing, include CI columns
        available_metrics = [c for c in metrics if c in raw_data.columns]

        # Exclude primary metric if requested
        if exclude_primary_metric:
            # Remove primary metric and its cost from available_metrics
            primary_cost = primary.replace("/score", "/cost")
            available_metrics = [
                m for m in available_metrics if m not in (primary, primary_cost)
            ]

            # Also remove corresponding CI columns
            primary_ci = f"{primary}_ci"
            primary_cost_ci = f"{primary_cost}_ci"
            ci_cols = [c for c in ci_cols if c not in (primary_ci, primary_cost_ci)]

        raw_df = raw_data.loc[
            :,
            raw_cols + available_metrics + ci_cols,
        ].reset_index(drop=True)

        # Always filter out rows with all NaN score values (no point showing agents with no data)
        score_cols = [c for c in available_metrics if c.endswith("/score")]
        if score_cols:
            raw_df = raw_df.dropna(subset=score_cols, how="all")

        # Filter out agents matching any exclude pattern (based on display name)
        if exclude_agent_patterns:
            for pattern in exclude_agent_patterns:
                mask = ~raw_df["display_name"].str.match(pattern, case=False, na=False)
                raw_df = raw_df[mask].reset_index(drop=True)

        # Build scatter pairs for score/cost metrics
        scatter_pairs = []
        if tag is None:
            # Overall view: primary="overall/score", group=[tag names]
            scatter_pairs.append(
                (primary, "overall/cost")
            )  # ("overall/score", "overall/cost")
            for tag_name in group:
                scatter_pairs.append((f"tag/{tag_name}/score", f"tag/{tag_name}/cost"))
        else:
            # Tag view: primary="tag/{tag}/score", group=[task names]
            scatter_pairs.append(
                (primary, f"tag/{tag}/cost")
            )  # ("tag/lit/score", "tag/lit/cost")
            for task_name in group:
                scatter_pairs.append(
                    (f"task/{task_name}/score", f"task/{task_name}/cost")
                )

        plots: dict[str, Figure] = {}
        if with_plots:
            # Use available_metrics which already has primary excluded if requested
            plots["bar"] = _plot_hbar(
                raw_df,
                agent_col="display_name",
                metrics=available_metrics,
            )

            # Filter to only valid pairs that exist in the data
            valid_pairs = [
                (y, x)
                for y, x in scatter_pairs
                if x in raw_df.columns and y in raw_df.columns
            ]

            # Always generate combined scatter plot if we have valid pairs
            if valid_pairs:
                plots["scatter"] = _plot_combined_scatter(
                    raw_df,
                    scatter_pairs=valid_pairs,
                    agent_col="display_name",
                    use_cost_fallback=scatter_show_missing_cost,
                    legend_max_width=scatter_legend_max_width,
                    figure_width=scatter_figure_width,
                    subplot_height=scatter_subplot_height,
                    subplot_spacing=scatter_subplot_spacing,
                    use_log_scale=scatter_x_log_scale,
                )

            # Also generate individual scatter plots
            for y, x in valid_pairs:
                # Create plot name from raw column name
                plot_name = (
                    y.replace("/score", "").replace("tag/", "").replace("task/", "")
                )
                plots[f"scatter_{plot_name}"] = _plot_scatter(
                    raw_df,
                    x=x,
                    y=y,
                    agent_col="display_name",
                    use_cost_fallback=scatter_show_missing_cost,
                    legend_max_width=scatter_legend_max_width,
                    figure_width=scatter_figure_width,
                    subplot_height=scatter_subplot_height,
                    use_log_scale=scatter_x_log_scale,
                )

        # Calculate frontier information for each scatter pair
        # scatter_pairs was built earlier: [(score_col, cost_col), ...]
        for y_col, x_col in scatter_pairs:
            if x_col in raw_df.columns and y_col in raw_df.columns:
                # Create frontier column name from the score column
                # Keep the raw column name format for now, will be prettified later
                frontier_col_name = y_col.replace("/score", "/frontier")
                # Get frontier indices and create boolean series
                frontier_indices = _get_frontier_indices(raw_df, x_col, y_col)
                raw_df[frontier_col_name] = raw_df.index.isin(frontier_indices)

        # Create final display DataFrame with pretty column names
        display_df = raw_df.copy()
        pretty_cols = {c: _pretty_column_name(c) for c in display_df.columns}
        display_df = display_df.rename(columns=pretty_cols)

        return display_df, plots


def _agent_with_probably_incomplete_model_usage_info(agent_name):
    # See https://github.com/allenai/astabench-issues/issues/330
    lowered_agent_name = agent_name.lower()
    is_elicit = lowered_agent_name == "elicit"
    is_scispace = lowered_agent_name == "scispace"
    is_you_dot_com = ("you" in lowered_agent_name) and ("com" in lowered_agent_name)
    return any([is_elicit, is_scispace, is_you_dot_com])


@dataclass(frozen=True)
class GitRevision:
    normalized_origin: str
    commit: str

    def source_url(self) -> str:
        return f"{self.normalized_origin}/tree/{self.commit}"


def construct_reproducibility_url(task_revisions: list[EvalRevision]) -> str | None:
    source_url = None

    complete_git_revisions = [
        r for r in task_revisions if r.type == "git" and r.origin and r.commit
    ]
    if len(complete_git_revisions) > 0:

        revs_of_interest: set[GitRevision] = set([])
        for revision in complete_git_revisions:
            origin = revision.origin
            commit = revision.commit

            # Convert SSH URLs to HTTPS URLs
            if origin.startswith("git@"):
                # Convert git@github.com:user/repo.git to https://github.com/user/repo
                origin = origin.replace(":", "/", 1).replace("git@", "https://")

            # Remove .git suffix if present
            if origin.endswith(".git"):
                origin = origin[:-4]

            # Only create URL if it looks like a valid HTTP(S) URL
            if origin.startswith(("http://", "https://")):
                revs_of_interest.add(
                    GitRevision(normalized_origin=origin, commit=commit)
                )

        if len(revs_of_interest) > 0:
            # Try to be somewhat consistent about what gets picked...
            source_url = sorted(list(revs_of_interest), key=lambda r: r.commit)[
                -1
            ].source_url()

    return source_url


def _get_dataframe(
    eval_results: datasets.DatasetDict,
    split: str,
    is_internal: bool,
    suite_config: SuiteConfig,
    timezone: str = "US/Pacific",
    apply_pretty_names: bool = True,
    preserve_none_scores: bool = False,
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
        sub = ev.submission

        probably_incomplete_model_info = (
            _agent_with_probably_incomplete_model_usage_info(sub.agent_name)
        )

        model_token_counts: dict[str, int] = {}
        if ev.results:
            for task_result in ev.results:

                if probably_incomplete_model_info:
                    logger.warning(
                        f"Dropping model_usages and model_costs for the following submission because model usage info may be incomplete: {sub}."
                    )
                    task_result.model_usages = None
                    task_result.model_costs = None

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
            suite_config=cfg,
            split=split,
            results=ev.results,
            preserve_none_scores=preserve_none_scores,
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
            source_url = construct_reproducibility_url(task_revisions)

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

    if apply_pretty_names:
        # prepare pretty column mapping
        pretty_cols = {c: _pretty_column_name(c) for c in df.columns}
        # construct overview table with human-friendly names
        overview = df.rename(columns=pretty_cols)
        return overview
    else:
        return df


def _pretty_column_name(col: str) -> str:
    """Map raw column name to display name."""
    # fixed mappings
    mapping = {
        "submit_time": "Submission date",
        "agent_name": "Agent",
        "display_name": "Agent (with models)",
        "agent_description": "Agent description",
        "username": "User/organization",
        "openness": "Openness",
        "tool_usage": "Agent tooling",
        "base_models": "LLM base",
        "logs_url": "Logs",
        "source_url": "Source",
        "overall/score": "Overall",
        "overall/cost": "Overall cost",
        "overall/frontier": "Overall frontier",
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
        if metric == "frontier":
            return f"{name} frontier"
    # fallback to last segment
    return parts[-1]


def _plot_hbar(
    data: pd.DataFrame,
    agent_col: str,
    metrics: list[str],
) -> Figure:
    """Horizontal bar chart of metrics, one row per agent."""
    import seaborn as sns

    n = len(metrics)
    # color each metric pair the same
    group_count = (n + 1) // 2
    palette = sns.color_palette(n_colors=group_count)

    # Set minimum width per subplot for readable x-axis labels, scale height with number of agents
    min_width_per_subplot = 4
    min_height_per_agent = 0.4  # Minimum height per agent row
    fig_width = n * min_width_per_subplot
    fig_height = max(
        6, len(data) * min_height_per_agent
    )  # At least 6 inches, scale with agents
    fig, axes = plt.subplots(ncols=n, sharey=True, figsize=(fig_width, fig_height))

    if n == 1:
        axes = [axes]

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        color = palette[idx // 2]

        sns.barplot(data=data, y=agent_col, x=metric, ax=ax, color=color)
        ci_col = f"{metric}_ci"
        if ci_col in data.columns:
            ci = data[ci_col]  # CI already computed as 95% CI
            # Get actual y-positions from this subplot's barplot
            y_positions = [
                patch.get_y() + patch.get_height() / 2 for patch in ax.patches
            ]
            # Ensure we have the right number of positions
            if len(y_positions) == len(data):
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
        # Adjust font sizes to be proportional to the scaled figure height
        # Since we scale height with number of agents, text should scale too
        font_size = max(10, min(16, len(data) * 0.5))
        ax.tick_params(axis="y", labelsize=font_size)
        ax.tick_params(axis="x", labelsize=font_size)
        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size)

    plt.tight_layout()
    return fig


def _plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    agent_col: str,
    use_cost_fallback: bool = False,
    legend_max_width: int | None = None,
    figure_width: float | None = None,
    subplot_height: float | None = None,
    use_log_scale: bool = False,
) -> Figure:
    """Scatter plot of agent results, for showing score vs cost."""
    import seaborn as sns

    # Create figure with constrained layout for automatic spacing
    if figure_width is not None or subplot_height is not None:
        # Only specify figsize if user provided dimensions
        width = figure_width if figure_width is not None else 8
        if subplot_height is not None:
            fig, ax = plt.subplots(
                figsize=(width, subplot_height), layout="constrained"
            )
        else:
            # Width specified but not height - use matplotlib's default aspect
            fig, ax = plt.subplots(layout="constrained")
            fig.set_size_inches(width, fig.get_figheight())
    else:
        # Use matplotlib defaults for everything
        fig, ax = plt.subplots(layout="constrained")

    # Get unique agents for consistent coloring
    unique_agents = data[agent_col].unique()
    palette = sns.color_palette(n_colors=len(unique_agents))
    agent_colors = dict(zip(unique_agents, palette))

    _plot_single_scatter_subplot(
        ax,
        data,
        x,
        y,
        agent_col,
        agent_colors,
        use_cost_fallback,
        collect_legend=True,  # Need to collect legend entries for single plots
        use_log_scale=use_log_scale,
    )

    # Sort and format legend entries
    handles, labels = ax.get_legend_handles_labels()
    sorted_handles, sorted_labels = _sort_legend_entries(handles, labels)

    # Apply text wrapping if specified
    if legend_max_width is not None:
        legend_labels = [
            _wrap_legend_text(label, legend_max_width) for label in sorted_labels
        ]
    else:
        legend_labels = sorted_labels

    # Place legend to the right of plot
    ax.legend(
        sorted_handles,
        legend_labels,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=SCATTER_LEGEND_FONTSIZE,
        frameon=True,
    )
    return fig


def _plot_single_scatter_subplot(
    ax,
    data: pd.DataFrame,
    x: str,
    y: str,
    agent_col: str,
    agent_colors: dict,
    use_cost_fallback: bool = False,
    collect_legend: bool = False,
    show_xlabel: bool = True,
    use_log_scale: bool = False,
) -> tuple[list, list]:
    """Plot a single scatter subplot. Returns (handles, labels) if collect_legend=True."""
    plot_data = data.dropna(subset=[y])

    # Apply x-axis scaling BEFORE plotting any data
    if use_log_scale:
        ax.set_xscale("log")

    # Separate agents with real costs vs those without cost data
    has_cost_data = (
        plot_data[x].notna()
        if x in plot_data.columns
        else pd.Series(False, index=plot_data.index)
    )
    real_cost_data = plot_data[has_cost_data]
    no_cost_data = plot_data[~has_cost_data]

    handles, labels = [], []

    # Plot agents with real cost data
    if not real_cost_data.empty:
        # Get frontier indices first for drawing the frontier line
        frontier_indices = _get_frontier_indices(real_cost_data, x, y)

        for agent in real_cost_data[agent_col].unique():
            agent_data = real_cost_data[real_cost_data[agent_col] == agent]

            # Plot all points with the same marker
            scatter = ax.scatter(
                agent_data[x],
                agent_data[y],
                color=agent_colors[agent],
                marker="o",
                label=agent if collect_legend else "",
            )

            if collect_legend:
                handles.append(scatter)
                labels.append(agent)

        max_x = real_cost_data[x].max()

        # Add score-cost frontier curve
        if frontier_indices:
            frontier_points = real_cost_data.loc[frontier_indices, [x, y]]
            frontier_line = ax.plot(
                frontier_points[x],
                frontier_points[y],
                color="firebrick",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
                label="Efficiency Frontier",
            )[
                0
            ]  # plot returns a list, we want the line object

            # Add frontier to legend entries (will be sorted to top by _sort_legend_entries)
            if collect_legend:
                handles.append(frontier_line)
                labels.append("Efficiency Frontier")
    else:
        max_x = 1

    # Add error bars for real cost data
    if not real_cost_data.empty:
        _plot_error_bars(ax, real_cost_data, x, y)

    # Error bars for fallback agents are handled later when we plot the no-cost points

    # Store dividing line position for later tick cleanup and drawing
    dividing_line_x = None
    should_draw_dividing_line = False
    if use_cost_fallback and not real_cost_data.empty:
        # Calculate dividing line position directly here to avoid any scoping issues
        dividing_line_x = real_cost_data[x].max() * 1.15
        should_draw_dividing_line = not no_cost_data.empty

    # Set y-axis limits with minimal padding
    ax.set_ylim(bottom=0, top=None)  # Let matplotlib auto-scale the top
    ax.margins(y=0.05)  # Add only 5% margin at the top

    # Apply consistent axis formatting
    _setup_axis_formatting(ax, x, y, show_xlabel)
    # Note: Legend font at 7pt visually matches 8pt tick labels due to matplotlib rendering differences

    # Set axis limits based on scale type
    if use_log_scale and not real_cost_data.empty:
        # For log scale, set left limit to a nice round value
        min_x = real_cost_data[x].min()
        import numpy as np

        # Use natural left limit based on each subplot's data for better visual consistency
        floor_log = np.floor(np.log10(min_x))
        left_limit = 10**floor_log

        # Disable autoscaling before setting limits
        ax.autoscale(enable=False, axis="x")

        if use_cost_fallback and not no_cost_data.empty:
            # Use percentage-of-individual-span approach for consistent visual proportions
            fallback_x_position, right_limit = _calculate_fallback_position_and_limits(
                max_x, use_log_scale, left_limit
            )
            ax.set_xlim(left=left_limit, right=right_limit)

            # Plot no-cost agents at fallback position
            for agent in no_cost_data[agent_col].unique():
                agent_data = no_cost_data[no_cost_data[agent_col] == agent]
                scatter = ax.scatter(
                    [fallback_x_position] * len(agent_data),
                    agent_data[y],
                    facecolors="none",
                    edgecolors=agent_colors[agent],
                    linewidths=2,
                    label=f"{agent} (no cost)",
                )
                if collect_legend:
                    handles.append(scatter)
                    labels.append(f"{agent} (no cost)")
        else:
            # Add some padding on the right
            max_x = real_cost_data[x].max()
            ax.set_xlim(left=left_limit, right=max_x * 2)

    else:
        # For linear scale, use 0 as left limit
        if use_cost_fallback and not no_cost_data.empty:
            # Use fixed percentage approach: no-cost section takes 10% of plot width
            fallback_x_position, right_limit = _calculate_fallback_position_and_limits(
                max_x, use_log_scale=False, left_limit=0
            )
            ax.set_xlim(left=0, right=right_limit)

            # Plot no-cost agents at fallback position
            for agent in no_cost_data[agent_col].unique():
                agent_data = no_cost_data[no_cost_data[agent_col] == agent]
                scatter = ax.scatter(
                    [fallback_x_position] * len(agent_data),
                    agent_data[y],
                    facecolors="none",
                    edgecolors=agent_colors[agent],
                    linewidths=2,
                    label=f"{agent} (no cost)",
                )
                if collect_legend:
                    handles.append(scatter)
                    labels.append(f"{agent} (no cost)")
        else:
            ax.set_xlim(left=0)

    # Now draw the dividing line if needed
    if should_draw_dividing_line and dividing_line_x is not None:
        ax.axvline(x=dividing_line_x, color="gray", linestyle="--", alpha=0.5)

    # Filter ticks to respect our limits and dividing line
    from matplotlib.ticker import FixedLocator

    if dividing_line_x is not None:
        # Get current ticks and filter those past the dividing line
        all_ticks = ax.get_xticks()
        # Be more conservative - filter ticks at or past the dividing line
        visible_ticks = [
            t for t in all_ticks if t < dividing_line_x * 0.99
        ]  # Small buffer for floating point

        # For log scale, also ensure ticks are within axis limits
        if use_log_scale:
            xlim = ax.get_xlim()
            visible_ticks = [t for t in visible_ticks if xlim[0] <= t <= xlim[1]]

            # Use FixedLocator to prevent matplotlib from adding ticks back
            ax.xaxis.set_major_locator(FixedLocator(visible_ticks))

            # Also filter minor ticks
            minor_ticks = ax.xaxis.get_minorticklocs()
            visible_minor_ticks = [
                t
                for t in minor_ticks
                if t < dividing_line_x * 0.99 and xlim[0] <= t <= xlim[1]
            ]
            ax.xaxis.set_minor_locator(FixedLocator(visible_minor_ticks))
        else:
            # For non-log scale, just set major ticks
            ax.xaxis.set_major_locator(FixedLocator(visible_ticks))

    elif use_log_scale:
        # Even without dividing line, filter out ticks outside limits for log scale
        xlim = ax.get_xlim()
        all_ticks = ax.get_xticks()
        visible_ticks = [t for t in all_ticks if xlim[0] <= t <= xlim[1]]
        ax.xaxis.set_major_locator(FixedLocator(visible_ticks))

    return handles, labels


def _plot_combined_scatter(
    data: pd.DataFrame,
    scatter_pairs: list[tuple[str, str]],
    agent_col: str,
    use_cost_fallback: bool = False,
    legend_max_width: int | None = None,
    figure_width: float | None = None,
    subplot_height: float | None = None,
    subplot_spacing: float | None = None,
    use_log_scale: bool = False,
) -> Figure:
    """Combined scatter plot with multiple score/cost pairs in subplots and single legend."""
    import seaborn as sns

    n_plots = len(scatter_pairs)

    # Always use single column layout for simplicity
    cols = 1
    rows = n_plots

    # Use tight layout instead of constrained for better control
    # Determine figure size
    figsize = None
    if figure_width is not None or subplot_height is not None:
        if figure_width is not None:
            fig_width = figure_width
        else:
            fig_width = 8  # Default width when only height is specified

        if subplot_height is not None:
            figure_height = subplot_height * rows
        else:
            # Only width specified, let matplotlib determine height based on default aspect
            figure_height = plt.rcParams["figure.figsize"][1] * rows
        figsize = (fig_width, figure_height)

    # Create subplots with optional parameters
    if subplot_spacing is not None:
        gridspec_kw = {"hspace": subplot_spacing}
    else:
        gridspec_kw = None

    if figsize is not None:
        fig, axes = plt.subplots(rows, cols, figsize=figsize, gridspec_kw=gridspec_kw)
    else:
        # Use all matplotlib defaults
        fig, axes = plt.subplots(rows, cols, gridspec_kw=gridspec_kw)

    # Handle layout based on whether spacing is customized
    legend_space = (
        0.28 if legend_max_width and figure_width and figure_width <= 6.5 else 0
    )

    if subplot_spacing is not None:
        # When spacing is overridden, use subplots_adjust for precise control
        # Align top with legend
        fig.subplots_adjust(
            top=0.97,
            bottom=0.02,
            left=0.1,
            right=1 - legend_space,
            hspace=subplot_spacing,
        )
    else:
        # Use tight_layout for nice matplotlib defaults
        fig.tight_layout(rect=(0, 0, 1 - legend_space, 1))

    # Handle axes normalization for single column
    if n_plots == 1:
        axes = [axes]
    else:
        axes = list(axes)

    # Get unique agents for consistent coloring
    unique_agents = data[agent_col].unique()
    palette = sns.color_palette(n_colors=len(unique_agents))
    agent_colors = dict(zip(unique_agents, palette))

    # Plot each subplot
    all_handles, all_labels = [], []
    for idx, (y, x) in enumerate(scatter_pairs):
        ax = axes[idx]

        # Only show x-axis label on the bottom subplot
        is_bottom_subplot = idx == len(scatter_pairs) - 1

        # Plot subplot and collect legend info from all subplots
        handles, labels = _plot_single_scatter_subplot(
            ax,
            data,
            x,
            y,
            agent_col,
            agent_colors,
            use_cost_fallback,
            collect_legend=True,
            show_xlabel=is_bottom_subplot,
            use_log_scale=use_log_scale,
        )

        # Merge legend entries from all subplots
        for handle, label in zip(handles, labels):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)

        # Set subplot title
        title = (
            y.replace("/score", "")
            .replace("tag/", "")
            .replace("task/", "")
            .replace("overall", "Overall")
        )
        ax.set_title(title, fontsize=SCATTER_SUBPLOT_TITLE_FONTSIZE)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    # Add single legend with sorted entries
    if all_handles:
        sorted_handles, sorted_labels = _sort_legend_entries(all_handles, all_labels)

        # Wrap legend text if specified
        legend_labels = (
            [_wrap_legend_text(label, legend_max_width) for label in sorted_labels]
            if legend_max_width
            else sorted_labels
        )

        # Position legend to the right of plots
        bbox = axes[-1].get_position()

        fig.legend(
            sorted_handles,
            legend_labels,
            bbox_to_anchor=(bbox.x1 + 0.02, 0.99),
            loc="upper left",
            fontsize=SCATTER_LEGEND_FONTSIZE,
            ncol=1,
            columnspacing=0.5,
            handletextpad=0.5,
            borderaxespad=0.5,
            frameon=True,
            markerfirst=True,
            markerscale=1.0,
            bbox_transform=fig.transFigure,
        )

    return fig


def _get_frontier_indices(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> list:
    """
    Get indices of rows that are on the efficiency frontier.
    Core frontier calculation logic used by both plotting and data marking.
    """
    # Filter to rows with valid cost and score data
    valid_mask = data[x_col].notna() & data[y_col].notna()
    valid_data = data[valid_mask]

    if valid_data.empty:
        return []

    # Sort by cost (ascending) and score (descending)
    sorted_data = valid_data.sort_values(by=[x_col, y_col], ascending=[True, False])

    frontier_indices = []
    max_score_so_far = float("-inf")

    for idx, row in sorted_data.iterrows():
        score = row[y_col]
        if score > max_score_so_far:
            frontier_indices.append(idx)
            max_score_so_far = score

    return frontier_indices


def _sort_legend_entries(handles, labels):
    """Sort legend entries: Efficiency Frontier first, then regular entries (alphabetical), then '(no cost)' entries (alphabetical)."""
    if not handles or not labels:
        return handles, labels

    legend_pairs = list(zip(handles, labels))

    # Sort with custom key:
    # - Priority 0: Efficiency Frontier (always first)
    # - Priority 1: Regular entries (alphabetical)
    # - Priority 2: (no cost) entries (alphabetical)
    def sort_key(pair):
        label = pair[1]
        if label == "Efficiency Frontier":
            return (0, label)
        elif label.endswith("(no cost)"):
            return (2, label)
        else:
            return (1, label)

    legend_pairs.sort(key=sort_key)
    return zip(*legend_pairs)


def _wrap_legend_text(text, max_width=35):
    """Wrap legend text to fit within specified width."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width=max_width))


def _setup_axis_formatting(ax, x_label: str, y_label: str, show_xlabel: bool = True):
    """Apply consistent axis formatting including labels and tick styling."""
    if show_xlabel:
        # Simplify x-axis label to just the part after final "/"
        simplified_xlabel = x_label.split("/")[-1]
        ax.set_xlabel(simplified_xlabel, fontsize=SCATTER_AXIS_LABEL_FONTSIZE)
    else:
        ax.set_xlabel("", fontsize=SCATTER_AXIS_LABEL_FONTSIZE)  # Hide x-axis label

    # Simplify y-axis label to just the part after final "/"
    simplified_ylabel = y_label.split("/")[-1]
    ax.set_ylabel(simplified_ylabel, fontsize=SCATTER_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=SCATTER_TICK_LABEL_FONTSIZE)


def _plot_error_bars(
    ax, data: pd.DataFrame, x: str, y: str, x_positions=None, y_positions=None
):
    """Add error bars to a plot if CI columns are available."""
    x_ci = f"{x}_ci"
    y_ci = f"{y}_ci"

    # Determine if we have error bar data
    has_x_err = x_ci in data.columns
    has_y_err = y_ci in data.columns

    if not (has_x_err or has_y_err):
        return

    # Use provided positions or data columns
    x_vals = x_positions if x_positions is not None else data[x]
    y_vals = y_positions if y_positions is not None else data[y]

    x_err = data[x_ci] if has_x_err else None
    y_err = data[y_ci] if has_y_err else None

    ax.errorbar(
        x=x_vals,
        y=y_vals,
        xerr=x_err,
        yerr=y_err,
        fmt="none",
        ecolor="gray",
        alpha=0.5,
        capsize=3,
    )


def _calculate_fallback_position_and_limits(
    max_x: float, use_log_scale: bool = False, left_limit: float = 0
) -> tuple[float, float]:
    """Calculate the x-position for agents without cost data and the right axis limit.

    Uses a proportional approach that scales the no-cost section based on the
    total data range for more consistent visual percentages across plots.
    Returns (fallback_position, right_limit).
    """
    if use_log_scale:
        import numpy

        # Fixed log width that provides good breathing room for no-cost points
        dividing_line = max_x * 1.15

        # Calculate log width as a proportion of the total log range
        # This gives more consistent visual percentages across different data ranges
        log_left = numpy.log10(left_limit) if left_limit > 0 else numpy.log10(max_x) - 2
        log_dividing = numpy.log10(dividing_line)
        total_log_range = log_dividing - log_left

        # Use 10% of the total log range for no-cost section
        # This matches the linear scale percentage for consistency
        nocost_log_width = total_log_range * 0.10

        log_right = log_dividing + nocost_log_width
        right_limit = 10**log_right

        # Fallback position geometrically centered in no-cost section
        nocost_log_center = log_dividing + (nocost_log_width / 2)
        fallback_position = 10**nocost_log_center

        return fallback_position, right_limit
    else:
        # For linear scale: no-cost section should be 10% of total range
        # Data range: 0 to max_x * 1.15
        data_range = max_x * 1.15

        # No-cost section should be 10% of total plot width
        # If data takes 90% and no-cost takes 10%, then:
        # nocost_section_width = (10/90) * data_range = 0.111... * data_range
        nocost_section_width = (1.0 / 9.0) * data_range

        # Right limit
        right_limit = data_range + nocost_section_width

        # Fallback position centered in no-cost section
        fallback_position = data_range + (nocost_section_width / 2)

        return fallback_position, right_limit


__all__ = ["LeaderboardViewer"]
