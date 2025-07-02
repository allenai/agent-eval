#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import datasets

from .cli_utils import AliasedChoice, generate_choice_help
from .config import load_suite_config
from .leaderboard.upload import (
    compress_model_usages,
    sanitize_path_component,
    upload_folder_to_hf,
    upload_summary_to_hf,
)
from .models import EvalConfig, EvalResult
from .score import process_eval_logs
from .summary import compute_summary_statistics

EVAL_FILENAME = "agenteval.json"
OPENNESS_MAPPING = {
    "c": "Closed",
    "api": "API Available",
    "os": "Open Source",
    "ow": "Open Source + Open Weights"
}
TOOL_MAPPING = {
    "s": "Standard",
    "css": "Custom with Standard Search",
    "c": "Fully Custom"
}


def verify_git_reproducibility(ignore_git: bool) -> None:
    if ignore_git:
        return
    try:
        # Get current commit SHA and origin
        sha_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        origin_result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = sha_result.stdout.strip() if sha_result.returncode == 0 else None
        origin = origin_result.stdout.strip() if origin_result.returncode == 0 else None

        # Check for dirty working directory
        git_dirty = (
            subprocess.run(
                ["git", "diff", "--quiet", "--exit-code"],
                capture_output=True,
                check=False,
            ).returncode
            != 0
        )

        # Warn about untracked (non-ignored) files
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True,
        )
        untracked_files = untracked_result.stdout.strip().splitlines()
        if untracked_files:
            click.echo(
                f"Warning: Untracked files present: {', '.join(untracked_files)}. "
                "For reproducibility, please add, ignore, or remove these files."
            )

        # Abort if worktree is dirty
        if git_dirty:
            raise click.ClickException(
                f"Git working directory contains uncommitted changes. "
                f"For reproducibility, Inspect will save: origin={origin}, sha={sha}. "
                "Please commit your changes or use --ignore-git to bypass this check (not recommended)."
            )

        # Check if commit exists on remote
        if sha:
            remote_exists = subprocess.run(
                ["git", "branch", "-r", "--contains", sha],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if not remote_exists:
                raise click.ClickException(
                    f"Commit {sha} not found on remote '{origin}'. Others won't be able to "
                    "access this code version. Please push your changes or use --ignore-git "
                    "to bypass this check (not recommended)."
                )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        if isinstance(e, click.ClickException):
            raise
        raise click.ClickException(
            f"Unable to verify git status for reproducibility: {e}. "
            "Use --ignore-git to bypass this check if git is not available."
        )


@click.group()
def cli():
    pass


@click.command(
    name="score",
    help="Score a directory of evaluation logs.",
)
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--config-path",
    "config_path",
    type=str,
    help=f"Path to a yml config file. Ignored if {EVAL_FILENAME} exists.",
    default=None,
)
@click.option(
    "--split",
    type=str,
    help=f"Config data split. Ignored if {EVAL_FILENAME} exists.",
    default=None,
)
def score_command(
    log_dir: str,
    config_path: str | None,
    split: str | None,
):
    # Load or create EvalResult and process logs (inlined from processor)
    json_path = Path(log_dir) / EVAL_FILENAME
    if json_path.exists():
        try:
            raw = json_path.read_text(encoding="utf-8")
            eval_result = EvalResult.model_validate_json(raw)
        except Exception as e:
            raise click.ClickException(
                f"Failed to load existing '{EVAL_FILENAME}' at {json_path}: {e}"
            )
        if config_path:
            try:
                cli_cfg = load_suite_config(config_path)
                if cli_cfg.version != eval_result.suite_config.version:
                    click.echo(
                        f"Warning: CLI config version '{cli_cfg.version}' "
                        f"does not match JSON config version '{eval_result.suite_config.version}'."
                    )
            except Exception as e:
                click.echo(
                    f"Warning: could not load CLI config '{config_path}' for comparison: {e}"
                )
        if split and split != eval_result.split:
            raise click.ClickException(
                f"Split mismatch: JSON split '{eval_result.split}' != CLI split '{split}'"
            )
    else:
        if not config_path or not split:
            raise click.ClickException(
                "--config-path and --split must be provided when no existing result JSON"
            )
        suite_cfg = load_suite_config(config_path)
        eval_result = EvalResult(suite_config=suite_cfg, split=split)

    task_results, eval_specs, had_errors = process_eval_logs(log_dir)
    eval_result.eval_specs = eval_specs
    eval_result.results = task_results

    # Warn if multiple evaluation specs present
    if eval_result.eval_specs and len(eval_result.eval_specs) > 1:
        click.echo(
            f"Warning: Found {len(eval_result.eval_specs)} different eval specs. "
            "Logs may come from mixed runs."
        )

    # Warn about any missing tasks
    missing_tasks = eval_result.find_missing_tasks()
    if missing_tasks:
        click.echo(f"Warning: Missing tasks in result set: {', '.join(missing_tasks)}")

    # Compute and display summary statistics
    stats = compute_summary_statistics(
        eval_result.suite_config,
        eval_result.split,
        eval_result.results or [],
    )
    click.echo("Summary statistics:")
    click.echo(json.dumps({k: v.model_dump() for k, v in stats.items()}, indent=2))

    if had_errors:
        click.echo(
            "Error: Errors occurred while computing some metrics. No scores will be written to `agenteval.json`"
        )
        sys.exit(1)

    # Persist updated EvalResult JSON
    eval_result.save_json(Path(log_dir) / EVAL_FILENAME)

    click.echo(f"Saved results to {log_dir}/{EVAL_FILENAME}")
    ctx = click.get_current_context()
    click.echo(
        f"You can now run '{ctx.parent.info_name if ctx.parent else 'cli'} publish --agent-name <your-agent-name> --submissions-repo-id <your-submissions-repo-id> --results-repo-id <your-results-repo-id> {log_dir}' to publish the results"
    )


cli.add_command(score_command)


@click.command(
    name="publish",
    help="Publish scored results in log_dir to Hugging Face leaderboard.",
)
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--submissions-repo-id",
    type=str,
    default=lambda: os.environ.get("SUBMISSIONS_REPO_ID", ""),
    help="HF repo id for submissions. Defaults to SUBMISSIONS_REPO_ID env var.",
)
@click.option(
    "--results-repo-id",
    type=str,
    default=lambda: os.environ.get("RESULTS_REPO_ID", ""),
    help="HF repo id for result stats. Defaults to RESULTS_REPO_ID env var.",
)
@click.option(
    "-o", "--openness",
    type=AliasedChoice(OPENNESS_MAPPING),
    required=True,
    help=generate_choice_help(OPENNESS_MAPPING, "Level of openness for the agent."),
)
@click.option(
    "-t", "--tool-usage",
    type=AliasedChoice(TOOL_MAPPING),
    required=True,
    help=generate_choice_help(TOOL_MAPPING, "Tool choices available to the agent."),
)
@click.option(
    "--username",
    type=str,
    default=None,
    help="HF username/org for submission. Defaults to your HF account name.",
)
@click.option(
    "--agent-name",
    type=str,
    required=True,
    help="Descriptive agent name for submission.",
)
@click.option(
    "--agent-description",
    type=str,
    default=None,
    help="Description of the agent being submitted.",
)
@click.option(
    "--agent-url",
    type=str,
    default=None,
    help="URL to the agent's repository or documentation.",
)
def publish_command(
    log_dir: str,
    submissions_repo_id: str,
    results_repo_id: str,
    openness: str,
    tool_usage: str,
    username: str | None,
    agent_name: str,
    agent_description: str | None,
    agent_url: str | None,
):
    # Allow huggingface imports to be optional
    from huggingface_hub import HfApi

    # Derive a filesafe agent_name
    safe_agent_name = sanitize_path_component(agent_name)
    if safe_agent_name != agent_name:
        click.echo(
            f"Note: agent_name '{agent_name}' contains unsafe characters; "
            f"using '{safe_agent_name}' for submission filenames."
        )

    # Load existing scored results from JSON
    json_path = Path(log_dir) / EVAL_FILENAME
    if not json_path.exists():
        raise click.ClickException(f"No scored results found at {json_path}")
    raw = json_path.read_text(encoding="utf-8")
    eval_result = EvalResult.model_validate_json(raw)

    # Validate eval result
    if not eval_result.is_scored():
        raise click.ClickException(
            f"{EVAL_FILENAME} is not scored. Please run 'score {log_dir}' first."
        )
    missing_tasks = eval_result.find_missing_tasks()
    if missing_tasks:
        click.echo(f"Warning: Missing tasks in result set: {', '.join(missing_tasks)}")

    # Determine HF user
    hf_api = HfApi()
    if not username:
        try:
            username = hf_api.whoami()["name"]
            assert isinstance(username, str), "Invalid username type from HF API"
            click.echo(f"Defaulting username to Hugging Face account: {username}")
        except Exception:
            raise click.ClickException(
                "--username must be provided or ensure HF authentication is configured"
            )

    # Derive a filesafe username
    safe_username = sanitize_path_component(username)
    if safe_username != username:
        click.echo(
            f"Note: username '{username}' contains unsafe characters; "
            f"using '{safe_username}' for submission filenames."
        )

    # Fill submission metadata
    eval_result.submission.username = username
    eval_result.submission.agent_name = agent_name
    eval_result.submission.agent_description = agent_description
    eval_result.submission.agent_url = agent_url
    eval_result.submission.submit_time = datetime.now(timezone.utc)
    eval_result.submission.openness = openness
    eval_result.submission.tool_usage = tool_usage

    # Validate suite config version
    config_name = eval_result.suite_config.version
    if not config_name:
        raise click.ClickException("Suite config version is required for upload.")

    # Build submission name
    ts = eval_result.submission.submit_time.strftime("%Y-%m-%dT%H-%M-%S")
    subm_name = f"{safe_username}_{safe_agent_name}_{ts}"

    # Upload logs and summary
    logs_url = upload_folder_to_hf(
        hf_api, log_dir, submissions_repo_id, config_name, eval_result.split, subm_name
    )
    click.echo(f"Uploaded submission logs dir to {logs_url}")
    eval_result.submission.logs_url = logs_url

    compressed_usages_result = compress_model_usages(eval_result)

    summary_url = upload_summary_to_hf(
        hf_api,
        compressed_usages_result,
        results_repo_id,
        config_name,
        eval_result.split,
        subm_name,
    )
    click.echo(f"Uploaded results summary file to {summary_url}")
    eval_result.submission.summary_url = summary_url

    # Save updated JSON
    eval_result.save_json(Path(log_dir) / EVAL_FILENAME)
    click.echo(f"Updated {EVAL_FILENAME} with publication metadata.")


@click.group(name="lb", help="Leaderboard related commands")
def lb():
    pass


def validate_config(ctx, param, value):
    if value is not None:
        return value
    repo_id = ctx.params.get("repo_id")
    configs = datasets.get_dataset_config_names(repo_id)
    click.echo(f"Available configs: {configs}")
    click.echo("Please specify a config via --config")
    ctx.exit()


def validate_split(ctx, param, value):
    if value is not None:
        return value
    repo_id = ctx.params.get("repo_id")
    config = ctx.params.get("config")
    splits = datasets.get_dataset_split_names(repo_id, config_name=config)
    click.echo(f"Available splits: {splits}")
    click.echo("Please specify a split via --split")
    ctx.exit()


@lb.command(name="view", help="View leaderboard results.")
@click.option(
    "--repo-id",
    envvar="RESULTS_REPO_ID",
    required=True,
    help="HuggingFace dataset ID",
)
@click.option(
    "--config",
    default=None,
    callback=validate_config,
    help="Name of the dataset configuration to load",
)
@click.option(
    "--split",
    default=None,
    callback=validate_split,
    help="Dataset split to load",
)
@click.option(
    "--tag",
    default=None,
    help="If provided, show detail for this tag instead of overview",
)
@click.option(
    "--dump-plots/--no-plots",
    default=False,
    help="Enable saving plots",
)
@click.option(
    "--plot-dir",
    default="plots",
    type=click.Path(),
    show_default=True,
    help="Base directory for saving plots",
)
def view_command(repo_id, config, split, tag, dump_plots, plot_dir):
    """View a specific config and split; show overview or tag detail."""
    from .leaderboard.view import LeaderboardViewer

    viewer = LeaderboardViewer(repo_id, config, split)

    df, plots = viewer.view(tag, with_plots=True)
    click.echo(df.to_string(index=False))

    if dump_plots:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_repo = repo_id.replace("/", "_")
        base = plot_dir
        sub = f"{safe_repo}_{config}_{split}"
        subdir = tag or "overview"
        outdir = os.path.join(base, sub, f"{subdir}_{ts}")
        os.makedirs(outdir, exist_ok=True)

        csv_path = os.path.join(outdir, f"{subdir}.csv")
        df.to_csv(csv_path, index=False)
        click.echo(f"Saved data: {csv_path}")

        for name, fig in plots.items():
            path = os.path.join(outdir, f"{name}.png")
            fig.savefig(path, bbox_inches="tight")
            click.echo(f"Saved plot: {path}")


lb.add_command(publish_command)
cli.add_command(lb)


@cli.command(
    name="eval",
    help="Run inspect eval-set on specified tasks with the given arguments",
    context_settings={"ignore_unknown_options": True},
)
@click.option(
    "--log-dir",
    type=str,
    help="Log directory. Defaults to INSPECT_LOG_DIR or auto-generated under ./logs.",
)
@click.option(
    "--config-path",
    "config_path",
    type=str,
    help="Path to a yml config file.",
    required=True,
)
@click.option(
    "--split",
    type=str,
    help="Config data split.",
    required=True,
)
@click.option(
    "--ignore-git",
    is_flag=True,
    help="Ignore git reproducibility checks (not recommended).",
)
@click.option(
    "--display",
    type=str,
    # https://github.com/UKGovernmentBEIS/inspect_ai/issues/1891 and
    # https://github.com/allenai/nora-issues-research/issues/77#issuecomment-2877262319
    # TODO: remove this once fixed
    help="Display format. Defaults to plain.",
    default="plain",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def eval_command(
    log_dir: str | None,
    config_path: str,
    split: str,
    ignore_git: bool,
    display: str,
    args: tuple[str],
):
    """Run inspect eval-set with arguments and append tasks"""
    suite_config = load_suite_config(config_path)
    tasks = suite_config.get_tasks(split)

    # Verify git status for reproducibility
    verify_git_reproducibility(ignore_git)

    if not log_dir:
        log_dir = os.environ.get("INSPECT_LOG_DIR")
        if not log_dir:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            log_dir = os.path.join(
                ".",
                "logs",
                f"{suite_config.name}_{suite_config.version}_{split}_{timestamp}",
            )
            click.echo(f"No log dir was manually set; using {log_dir}")
    logd_args = ["--log-dir", log_dir]
    display_args = ["--display", display]

    # Write the config portion of the results file
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, EVAL_FILENAME), "w", encoding="utf-8") as f:
        unscored_eval_config = EvalConfig(suite_config=suite_config, split=split)
        f.write(unscored_eval_config.model_dump_json(indent=2))

    # We use subprocess here to keep arg management simple; an alternative
    # would be calling `inspect_ai.eval_set()` directly, which would allow for
    # programmatic execution
    full_command = (
        ["inspect", "eval-set"]
        + list(args)
        + logd_args
        + display_args
        + [x.path for x in tasks]
    )
    click.echo(f"Running {config_path}: {' '.join(full_command)}")
    proc = subprocess.run(full_command)

    if proc.returncode != 0:
        raise click.ClickException(
            f"inspect eval-set failed while running {config_path}"
        )

    ctx = click.get_current_context()
    click.echo(
        f"You can now run '{ctx.parent.info_name if ctx.parent else 'cli'} score {log_dir}' to score the results"
    )


cli.add_command(eval_command)


if __name__ == "__main__":
    cli()
