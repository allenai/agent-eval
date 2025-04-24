#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import click

from .config import load_suite_config
from .models import EvalConfig, EvalResult
from .processor import score_directory
from .summary import compute_summary_statistics
from .upload import sanitize_path_component, upload_folder_to_hf, upload_summary_to_hf

EVAL_FILENAME = "agenteval.json"


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
    "--config",
    type=str,
    help=f"Path to a yml config file. Ignored if {EVAL_FILENAME} exists.",
    default=None,
)
@click.option(
    "--split",
    type=str,
    help="Config data split. Ignored if {EVAL_FILENAME} exists.",
    default=None,
)
def score_command(
    log_dir: str,
    config: str | None,
    split: str | None,
):
    # Load or create EvalResult, process logs
    eval_result = score_directory(
        log_dir,
        config,
        split,
        eval_filename=EVAL_FILENAME,
    )

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

    # Persist updated EvalResult JSON
    eval_result.save_json(Path(log_dir) / EVAL_FILENAME)

    click.echo(f"Saved results to {log_dir}/{EVAL_FILENAME}")
    ctx = click.get_current_context()
    click.echo(
        f"You can now run '{ctx.parent.info_name if ctx.parent else 'cli'} publish {log_dir}' to publish the results"
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

    summary_url = upload_summary_to_hf(
        hf_api, eval_result, results_repo_id, config_name, eval_result.split, subm_name
    )
    click.echo(f"Uploaded results summary file to {summary_url}")
    eval_result.submission.summary_url = summary_url

    # Save updated JSON
    eval_result.save_json(Path(log_dir) / EVAL_FILENAME)
    click.echo(f"Updated {EVAL_FILENAME} with publication metadata.")


cli.add_command(publish_command)


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
    "--config",
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
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def eval_command(
    log_dir: str | None, config: str, split: str, ignore_git: bool, args: tuple[str]
):
    """Run inspect eval-set with arguments and append tasks"""
    suite_config = load_suite_config(config)
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

    # We use subprocess here to keep arg management simple; an alternative
    # would be calling `inspect_ai.eval_set()` directly, which would allow for
    # programmatic execution
    full_command = (
        ["inspect", "eval-set"] + list(args) + logd_args + [x.path for x in tasks]
    )
    click.echo(f"Running {config}: {' '.join(full_command)}")
    proc = subprocess.run(full_command)

    if proc.returncode != 0:
        raise click.ClickException(f"inspect eval-set failed while running {config}")

    # Write the config portion of the results file
    with open(os.path.join(log_dir, EVAL_FILENAME), "w", encoding="utf-8") as f:
        unscored_eval_config = EvalConfig(suite_config=suite_config, split=split)
        f.write(unscored_eval_config.model_dump_json(indent=2))

    ctx = click.get_current_context()
    click.echo(
        f"You can now run '{ctx.parent.info_name if ctx.parent else 'cli'} score {log_dir}' to score the results"
    )


cli.add_command(eval_command)


if __name__ == "__main__":
    cli()
