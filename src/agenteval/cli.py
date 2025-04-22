#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import click

from .config import load_suite_config
from .models import EvalConfig

EVAL_FILENAME = "agenteval.json"


@click.group()
def cli():
    pass


@click.command(
    name="score",
    help="Score a directory of evaluation logs and optionally upload to leaderboard.",
)
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--config",
    type=str,
    help="Path to a yml config file.",
    default=None,
)
@click.option(
    "--split",
    type=str,
    help="Config data split.",
    default=None,
)
@click.option(
    "--upload-hf",
    is_flag=True,
    help="Upload results to Huggingface leaderboard after scoring.",
)
@click.option(
    "--submissions-repo-id",
    type=str,
    default=lambda: os.environ.get("SUBMISSIONS_REPO_ID", ""),
    help="Hugging Face repository id for submissions. Defaults to SUBMISSIONS_REPO_ID env var.",
)
@click.option(
    "--results-repo-id",
    type=str,
    default=lambda: os.environ.get("RESULTS_REPO_ID", ""),
    help="Hugging Face repository id for result stats. Defaults to RESULTS_REPO_ID env var.",
)
@click.option(
    "--username",
    type=str,
    default=None,
    help="User or organization name for submission. Defaults to Hugging Face account name when not provided.",
)
@click.option(
    "--agent-name",
    type=str,
    help="Descriptive agent name for submission. Required when using --upload-hf.",
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
def score_command(
    log_dir: str,
    config: str | None,
    split: str | None,
    upload_hf: bool,
    submissions_repo_id: str,
    results_repo_id: str,
    username: str | None,
    agent_name: str | None,
    agent_description: str | None,
    agent_url: str | None,
):
    from .processor import score_directory
    from .summary import compute_summary_statistics

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

    # Perform Hugging Face upload and update submission metadata
    if upload_hf:
        from huggingface_hub import HfApi

        from .upload import upload_folder_to_hf, upload_summary_to_hf

        hf_api = HfApi()
        # Default username to Hugging Face account if not provided
        if not username:
            try:
                username = hf_api.whoami()["name"]
                click.echo(f"Defaulting username to Hugging Face account: {username}")
            except Exception:
                raise click.ClickException(
                    "--username must be provided or ensure HF authentication is configured"
                )

        # Require only agent name explicitly
        if not agent_name:
            raise click.ClickException(
                "--agent-name must be provided when using --upload-hf."
            )

        eval_result.submission.username = username
        eval_result.submission.agent_name = agent_name
        eval_result.submission.agent_description = agent_description
        eval_result.submission.agent_url = agent_url
        eval_result.submission.submit_time = datetime.now(timezone.utc)
        config_name = eval_result.suite_config.version
        if not config_name:
            raise click.ClickException(
                "Suite config version is required for uploading to Hugging Face."
            )
        timestamp = eval_result.submission.submit_time.strftime("%Y-%m-%dT%H-%M-%S")
        submission_name = f"{username}_{agent_name}_{timestamp}"

        logs_url = upload_folder_to_hf(
            hf_api,
            log_dir,
            submissions_repo_id,
            config_name,
            eval_result.split,
            submission_name,
        )
        click.echo(f"Uploaded submission logs dir to {logs_url}")
        eval_result.submission.logs_url = logs_url

        summary_url = upload_summary_to_hf(
            hf_api,
            eval_result,
            results_repo_id,
            config_name,
            eval_result.split,
            submission_name,
        )
        click.echo(f"Uploaded results summary file to {summary_url}")
        eval_result.submission.summary_url = summary_url

        # Save scored eval file with submission metadata
        eval_result.save_json(Path(log_dir) / EVAL_FILENAME)


cli.add_command(score_command)


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
    if not ignore_git:
        try:
            # Get current commit SHA and origin
            sha_result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )

            origin_result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=False,
            )

            sha = sha_result.stdout.strip() if sha_result.returncode == 0 else None
            origin = (
                origin_result.stdout.strip() if origin_result.returncode == 0 else None
            )

            # Check for dirty working directory
            git_dirty = (
                subprocess.run(
                    ["git", "diff", "--quiet", "--exit-code"],
                    capture_output=True,
                    check=False,
                ).returncode
                != 0
            )

            if git_dirty:
                raise click.ClickException(
                    f"Git working directory contains uncommitted changes. "
                    f"For reproducibility, Inspect will save: origin={origin}, sha={sha}. "
                    f"Please commit your changes or use --ignore-git to bypass this check (not recommended)."
                )

            # Check if commit exists on remote
            if sha:
                remote_exists = subprocess.run(
                    ["git", "branch", "-r", "--contains", sha],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()

                if not remote_exists:
                    raise click.ClickException(
                        f"Commit {sha} not found on remote '{origin}'. Others won't be able to "
                        f"access this code version. Please push your changes or use --ignore-git "
                        f"to bypass this check (not recommended)."
                    )
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            if isinstance(e, click.ClickException):
                raise
            raise click.ClickException(
                "Unable to verify git status for reproducibility. "
                "Use --ignore-git to bypass this check if git is not available."
            )

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
        f"You can now run '{ctx.parent.info_name if ctx.parent else 'cli'} score {log_dir}' to score the results and (optionally) upload to the leaderboard"
    )


cli.add_command(eval_command)


if __name__ == "__main__":
    cli()
