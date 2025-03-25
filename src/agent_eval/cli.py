#!/usr/bin/env python3
import os
import shutil
import subprocess
from datetime import datetime
from statistics import mean

import click
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from .config import Task, get_tasks
from .score import ResultSet, get_result_set


def upload_to_hf(repo_id: str, path_in_repo: str, log_dir: str):
    hf_api = HfApi()
    hf_api.upload_folder(
        folder_path=log_dir,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )


def check_hf_repo(repo_id: str, create_repo: bool, create_private_repo: bool) -> None:
    hf_api = HfApi()
    try:
        # Check if the repository exists.
        hf_api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        if create_repo:
            try:
                hf_api.create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=create_private_repo,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create repository: {e}")
        else:
            raise RuntimeError(
                f"Repository {repo_id} does not exist. Use --create-repo to create it."
            )


def find_missing_tasks(result_set: ResultSet, taskset: list[Task]) -> list[str]:
    return [
        task.task_name for task in taskset if task.task_name not in result_set.results
    ]


@click.group()
def cli():
    pass


@click.command(
    name="score",
    help="Score a directory of evaluation logs and optionally upload to leaderboard.",
)
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--upload-hf",
    is_flag=True,
    help="Upload results to Huggingface leaderboard after scoring.",
)
@click.option(
    "--results-repo-id",
    type=str,
    default=lambda: os.environ.get("RESULTS_REPO_ID", ""),
    help="Hugging Face repository identifier to upload results. Defaults to RESULTS_REPO_ID env var.",
)
@click.option(
    "--solver-name",
    type=str,
    help='Solver name to use as submission key (e.g. "username/descriptive_solver_name"). Required when using --upload-hf.',
)
@click.option(
    "--create-repo",
    is_flag=True,
    help="Create the repository if it does not exist.",
)
@click.option(
    "--create-private-repo/--no-create-private-repo",
    default=True,
    help="Create the repository as private (default: True).",
)
@click.option(
    "--taskset",
    type=str,
    default="astabench",
    help="Name of the task configuration to use or path to a custom YAML file.",
)
def score_command(
    log_dir: str,
    upload_hf: bool,
    results_repo_id: str,
    solver_name: str,
    create_repo: bool,
    create_private_repo: bool,
    taskset: str,
):
    result_set = get_result_set(log_dir)
    result_set.submission_name = solver_name or None
    result_set.created_at = datetime.now()
    click.echo("Scores:\n" + str({t: r.metrics for t, r in result_set.results.items()}))
    click.echo(
        "Average cost for 1k samples:\n"
        + str(
            {
                t: f"${mean(r.model_costs)*1000:.2f}"
                for t, r in result_set.results.items()
            }
        )
    )
    results_json_path = os.path.join(log_dir, "results.json")

    tasks = get_tasks(taskset)
    missing_tasks = find_missing_tasks(result_set, tasks)
    if missing_tasks:
        click.echo(
            f"Warning: Some tasks from the '{taskset}' configuration are missing from this result set: {', '.join(missing_tasks)}"
        )

    if upload_hf:
        if not solver_name:
            raise click.ClickException(
                "--solver-name must be provided when using --upload-hf."
            )
        # Validate solver_name format
        parts = solver_name.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise click.ClickException(
                "--solver-name must be in the format 'username/descriptive_solver_name'."
            )
        if results_repo_id:
            try:
                check_hf_repo(results_repo_id, create_repo, create_private_repo)
            except RuntimeError as e:
                raise click.ClickException(str(e))
            path_in_repo = f"results/{solver_name}"
            result_set.logs_url = f"hf://datasets/{results_repo_id}/{path_in_repo}"
            with open(results_json_path, "w", encoding="utf-8") as f:
                f.write(result_set.model_dump_json(indent=2))
            # Make a timestamped record of the results
            timestamp = result_set.created_at.strftime("%Y-%m-%dT%H-%M-%S")
            timestamped_path = os.path.join(log_dir, f"results_{timestamp}.json")
            shutil.copy(results_json_path, timestamped_path)
            upload_to_hf(results_repo_id, path_in_repo, log_dir)
        else:
            click.echo("No repository ID provided for upload.")
    else:
        with open(results_json_path, "w", encoding="utf-8") as f:
            f.write(result_set.model_dump_json(indent=2))


cli.add_command(score_command)


@cli.command(
    name="eval",
    help="Run inspect eval-set on specified tasks with the given arguments",
    context_settings={"ignore_unknown_options": True},
)
@click.option("--log-dir", type=str)
@click.option(
    "--taskset",
    type=str,
    default="astabench",
    help="Name of the task configuration to use or path to a custom YAML file.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def run_eval(log_dir: str | None, taskset: str, args: tuple[str]):
    """Run inspect eval-set with arguments and append tasks"""
    if not log_dir:
        log_dir = os.environ.get("INSPECT_LOG_DIR")
        if not log_dir:
            log_dir = (
                f"./logs/{taskset}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
            )
            click.echo(f"No log dir was manually set; using {log_dir}")
    logd_args = ["--log-dir", log_dir]

    tasks = get_tasks(taskset)

    # We use subprocess here to keep arg management simple; an alternative
    # would be calling `inspect_ai.eval_set()` directly, which would allow for
    # programmatic execution
    full_command = (
        ["inspect", "eval-set"] + list(args) + logd_args + [x.task_path for x in tasks]
    )
    click.echo(f"Running {taskset}: {' '.join(full_command)}")
    proc = subprocess.run(full_command)

    if proc.returncode != 0:
        raise click.ClickException(f"inspect eval-set failed while running {taskset}")

    click.echo(
        f"You can now run '{cli.name} score {log_dir} --taskset {taskset}' to score the results and (optionally) upload to the leaderboard"
    )


cli.add_command(run_eval)

if __name__ == "__main__":
    cli()
