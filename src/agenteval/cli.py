#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO

import click
import datasets

from agenteval.leaderboard.schema_generator import load_dataset_features

from .cli_utils import AliasedChoice, generate_choice_help
from .config import load_suite_config
from .io import atomic_write_file
from .leaderboard.models import LeaderboardSubmission, Readme
from .leaderboard.upload import (
    compress_model_usages,
    sanitize_path_component,
    upload_folder_to_hf,
)
from .models import EvalConfig, SubmissionMetadata, TaskResults
from .score import process_eval_logs
from .summary import compute_summary_statistics

HF_URL_PATTERN = r"^hf://(?:datasets/)?(?P<repo_id>[^/]+/[^/]+)/(?P<path>.*)$"
EVAL_CONFIG_FILENAME = "eval_config.json"
SCORES_FILENAME = "scores.json"
SUMMARY_FILENAME = "summary_stats.json"
SUBMISSION_METADATA_FILENAME = "submission.json"
SUMMARIES_PREFIX = "summaries"
OPENNESS_MAPPING = {
    "c": "Closed",
    "api": "API Available",
    "os": "Open Source",
    "ow": "Open Source + Open Weights",
}
TOOL_MAPPING = {
    "s": "Standard",
    "css": "Custom with Standard Search",
    "c": "Fully Custom",
}


def parse_hf_url(url: str) -> tuple[str, str]:
    hf_url_match = re.match(HF_URL_PATTERN, url)
    if not hf_url_match:
        click.echo(
            f"Invalid URL: {url}. " "Expected format: hf://<repo_id>/<submission_path>"
        )
        sys.exit(1)

    return hf_url_match.group("repo_id"), hf_url_match.group("path")


def verify_git_reproducibility() -> None:
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
    help="Score a directory of evaluation logs. Can be a local directory or a HuggingFace URL.",
)
@click.argument(
    "log_dir",
    type=str,
)
def score_command(
    log_dir: str,
):
    hf_url_match = re.match(HF_URL_PATTERN, log_dir)
    temp_dir: tempfile.TemporaryDirectory | None = None
    if hf_url_match is not None:
        # Download the logs from HF URL
        from huggingface_hub import snapshot_download

        repo_id = hf_url_match.group("repo_id")
        submission_path = hf_url_match.group("path")
        temp_dir = tempfile.TemporaryDirectory()
        download_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=f"{submission_path}/*",
            local_dir=temp_dir.__enter__(),
        )
        log_dir = os.path.join(download_dir, submission_path)

    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        click.echo(f"No directory named {log_dir}")
        sys.exit(1)

    click.echo(f"Processing logs in {log_dir}")
    with open(os.path.join(log_dir, EVAL_CONFIG_FILENAME), "r", encoding="utf-8") as f:
        eval_config = EvalConfig.model_validate_json(f.read())

    log_processing_outcome = process_eval_logs(log_dir)

    if log_processing_outcome.errors:
        click.echo("Errors processing logs")
        for error in log_processing_outcome.errors:
            click.echo(f"  - {error}")
        sys.exit(1)

    task_results = TaskResults(results=log_processing_outcome.results)

    # Warn if multiple evaluation specs present
    if len(task_results.agent_specs) > 1:
        click.echo(
            f"Warning: Found {len(task_results.agent_specs)} different agent configurations. "
            "Use a single solver + model config per log directory to measure a single "
            "agent's performance across tasks."
        )
    if len(task_results.code_specs) > 1:
        click.echo(
            f"Warning: Found {len(task_results.code_specs)} different code versions "
            "(revision/packages). This may indicate mixed evaluation runs from "
            "different code states."
        )

        # Warn if user-specified task arguments are present

    if task_results.tasks_with_args:
        click.echo(
            f"Warning: User-specified task arguments found for tasks: {', '.join(task_results.tasks_with_args)}. "
            "For fair comparison, do not override the task arg defaults."
        )

    # Warn about any missing tasks
    missing_tasks = eval_config.task_names - task_results.task_names
    if missing_tasks:
        click.echo(f"Warning: Missing tasks in result set: {', '.join(missing_tasks)}")

    # Persist summary
    stats = compute_summary_statistics(
        eval_config.suite_config,
        eval_config.split,
        task_results.results or [],
    )

    if hf_url_match is None:
        # Persist scores
        scores_path = os.path.join(log_dir, SCORES_FILENAME)
        atomic_write_file(scores_path, task_results.model_dump_json(indent=2))
        click.echo(f"Wrote scores to {scores_path}")

        # Persist summary
        summary_path = os.path.join(log_dir, SUMMARY_FILENAME)
        atomic_write_file(
            summary_path, json.dumps(stats.model_dump(mode="json"), indent=2)
        )
        click.echo(f"Wrote summary scores to {summary_path}")

        if temp_dir is not None:
            temp_dir.__exit__(None, None, None)
    else:
        from huggingface_hub import HfApi

        hf_api = HfApi()
        path_in_repo = f"{SUMMARIES_PREFIX}/{submission_path}/{SCORES_FILENAME}"
        hf_api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=BytesIO(
                task_results.model_dump_json(indent=2).encode("utf-8")
            ),
            path_in_repo=path_in_repo,
        )
        click.echo(f"Uploaded scores to hf://{repo_id}/{path_in_repo}")

        path_in_repo = f"{SUMMARIES_PREFIX}/{submission_path}/{SUMMARY_FILENAME}"
        hf_api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=BytesIO(stats.model_dump_json(indent=2).encode("utf-8")),
            path_in_repo=path_in_repo,
        )
        click.echo(f"Uploaded summary to hf://{repo_id}/{path_in_repo}")


cli.add_command(score_command)


@click.command(
    name="publish",
    help="Upload Inspect logs to HuggingFace for official scoring",
)
@click.argument("log_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--submissions-repo-id",
    type=str,
    default=lambda: os.environ.get("SUBMISSIONS_REPO_ID", ""),
    help="HF repo id for submissions. Defaults to SUBMISSIONS_REPO_ID env var.",
)
@click.option(
    "-o",
    "--openness",
    type=AliasedChoice(OPENNESS_MAPPING),
    required=True,
    help=generate_choice_help(OPENNESS_MAPPING, "Level of openness for the agent."),
)
@click.option(
    "-t",
    "--tool-usage",
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
def publish_logs_command(
    log_dir: str,
    submissions_repo_id: str,
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

    with open(os.path.join(log_dir, EVAL_CONFIG_FILENAME), "r", encoding="utf-8") as f:
        eval_config = EvalConfig.model_validate_json(f.read())

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
    submission = SubmissionMetadata(
        username=username,
        agent_name=agent_name,
        agent_description=agent_description,
        agent_url=agent_url,
        submit_time=datetime.now(timezone.utc),
        openness=openness,
        tool_usage=tool_usage,
    )

    atomic_write_file(
        os.path.join(log_dir, SUBMISSION_METADATA_FILENAME),
        submission.model_dump_json(indent=2),
    )

    # Validate suite config version
    config_name = eval_config.suite_config.version
    if not config_name:
        raise click.ClickException("Suite config version is required for upload.")

    # Build submission name
    if submission.submit_time is None:
        raise click.ClickException("Submission timestamp is required for upload.")
    ts = submission.submit_time.strftime("%Y-%m-%dT%H-%M-%S")
    submission_name = f"{safe_username}_{safe_agent_name}_{ts}"

    # Upload logs and summary
    logs_url = upload_folder_to_hf(
        hf_api,
        log_dir,
        submissions_repo_id,
        config_name,
        eval_config.split,
        submission_name,
    )
    click.echo(f"Uploaded submission logs dir to {logs_url}")


cli.add_command(publish_logs_command)


@click.command(
    name="backfill",
    help="Backfill eval_config, scores, and submission files from legacy agenteval.json file",
)
@click.option(
    "--results-repo-id",
    type=str,
    required=False,
    default="allenai/asta-bench-internal-results",
)
@click.option(
    "--submissions-repo-id",
    type=str,
    required=False,
    default="allenai/asta-bench-internal-submissions",
)
@click.argument("submission_path", type=str)
def backfill_command(results_repo_id, submissions_repo_id, submission_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        submissions_dir = os.path.join(temp_dir, "submissions")
        results_dir = os.path.join(temp_dir, "results")

        import huggingface_hub

        api = huggingface_hub.HfApi()
        api.snapshot_download(
            repo_id=results_repo_id,
            repo_type="dataset",
            allow_patterns=[f"{submission_path}.json"],
            local_dir=results_dir,
        )
        api.snapshot_download(
            repo_id=submissions_repo_id,
            repo_type="dataset",
            allow_patterns=[f"{submission_path}/agenteval.json"],
            local_dir=submissions_dir,
        )

        lb_submission = LeaderboardSubmission.model_validate_json(
            open(os.path.join(results_dir, f"{submission_path}.json")).read()
        )
        with open(
            os.path.join(submissions_dir, f"{submission_path}/submission.json"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(lb_submission.submission.model_dump_json(indent=2))
        with open(
            os.path.join(submissions_dir, f"{submission_path}/eval_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            eval_config = EvalConfig(
                suite_config=lb_submission.suite_config, split=lb_submission.split
            )
            f.write(eval_config.model_dump_json(indent=2))
        os.makedirs(
            os.path.join(submissions_dir, f"{SUMMARIES_PREFIX}/{submission_path}"),
            exist_ok=True,
        )
        with open(
            os.path.join(
                submissions_dir, f"{SUMMARIES_PREFIX}/{submission_path}/scores.json"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            results = TaskResults(results=lb_submission.results)
            f.write(results.model_dump_json(indent=2))
        api.upload_folder(
            repo_id=submissions_repo_id,
            repo_type="dataset",
            folder_path=submissions_dir,
            path_in_repo="",
        )
        click.echo(f"Backfilled submission {submission_path} in {submissions_repo_id}")


cli.add_command(backfill_command)


@click.command(
    name="publish",
    help="Publish scored results in log_dir to HuggingFace leaderboard.",
)
@click.argument("submission_urls", nargs=-1, required=True, type=str)
@click.option(
    "--repo-id",
    default="allenai/asta-bench-internal-results",
    required=False,
    help="HuggingFace repo",
)
def publish_lb_command(repo_id: str, submission_urls: tuple[str, ...]):
    if not submission_urls:
        click.echo("At least one submission URL is required.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        from huggingface_hub import HfApi, snapshot_download

        local_submissions_dir = os.path.join(temp_dir, "submissions")
        local_results_dir = os.path.join(temp_dir, "results")

        hf_api = HfApi()

        submission_repo_ids = set()
        submission_paths = []

        # Validate URLs
        for submission_url in submission_urls:
            submission_repo_id, submission_path = parse_hf_url(
                submission_url
            )  # validates submission_url format "hf://<repo_id>/<submission_path>"
            submission_repo_ids.add(submission_repo_id)
            submission_paths.append(submission_path)

        if len(submission_repo_ids) > 1:
            click.echo("All submission URLs must reference the same repo")
            sys.exit(1)

        submission_repo_id = submission_repo_ids.pop()

        eval_config_rel_paths = [
            f"{p}/{EVAL_CONFIG_FILENAME}" for p in submission_paths
        ]
        scores_rel_paths = [
            f"{SUMMARIES_PREFIX}/{p}/{SCORES_FILENAME}" for p in submission_paths
        ]
        submission_metadata_rel_paths = [
            f"{p}/{SUBMISSION_METADATA_FILENAME}" for p in submission_paths
        ]

        # Download all input files in one shot
        snapshot_download(
            repo_id=submission_repo_id,
            repo_type="dataset",
            allow_patterns=eval_config_rel_paths
            + scores_rel_paths
            + submission_metadata_rel_paths,
            local_dir=local_submissions_dir,
        )

        # Create results files locally
        config_splits = defaultdict(
            list
        )  # Accumulate config names and splits being published
        for (
            submission_url,
            submission_path,
            eval_config_path,
            scores_path,
            submission_metadata_path,
        ) in zip(
            submission_urls,
            submission_paths,
            eval_config_rel_paths,
            scores_rel_paths,
            submission_metadata_rel_paths,
        ):
            local_eval_config_path = os.path.join(
                local_submissions_dir, eval_config_path
            )
            local_scores_path = os.path.join(local_submissions_dir, scores_path)
            local_submission_path = os.path.join(
                local_submissions_dir, submission_metadata_path
            )
            required_files = [
                local_eval_config_path,
                local_scores_path,
                local_submission_path,
            ]

            missing = [
                os.path.basename(f) for f in required_files if not os.path.exists(f)
            ]
            if missing:
                click.echo(
                    "Skipping {}: missing {}".format(
                        submission_path, ", ".join(missing)
                    )
                )
                continue

            eval_config = EvalConfig.model_validate_json(
                open(local_eval_config_path).read()
            )
            config_splits[eval_config.suite_config.name].append(eval_config.split)
            results = TaskResults.model_validate_json(
                open(local_scores_path).read()
            ).results
            submission = SubmissionMetadata.model_validate_json(
                open(local_submission_path).read()
            )
            submission.logs_url = submission_url.replace("hf://", "hf://datasets/", 1)
            lb_submission = LeaderboardSubmission(
                suite_config=eval_config.suite_config,
                split=eval_config.split,
                results=results,
                submission=submission,
            )
            lb_submission = compress_model_usages(lb_submission)
            os.makedirs(
                os.path.join(local_results_dir, os.path.dirname(submission_path)),
                exist_ok=True,
            )
            with open(
                os.path.join(local_results_dir, f"{submission_path}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(lb_submission.model_dump_json(indent=None))

        # Validate the config with the schema in HF
        readme = Readme.download_and_parse(repo_id)
        missing_configs = list(set(config_splits.keys()) - set(readme.configs.keys()))
        if missing_configs:
            click.echo(
                f"Config name {missing_configs} not present in hf://{repo_id}/README.md"
            )
            click.echo(
                f"Run 'update_readme.py add-config --repo-id {repo_id} --config-name {missing_configs[0]}' to add it"
            )
            sys.exit(1)
        missing_splits = list(
            set(((c, s) for c in config_splits.keys() for s in config_splits[c]))
            - set(((c, s) for c in readme.configs.keys() for s in readme.configs[c]))
        )
        if missing_splits:
            click.echo(
                f"Config/Split {missing_splits} not present in hf://{repo_id}/README.md"
            )
            click.echo(
                f"Run 'update_readme.py add-config --repo-id {repo_id} --config-name {missing_splits[0][0]} --split {missing_splits[0][1]}` to add it"
            )
            sys.exit(1)
        local_features = load_dataset_features()
        if local_features.arrow_schema != readme.features.arrow_schema:
            click.echo(
                "Schema in local dataset_features.yml does not match schema in hf://{repo_id}/README.md"
            )
            click.echo("Run 'update_readme.py sync-schema' to update it")
            sys.exit(1)

        # Upload all results files in one shot
        click.echo(f"Uploading {len(submission_paths)} results to {repo_id}...")
        hf_api.upload_folder(
            folder_path=local_results_dir,
            path_in_repo="",
            repo_id=repo_id,
            repo_type="dataset",
        )
        click.echo("Done")


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


lb.add_command(publish_lb_command)
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
    "--config-only",
    is_flag=True,
    help="Print the command that would be run and save eval_config locally.",
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
    config_only: bool,
    display: str,
    args: tuple[str],
):
    """Run inspect eval-set with arguments and append tasks"""
    suite_config = load_suite_config(config_path)
    tasks = suite_config.get_tasks(split)

    # Verify git status for reproducibility
    if not ignore_git:
        verify_git_reproducibility()

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

    inspect_command = (
        ["inspect", "eval-set"]
        + list(args)
        + logd_args
        + display_args
        + [x.path for x in tasks]
    )

    eval_config = EvalConfig(
        suite_config=suite_config, split=split, inspect_command=inspect_command
    )

    eval_config_path = os.path.join(log_dir, EVAL_CONFIG_FILENAME)
    if not os.path.exists(eval_config_path):
        with open(eval_config_path, "w", encoding="utf-8") as f:
            f.write(eval_config.model_dump_json(indent=2))
    else:
        with open(eval_config_path, "r", encoding="utf-8") as f:
            existing_config = EvalConfig.model_validate_json(f.read())
        if existing_config != eval_config:
            click.echo(
                f"Suite config does not match pre-existing config in {EVAL_CONFIG_FILENAME}. Rerun in an empty directory"
            )
            sys.exit(1)

    # We use subprocess here to keep arg management simple; an alternative
    # would be calling `inspect_ai.eval_set()` directly, which would allow for
    # programmatic execution
    if config_only:
        click.echo(f"Dry run: would run command: {' '.join(inspect_command)}")
        return

    click.echo(f"Running {' '.join(inspect_command)}")
    proc = subprocess.run(inspect_command)

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
