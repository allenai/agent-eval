import os
import subprocess
import tempfile
from pathlib import Path

import click


def atomic_write_file(
    path: str | Path,
    content: str,
    encoding: str = "utf-8",
) -> None:
    """
    Write the given content string to `path` atomically.

    Writes to a temporary file in the same directory, fsyncs, then replaces.
    """
    p = Path(path)
    # write to temp file in same directory
    tmp = tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding=encoding, dir=str(p.parent)
    )
    try:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    # replace target atomically; if successful, attempt to delete any leftover temp
    try:
        os.replace(tmp.name, str(p))
    except Exception:
        # keep temp file for debugging if replace fails
        raise
    else:
        # atomic replace removed tmp.name; if it still exists, remove it
        try:
            os.remove(tmp.name)
        except FileNotFoundError:
            pass


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
