#!/usr/bin/env python3
"""
Script to regenerate dataset_features.yml from the Pydantic schema.
"""
import sys
from pathlib import Path

import click

from agenteval.leaderboard.models import LeaderboardSubmission
from agenteval.leaderboard.schema_generator import (
    features_from_pydantic,
    load_dataset_features,
    write_dataset_features,
)


def update_schema():
    repo_root = Path(__file__).parent.parent
    # write schema under leaderboard subpackage
    schema_dir = repo_root / "src" / "agenteval" / "leaderboard"
    output_path = schema_dir / "dataset_features.yml"
    write_dataset_features(str(output_path))
    print("✅ dataset_features.yml updated under src/agenteval/leaderboard")


def check_schema():
    from_file = load_dataset_features()
    from_code = features_from_pydantic(LeaderboardSubmission)
    if from_file.arrow_schema == from_code.arrow_schema:
        click.echo("✅ Schema is up to date.")
    else:
        click.echo("❌ Schema is out of date. Please run the update command.")
        sys.exit(1)


@click.command()
@click.option(
    "--check", is_flag=True, help="Check if schema is up to date without writing"
)
def main(check: bool):
    """Regenerate dataset_features.yml from Pydantic schema"""
    if check:
        check_schema()
    else:
        update_schema()


if __name__ == "__main__":
    main()
