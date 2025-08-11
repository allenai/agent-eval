#!/usr/bin/env python3
"""
Script to update the README.md file in the HuggingFace dataset repo containing leaderboard results
Keeps it in sync with the schema in dataset_features.yml
Also tracks supported config names (1.0.0, etc.) and splits (validation, test, etc.)
"""
import click

from agenteval.leaderboard.models import Readme
from agenteval.leaderboard.schema_generator import load_dataset_features


@click.group()
def main():
    pass


@main.command
@click.option(
    "--repo-id",
    default="allenai/asta-bench-internal-results",
    help="HuggingFace repo",
)
@click.option(
    "--dry-run", is_flag=True, help="Only print what would be done, do not upload"
)
def sync_schema(repo_id: str, dry_run: bool):
    """Update the README.md file with the current schema in dataset_features.yml"""
    readme = Readme.download_and_parse(repo_id)
    local_features = load_dataset_features()
    if local_features.arrow_schema == readme.features.arrow_schema:
        click.echo(f"Schema unchanged. Nothing to do.")
    elif dry_run:
        click.echo(f"Would update hf://{repo_id}/README.md to:")
        print(str(readme))
    else:
        readme.upload(repo_id=repo_id, comment="Update results schema")
        click.echo(f"Updated hf://{repo_id}/README.md")


@main.command()
@click.option(
    "--repo-id",
    default="allenai/asta-bench-internal-results",
    help="HuggingFace repo",
)
@click.option(
    "--config-name",
    required=False,
    help="Add config name to README if not already present",
)
@click.option(
    "--split",
    type=str,
    multiple=True,
    required=False,
    help="Add split(s) to config if not already present (can be specified multiple times)",
)
@click.option(
    "--dry-run", is_flag=True, help="Only print what would be done, do not upload"
)
def add_config(repo_id: str, config_name: str, split: tuple, dry_run: bool):
    """
    Add a config/split(s) to the README file
    """
    readme = Readme.download_and_parse(repo_id)

    status = "existing"
    if config_name not in readme.configs:
        readme.configs[config_name] = []
        status = "new"

    config_splits = readme.configs[config_name]

    to_add = set(split) - set(config_splits)
    readme.configs[config_name].extend(to_add)
    if not to_add:
        click.echo("No new config/splits specified. Nothing to do.")
    elif dry_run:
        click.echo(f"Would update hf://{repo_id}/README.md to:")
        print(str(readme))
    else:
        comment = f"Adding splits {list(to_add)} to {status} config '{config_name}'"
        click.echo(comment)
        readme.upload(repo_id, comment=comment)


if __name__ == "__main__":
    main()
