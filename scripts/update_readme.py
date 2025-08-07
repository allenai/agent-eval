#!/usr/bin/env python3
"""
Script to update the README.md file in the HuggingFace dataset repo containing leaderboard results
Keeps it in sync with the schema in dataset_features.yml
Also tracks supported config names (1.0.0, etc.) and splits (validation, test, etc.)
"""
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List

import click
import yaml
from datasets import Features
from pydantic import BaseModel

from agenteval.leaderboard.schema_generator import load_dataset_features


@dataclass
class Readme:
    text_content: str
    """Plain-text content section of the README."""
    configs: Dict[str, List[str]]
    """Mapping of config names to the splits they define"""
    features: Features
    """
    Features schema for leaderboard submission data
    The dataset_features.yml file is the definitive definition of the schema.
    All config names and splits must use the same schema.
    Use the update_schema script to regenerate the schema from code.
    """

    def __str__(self):
        configs = []
        for config_name, splits in self.configs.items():
            config_def = {
                "config_name": config_name,
                "data_files": [
                    {"split": split, "path": f"{config_name}/{split}/*.json"}
                    for split in splits
                ],
                "features": self.features._to_yaml_list(),
            }
            configs.append(config_def)
        yaml_section = yaml.dump({"configs": configs})
        return f"---\n{yaml_section.strip()}\n---{self.text_content.lstrip()}"

    @staticmethod
    def download_and_parse(repo_id: str) -> "Readme":
        """
        Download the README.md file from the specified HuggingFace repo and parse it.
        """
        from huggingface_hub.hf_api import HfApi

        tmp_file = HfApi().hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="README.md"
        )
        with open(tmp_file, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.match(r"(?s)^---\n(.*?)\n---\n(.*)", content)

        if not match or len(match.groups()) < 2:
            raise ValueError("No YAML front matter found in README.md")

        yaml_content = yaml.safe_load(match.group(1))
        text_content = match.group(2)
        configs = {}
        features = None
        for config_def in yaml_content["configs"]:
            config_name = config_def["config_name"]
            splits = [d["split"] for d in config_def.get("data_files", [])]
            configs[config_name] = splits
            config_features = Features._from_yaml_list(config_def.get("features", []))
            if features is None:
                features = config_features
            elif features != config_features:
                raise ValueError(
                    "Features for config '{}' do not match features for other configs '{}'".format(
                        config_name, ",".join(configs.keys())
                    )
                )

        return Readme(text_content=text_content, configs=configs, features=features)

    def upload(self, repo_id: str, comment: str):
        """
        Upload the README.md content back to the specified HuggingFace repo.
        """
        from huggingface_hub.hf_api import HfApi

        HfApi().upload_file(
            path_or_fileobj=BytesIO(str(self).encode("utf-8")),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=comment,
        )


@click.group()
def main():
    pass


@main.command
@click.option(
    "--repo-id",
    default="allenai/asta-bench-internal-results",
    help="HuggingFace repo",
)
def sync_schema(repo_id: str):
    """Update the README.md file with the current schema in dataset_features.yml"""
    readme = Readme.download_and_parse(repo_id)
    local_features = load_dataset_features()
    if local_features.arrow_schema == readme.features.arrow_schema:
        click.echo(f"Schema unchanged. Nothing to do.")
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
def add_config(repo_id: str, config_name: str, split: tuple):
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

    if to_add:
        comment = f"Adding splits {list(to_add)} to {status} config '{config_name}'"
        click.echo(comment)
        readme.configs[config_name].extend(to_add)
        readme.upload(repo_id, comment=comment)
    else:
        click.echo("No new config/splits specified. Nothing to do.")


if __name__ == "__main__":
    main()
