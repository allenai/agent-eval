import re
from dataclasses import dataclass
from io import BytesIO

import yaml
from datasets import Features
from pydantic import BaseModel, Field

from ..models import SubmissionMetadata, SuiteConfig, TaskResult


class LeaderboardSubmission(BaseModel):
    suite_config: SuiteConfig
    """Task configuration for the results."""

    split: str
    """Split used for the results."""

    results: list[TaskResult] | None = None
    submission: SubmissionMetadata = Field(default_factory=SubmissionMetadata)


@dataclass
class Readme:
    text_content: str
    """Plain-text content section of the README."""
    configs: dict[str, list[str]]
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
