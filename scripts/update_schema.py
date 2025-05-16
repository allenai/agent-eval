#!/usr/bin/env python3
"""
Script to regenerate dataset_features.yml from the Pydantic schema.
"""
from pathlib import Path

from agenteval.leaderboard.schema_generator import write_dataset_features


def update_schema():
    repo_root = Path(__file__).parent.parent
    output_path = repo_root / "src" / "agenteval" / "dataset_features.yml"
    write_dataset_features(str(output_path))


def main():
    """Regenerate dataset_features.yml from Pydantic schema"""
    update_schema()
    print("✅ dataset_features.yml updated at src/agenteval/dataset_features.yml")


if __name__ == "__main__":
    main()
