#!/usr/bin/env python3
"""
Script to regenerate dataset_features.yml from the Pydantic schema.
"""
from pathlib import Path

from agenteval.leaderboard.schema_generator import write_dataset_features


def update_schema():
    repo_root = Path(__file__).parent.parent
    # write schema under leaderboard subpackage
    schema_dir = repo_root / "src" / "agenteval" / "leaderboard"
    output_path = schema_dir / "dataset_features.yml"
    write_dataset_features(str(output_path))


def main():
    """Regenerate dataset_features.yml from Pydantic schema"""
    update_schema()
    print("✅ dataset_features.yml updated under src/agenteval/leaderboard")


if __name__ == "__main__":
    main()
