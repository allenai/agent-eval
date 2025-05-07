#!/usr/bin/env python3
"""
Script to regenerate dataset_infos.json from the Pydantic schema.
"""
from pathlib import Path

from agenteval.schema_generator import generate_dataset_infos


def update_schema():
    repo_root = Path(__file__).parent.parent
    output_path = repo_root / "dataset_infos.json"
    generate_dataset_infos(str(output_path))


def main():
    """Regenerate dataset_infos.json from Pydantic schema"""
    update_schema()
    print("✅ dataset_infos.json updated at dataset_infos.json")


if __name__ == "__main__":
    main()
