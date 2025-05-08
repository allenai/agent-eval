# Development Instructions

# Setup

Clone the repo and run `pip install -e ".[dev]"`.

# Publication

To publish to pypi:

```shell
export PYPI_TOKEN=...
bash scripts/publish.sh
```

# Schema update

The results leaderboard on HuggingFace uses a fixed schema to prevent inferred schema problems.
As the results model changes over time, schema adjustments may be required (a CI check should fail).
The schema can be inferred from the Pydantic model and re-computed using:

```shell
python scripts/update_schema.py
```
