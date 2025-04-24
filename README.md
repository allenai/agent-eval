# agent-eval

This library provides utilites for evaluating agents on a suite of [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai)-formatted evals, with the following primary benefits:
1. Task suite specifications as config.
2. Utilities to extract the token usage of the agent from log files, and compute cost using `litellm`.
3. Utilities for submitting task suite results to a leaderboard, with submission metadata and easy upload to a HuggingFace repo for distribution of scores and logs.

# Installation

To install from pypi, use `pip install agent-eval`.

For development, clone the repo and run `pip install -e ".[dev]"`.

# Usage

It provides several CLI commands:
- `agenteval eval --config [CONFIG_PATH] [INSPECT_EVAL_SET_OPTIONS]`: This will evaluate an agent on the supplied eval suite configuration (see `src/agenteval/config.py` for the config model definition).
- `agenteval score [LOG_DIR]`: This will score the results in `results.json` and optionally upload the logs and results to HuggingFace datasets (using `--upload-hf` option).

# Defining a suite config

Note that the suite config requires specifying a `primary_metric` for each task, for aggregation in a leaderboard, specified as `{scorer_name}/{metric_name}`. The scoring utils will look for a corresponding stderr metric, by looking for another metric with the same `scorer_name` and with a `metric_name` containing the string "stderr".

# Publication

To publish to pypi, use `bash publish.sh` (pypi token should be supplied as an environment variable).
