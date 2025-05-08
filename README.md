# agent-eval

A utility for evaluating agents on a suite of [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai)-formatted evals, with the following primary benefits:
1. Task suite specifications as config.
2. Extracts the token usage of the agent from log files, and computes cost using `litellm`.
3. Submits task suite results to a leaderboard, with submission metadata and easy upload to a HuggingFace repo for distribution of scores and logs.

# Installation

To install from pypi, use `pip install agent-eval`.

# Usage

## Run evaluation suite
```shell
agenteval eval --config-path CONFIG_PATH --split SPLIT LOG_DIR
```
Evaluate an agent on the supplied eval suite configuration. Results are written to `agenteval.json` in the log directory. 

See [sample-config.yml](sample-config.yml) for a sample configuration file. 

For aggregation in a leaderboard, each task specifies a `primary_metric` as `{scorer_name}/{metric_name}`. 
The scoring utils will look for a corresponding stderr metric, 
by looking for another metric with the same `scorer_name` and with a `metric_name` containing the string "stderr".

## Score results 
```shell
agenteval score [OPTIONS] LOG_DIR
```
Compute scores for the results in `agenteval.json` and update the file with the computed scores.

## Publish scores
```shell
agenteval publish [OPTIONS] LOG_DIR
```
Upload the scored results to HuggingFace datasets.

# Administer the HuggingFace datasets
Prior to publishing scores, two HuggingFace datasets should be set up, one for full submissions and one for results files.

If you want to call `load_dataset()` on the results dataset (e.g., for populating a leaderboard), you probably want to explicitly tell HuggingFace about the schema and dataset structure (otherwise, HuggingFace may fail to propertly auto-convert to Parquet):
- *Schema:* Upload the [results schema](https://github.com/allenai/agent-eval/blob/main/dataset_infos.json) to the root of the results dataset.
- *Dataset structure:*  Specify the `configs` attribute in the YAML metadata block at the top of the `README.md` file at the root of the results dataset. For example, see the [sample metadata block](sample-config-dataset-structure.yml) for the [sample config](sample-config.yml). Using `agenteval publish` will automatically add the corresponding config name and split to the YAML metadata if it is missing.

# Development

See [Development.md](Development.md) for development instructions.
