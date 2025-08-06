# agent-eval

A utility for evaluating agents on a suite of [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai)-formatted evals, with the following primary benefits:
1. Task suite specifications as config.
2. Extracts the token usage of the agent from log files, and computes cost using `litellm`.
3. Submits task suite results to a leaderboard, with submission metadata and easy upload to a HuggingFace repo for distribution of scores and logs.

# Installation

To install from pypi, use `pip install agent-eval`.

For leaderboard extras, use `pip install agent-eval[leaderboard]`.

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

### Weighted Macro Averaging with Tags

Tasks can be grouped using `tags` for computing summary statistics. The tags support weighted macro averaging, allowing you to assign different weights to tasks within a tag group.

Tags are specified as simple strings on tasks. To adjust weights for specific tag-task combinations, use the `macro_average_weight_adjustments` field at the split level. Tasks not specified in the adjustments default to a weight of 1.0.

See [sample-config.yml](sample-config.yml) for an example of the tag and weight adjustment format.

## Score results 
```shell
agenteval score [OPTIONS] LOG_DIR
```
Compute scores for the results in `agenteval.json` and update the file with the computed scores.

## Publish scores to leaderboard
```shell
agenteval lb publish [OPTIONS] LOG_DIR
```
Upload the scored results to HuggingFace datasets.

## View leaderboard scores
```shell
agenteval lb view [OPTIONS]
```
View results from the leaderboard.

# Administer the leaderboard
Prior to publishing scores, two HuggingFace datasets should be set up, one for full submissions and one for results files.

If you want to call `load_dataset()` on the results dataset (e.g., for populating a leaderboard), you probably want to explicitly tell HuggingFace about the schema and dataset structure (otherwise, HuggingFace may fail to propertly auto-convert to Parquet).
This is done by updating the `configs` attribute in the YAML metadata block at the top of the `README.md` file at the root of the results dataset (the metadata block is identified by lines with just `---` above and below it).
This attribute should contain a list of configs, each of which specifies the schema (under the `features` key) and dataset structure (under the `data_files` key).
See [sample-config-hf-readme-metadata.yml](sample-config-hf-readme-metadata.yml) for a sample metadata block corresponding to [sample-comfig.yml](sample-config.yml) (note that the metadata references the [raw schema data](src/agenteval/leaderboard/dataset_features.yml), which must be copied).

To facilitate initializing new configs, `agenteval lb publish` will automatically add this metadata if it is missing.

# Development

See [Development.md](Development.md) for development instructions.
