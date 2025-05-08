import logging
import re
from io import BytesIO

import yaml
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from .models import EvalResult
from .schema_generator import load_dataset_features

logger = logging.getLogger(__name__)


def ensure_readme_configs(
    api: HfApi,
    repo_id: str,
    config_name: str,
    split_globs: dict[str, str],
):
    """
    Ensure the README.md file in the specified Hugging Face dataset
    repository specifies the config with the given split paths and the
    latest features schema supplied by the agenteval package.

    This structured data is necessary for HuggingFace to parse the repository
    into dataset splits and auto-convert to parquet.

    The README.md file is expected to contain a YAML block at the
    beginning, which is parsed and updated with the new config/split
    information. The YAML block should have the following structure:
    ```yaml
    configs:
      - config_name: <config_name>
        data_files:
          - split: <split_name>
            path: <path_glob>
        features:
          <features>
    ```

    If the README.md file does not exist, it will be created.
    """
    try:
        readme_path = api.hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="dataset",
        )
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            content = ""
        else:
            raise

    # parse the YAML block from the README
    match = re.match(r"(?s)^---\n(.*?)\n---\n(.*)", content)
    if match:
        yaml_block, rest = match.groups()
        parsed_yaml = yaml.safe_load(yaml_block) or {}
    else:
        parsed_yaml = {}
        rest = content

    parsed_yaml.setdefault("configs", [])
    config_list = parsed_yaml["configs"]

    config_lookup = {c["config_name"]: c for c in config_list}
    if config_name in config_lookup:
        config = config_lookup[config_name]
    else:
        config = {
            "config_name": config_name,
            "data_files": [],
            "features": load_dataset_features()._to_yaml_list(),
        }
        config_lookup[config_name] = config
    split_lookup = {s["split"]: s for s in config["data_files"]}

    for split, path in split_globs.items():
        if split not in split_lookup:
            config["data_files"].append({"split": split, "path": path})
        else:
            existing = split_lookup[split]["path"]
            existing_value = yaml.safe_load(str(existing))
            new_value = yaml.safe_load(str(path))

            if isinstance(existing_value, list):
                if new_value not in existing_value:
                    raise ValueError(
                        f"Path for split '{split}' already set to {existing_value}, cannot update to '{path}'."
                    )
            elif isinstance(existing_value, str):
                if existing_value != new_value:
                    raise ValueError(
                        f"Path for split '{split}' is already set to '{existing_value}', cannot update to '{new_value}'."
                    )
            else:
                raise TypeError(
                    f"Unexpected path type for split '{split}': {type(existing_value)}"
                )

    parsed_yaml["configs"] = list(config_lookup.values())
    updated_yaml = yaml.dump(parsed_yaml, sort_keys=False).strip()
    new_readme = f"---\n{updated_yaml}\n---\n{rest.lstrip()}"

    if new_readme.strip() != content.strip():
        api.upload_file(
            path_or_fileobj=BytesIO(new_readme.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )


def _validate_path_component(component: str, desc: str):
    # allow letters, digits, underscore, dash, and literal dot
    if not re.match(r"^[A-Za-z0-9._-]+$", component):
        raise ValueError(f"Invalid {desc}: {component}")


def sanitize_path_component(component: str) -> str:
    # replace any character not alphanumeric, dot, dash, or underscore with underscore
    return re.sub(r"[^A-Za-z0-9._-]", "_", component)


def upload_folder_to_hf(
    api: HfApi,
    folder_path: str,
    repo_id: str,
    config_name: str,
    split: str,
    submission_name: str,
) -> str:
    """Upload a folder to a HuggingFace dataset repository."""
    _validate_path_component(config_name, "config_name")
    _validate_path_component(split, "split")
    _validate_path_component(submission_name, "submission_name")
    api.upload_folder(
        folder_path=folder_path,
        path_in_repo=f"{config_name}/{split}/{submission_name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    return f"hf://datasets/{repo_id}/{config_name}/{split}/{submission_name}"


def upload_summary_to_hf(
    api: HfApi,
    eval_result: EvalResult,
    repo_id: str,
    config_name: str,
    split: str,
    submission_name: str,
) -> str:
    """Upload a summary of the evaluation result to a HuggingFace dataset repository."""
    _validate_path_component(config_name, "config_name")
    _validate_path_component(split, "split")
    _validate_path_component(submission_name, "submission_name")
    summary_bytes = BytesIO(eval_result.dump_json_bytes())
    api.upload_file(
        path_or_fileobj=summary_bytes,
        path_in_repo=f"{config_name}/{split}/{submission_name}.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    ensure_readme_configs(
        api,
        repo_id=repo_id,
        config_name=config_name,
        split_globs={split: f"{config_name}/{split}/*.json"},
    )
    return f"hf://datasets/{repo_id}/{config_name}/{split}/{submission_name}.json"
