import logging
import re
from io import BytesIO

import yaml
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from .models import EvalResult

logger = logging.getLogger(__name__)


def ensure_readme_configs(
    api: HfApi,
    repo_id: str,
    config_name: str,
    split_globs: dict[str, str],
):
    """
    Ensure the README.md file in the specified Hugging Face dataset
    repository identifies the config and split paths, using provided API.

    If the README.md file does not exist, it will be created.
    """
    # use provided api for repo operations

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
    config = config_lookup.setdefault(
        config_name, {"config_name": config_name, "data_files": []}
    )
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


def upload_folder_to_hf(
    api: HfApi,
    folder_path: str,
    repo_id: str,
    config_name: str,
    split: str,
    submission_name: str,
    create_repo_if_missing: bool = False,
    create_private: bool = True,
) -> tuple[str, bool]:
    """
    Upload a local folder of logs to a Hugging Face dataset
    repository and return the hf:// URL.
    Returns a tuple of (url, created_repo_flag).
    """
    created = False
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        # only handle missing repo, let other HF errors propagate
        if create_repo_if_missing:
            logger.info(
                f"Repo '{repo_id}' not found. "
                f"Creating {'private' if create_private else 'public'} dataset..."
            )
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=create_private,
                exist_ok=False,
            )
            logger.info(f"Initialized repo '{repo_id}' as dataset.")
            created = True
        else:
            raise
    api.upload_folder(
        folder_path=folder_path,
        path_in_repo=f"{config_name}/{split}/{submission_name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    return f"hf://datasets/{repo_id}/{config_name}/{split}/{submission_name}", created


def upload_summary_to_hf(
    api: HfApi,
    eval_result: EvalResult,
    repo_id: str,
    config_name: str,
    split: str,
    submission_name: str,
    create_repo_if_missing: bool = False,
    create_private: bool = True,
) -> tuple[str, bool]:
    """
    Upload an EvalResult JSON summary to a Hugging Face
    dataset repository, update README, and return the hf:// URL.
    Returns a tuple of (url, created_repo_flag).
    """
    created = False
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        # only handle missing repo, let other HF errors propagate
        if create_repo_if_missing:
            logger.info(
                f"Repo '{repo_id}' not found. "
                f"Creating {'private' if create_private else 'public'} dataset..."
            )
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=create_private,
                exist_ok=False,
            )
            logger.info(f"Initialized repo '{repo_id}' as dataset.")
            created = True
        else:
            raise
    summary_bytes = BytesIO(eval_result.dump_json_bytes())
    api.upload_file(
        path_or_fileobj=summary_bytes,
        path_in_repo=f"{config_name}/{split}/{submission_name}.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    # update README with new config/split via provided API
    ensure_readme_configs(
        api,
        repo_id=repo_id,
        config_name=config_name,
        split_globs={split: f"{config_name}/{split}/*.json"},
    )
    return (
        f"hf://datasets/{repo_id}/{config_name}/{split}/" f"{submission_name}.json"
    ), created
