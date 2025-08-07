import logging
import re
from collections import defaultdict

from huggingface_hub import HfApi
from inspect_ai.model import ModelUsage

from ..log import ModelUsageWithName
from .models import LeaderboardSubmission

logger = logging.getLogger(__name__)


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


def compress_model_usages(eval_result: LeaderboardSubmission):
    """
    Reduce the size of model usages by compressing to aggregate token
    counts for each token type, model, and task problem
    """
    if not eval_result.results:
        return eval_result

    compressed_results = []
    for task_result in eval_result.results:
        # replace list[None] with None if any costs are None
        model_costs = task_result.model_costs
        if model_costs is not None and any(cost is None for cost in model_costs):
            model_costs = None

        # Create a new TaskResult with compressed model_usages
        compressed_task_result = task_result.model_copy(
            update={
                "model_costs": model_costs,
                "model_usages": None if task_result.model_usages is None else [],
            }
        )

        if task_result.model_usages and compressed_task_result.model_usages is not None:
            for problem_usages in task_result.model_usages:
                compressed_problem_usages = compress_usages_by_problem(problem_usages)
                compressed_task_result.model_usages.append(compressed_problem_usages)

        compressed_results.append(compressed_task_result)

    # Create new EvalResult with compressed results
    compressed_eval_result = LeaderboardSubmission(
        **eval_result.model_dump(exclude={"results"}), results=compressed_results
    )

    return compressed_eval_result


def compress_usages_by_problem(usages_by_problem: list[ModelUsageWithName]):
    """
    Compress a list of ModelUsageWithName objects by aggregating usage for the same model.
    """
    model_usage_map: dict[str, ModelUsage] = defaultdict(lambda: ModelUsage())

    for usage_with_name in usages_by_problem:
        model_name = usage_with_name.model
        model_usage_map[model_name] += usage_with_name.usage

    return [
        ModelUsageWithName(model=model_name, usage=usage)
        for model_name, usage in model_usage_map.items()
    ]
