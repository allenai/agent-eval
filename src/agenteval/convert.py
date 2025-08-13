import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import SuiteConfig, Task
from .leaderboard.models import LeaderboardSubmission, Readme
from .leaderboard.schema_generator import (
    check_submissions_against_readme,
    load_dataset_features,
)
from .score import TaskResult

logger = logging.getLogger(__name__)


# src HF config -> target HF config -> split -> original name -> target name
TASK_NAME_ALIASES = {
    "1.0.0-dev1": {
        "1.0.0": {
            "test": {
                "paper_finder_test": "PaperFindingBench_test",
                "paper_finder_litqa2_test": "LitQA2_FullText_Search_test",
                "sqa_test": "ScholarQA_CS2_test",
                "arxivdigestables_test": "ArxivDIGESTables_Clean_test",
                "litqa2_test": "LitQA2_FullText_test",
                "discoverybench_test": "DiscoveryBench_test",
                "core_bench_test": "CORE_Bench_Hard_test",
                "ds1000_test": "DS_1000_test",
                "e2e_discovery_test": "E2E_Bench_test",
                "e2e_discovery_hard_test": "E2E_Bench_Hard_test",
                "super_test": "SUPER_Expert_test",
            },
            "validation": {
                "arxivdigestables_validation": "ArxivDIGESTables_Clean_validation",
                "sqa_dev": "ScholarQA_CS2_validation",
                "litqa2_validation": "LitQA2_FullText_validation",
                "paper_finder_validation": "PaperFindingBench_validation",
                "paper_finder_litqa2_validation": "LitQA2_FullText_Search_validation",
                "discoverybench_validation": "DiscoveryBench_validation",
                "core_bench_validation": "CORE_Bench_Hard_validation",
                "ds1000_validation": "DS_1000_validation",
                "e2e_discovery_validation": "E2E_Bench_validation",
                "e2e_discovery_hard_validation": "E2E_Bench_Hard_validation",
                "super_validation": "SUPER_Expert_validation",
            },
        }
    }
}


@dataclass
class WithinRepoPath:
    hf_config: str
    split: str
    filename: str

    @staticmethod
    def from_path(path: str, sep: str = "/"):
        hf_config, split, filename = path.split(sep)
        return WithinRepoPath(
            hf_config=hf_config,
            split=split,
            filename=filename,
        )

    def to_path(path: str, sep: str = "/"):
        return sep.join([self.hf_config, self.split, self.filename])

    def with_different_hf_config(self, new_hf_config: str):
        return WithinRepoPath(
            hf_config=new_hf_config,
            split=self.split,
            filename=self.filename,
        )


@dataclass
class LeaderboardSubmissionAndSubmissionPath:
    lb_submission: LeaderboardSubmission
    within_repo_path: WithinRepoPath

    def __post_init__(self):
        self.check()

    def check(self):
        hf_config_from_submission = self.lb_submission.suite_config.version
        hf_config_from_path = self.within_repo_path.hf_config
        if hf_config_from_submission != hf_config_from_path:
            logger.warning(
                f"For result {self.within_repo_path.to_path()}: HF config from path {hf_config_from_path} is different from HF config from submission {hf_config_from_submission}."
            )

        split_from_submission = self.lb_submission.split
        split_from_path = self.within_repo_path.split
        if split_from_submission != split_from_path:
            # I'm not sure this should ever happen. So error if it does.
            raise Exception(
                f"For result {self.within_repo_path.to_path()}: split from path {split_from_path} is different from split from submission: {split_from_submission}."
            )

    def hf_config(self):
        # use the one from submission if we have different ones
        return self.lb_submission.suite_config.version

    def split(self):
        # we shouldn't even get here if the splits are different so doesn't matter which one we pick
        return self.lb_submission.split


def convert_one_task_result(
    result: TaskResult,
    split: str,
    src_suite_config: SuiteConfig,
    src_tasks_by_name: Dict[str, Task],
    target_suite_config: SuiteConfig,
    target_tasks_by_name: Dict[str, Task],
):
    changed_something = False

    src_hf_config_version = src_suite_config.version
    target_hf_config_version = target_suite_config.version

    original_task_name = result.task_name
    if original_task_name in TASK_NAME_ALIASES.get(src_hf_config_version, {}).get(
        target_hf_config_version, {}
    ).get(split, {}):
        new_task_name = TASK_NAME_ALIASES[src_hf_config_version][
            target_hf_config_version
        ][split][original_task_name]
        result.task_name = new_task_name

    final_task_name = result.task_name
    if original_task_name != final_task_name:
        changed_something = True

    if final_task_name not in target_tasks_by_name:
        # TODO: probably error here?
        logger.warning(f"Unknown final task name {final_task_name}")
    else:
        expected_primary_metric_name = target_tasks_by_name[
            final_task_name
        ].primary_metric
        if expected_primary_metric_name not in result.available_metrics():
            logger.warning(
                f"Expected {expected_primary_metric_name} as the primary metric for task {final_task_name}, but don't have it."
            )

    return changed_something


def convert_task_results(
    task_results: List[TaskResult],
    split: str,
    src_suite_config: SuiteConfig,
    target_suite_config: SuiteConfig,
    target_tasks_by_name: Dict[str, Task],
):
    try:
        src_tasks_by_name = src_suite_config.get_tasks_by_name(split)
    except ValueError as exc:
        logger.warning(
            f"Issue getting tasks for split {split} from the source suite config."
        )
        # TODO: some error
        return

    # changes happen in place
    changed_something = False
    for result in task_results:
        # you can get convert_task_results from src_suite_config, but pass
        # it in to avoid recomputing it again and again
        changed_something = changed_something or convert_one_task_result(
            result=result,
            split=split,
            src_suite_config=src_suite_config,
            src_tasks_by_name=src_tasks_by_name,
            target_suite_config=target_suite_config,
            target_tasks_by_name=target_tasks_by_name,
        )

    return changed_something


def convert_lb_submission_with_path(
    lb_submission_with_path: LeaderboardSubmissionAndSubmissionPath,
    target_suite_config: SuiteConfig,
    target_tasks_by_name: Dict[str, Task],
) -> bool:
    changed_something = False

    # changes are made in place
    src_suite_config = lb_submission_with_path.lb_submission.suite_config
    changed_something = changed_something or convert_task_results(
        task_results=lb_submission_with_path.lb_submission.results,
        split=lb_submission_with_path.split(),
        src_suite_config=src_suite_config,
        target_suite_config=target_suite_config,
        target_tasks_by_name=target_tasks_by_name,
    )
    if src_suite_config != target_suite_config:
        lb_submission_with_path.lb_submission.suite_config = target_suite_config
        changed_something = True

    return changed_something


def convert_result_files(
    src_repo_id: str,
    src_result_paths: List[str],
    target_repo_id: str,
    target_suite_config: SuiteConfig,
):
    target_split_to_task_name_to_task = {}
    for split in target_suite_config.splits:
        target_split_to_task_name_to_task[split.name] = (
            target_suite_config.get_tasks_by_name(split.name)
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        from huggingface_hub import HfApi, snapshot_download

        src_results_root_dir = os.path.join(temp_dir, "current")
        target_results_root_dir = os.path.join(temp_dir, "updated")

        snapshot_download(
            repo_id=src_repo_id,
            repo_type="dataset",
            allow_patterns=src_result_paths,
            local_dir=src_results_root_dir,
        )

        changed_anything = False
        for src_result_path_within_repo in src_result_paths:
            logger.info(f"Looking at source path {src_result_path_within_repo}")

            src_result_local_path = os.path.join(
                src_results_root_dir, src_result_path_within_repo
            )

            src_structured_path = WithinRepoPath.from_path(src_result_path_within_repo)
            with open(src_result_local_path) as f_src:
                lb_submission = LeaderboardSubmission.model_validate(json.load(f_src))
                lb_submission_with_path = LeaderboardSubmissionAndSubmissionPath(
                    lb_submission=lb_submission,
                    within_repo_path=src_structured_path,
                )

                # we can get target_tasks_by_name from target_suite_config,
                # but pass it in to avoid recomputing it over and over
                changed_this_thing = convert_lb_submission_with_path(
                    lb_submission_with_path=lb_submission_with_path,
                    target_suite_config=target_suite_config,
                    target_tasks_by_name=target_split_to_task_name_to_task[
                        lb_submission_with_path.split()
                    ],
                )
                changed_anything = changed_anything or changed_this_thing

                if changed_this_thing:
                    target_structured_path = lb_submission_with_path.within_repo_path.with_different_hf_config(
                        target_suite_config.version
                    )
                    logger.info(
                        f"Writing updated version of {src_result_path_within_repo} to local file under {target_structured_path.to_path()}"
                    )
                    target_results_inner_dir = os.path.join(
                        target_results_root_dir,
                        target_structured_path.hf_config,
                        target_structured_path.split,
                    )
                    os.makedirs(target_results_inner_dir, exist_ok=True)
                    with open(
                        os.path.join(
                            target_results_inner_dir, target_structured_path.filename
                        ),
                        "w",
                        encoding="utf-8",
                    ) as f_target:
                        f_target.write(lb_submission.model_dump_json(indent=None))

        if changed_anything:
            # Validate the config with the schema in HF
            # readme = Readme.download_and_parse(repo_id)
            # check_submissions_against_readme(
            #     lb_submissions=lb_submissions, readme=readme, repo_id=repo_id
            # )
            logger.info(f"Uploading converted results to {target_repo_id}...")
            # hf_api = HfApi()
            # hf_api.upload_folder(
            #     folder_path=target_results_root_dir,
            #     path_in_repo="",
            #     repo_id=target_repo_id,
            #     repo_type="dataset",
            # )
