import json
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from .config import SuiteConfig
from .leaderboard.models import LeaderboardSubmission


@dataclass
class LeaderboardSubmissionAndSubmissionPath:
    lb_submission: LeaderboardSubmission
    within_repo_path: str


def organize_results(lb_submissions: List[LeaderboardSubmissionAndSubmissionPath]):
    # HF config -> split -> list of submissions
    organized: Dict[str, Dict[str, List[LeaderboardSubmissionAndSubmissionPath]]] = {}
    for submission_with_path in lb_submissions:
        submission = submission_with_path.lb_submission
        hf_config = submission.suite_config.version
        if hf_config not in organized:
            organized[hf_config] = {}
        split = submission.split
        if split not in organized[hf_config]:
            organized[hf_config][split] = []
        organized[hf_config][split].append(submission_with_path)
    return organized


def convert_results(
    target_suite_config: SuiteConfig,
    target_repo_id: str,
    src_repo_id: str,
    src_result_paths: List[str],
):
    with tempfile.TemporaryDirectory() as temp_dir:
        from huggingface_hub import snapshot_download

        src_results_root_dir = os.path.join(temp_dir, "current")

        snapshot_download(
            repo_id=src_repo_id,
            repo_type="dataset",
            allow_patterns=src_result_paths,
            local_dir=src_results_root_dir,
        )

        all_lb_submissions_with_paths = []

        for src_result_path_within_repo in src_result_paths:
            print(f"Looking at path {src_result_path_within_repo}")
            src_result_local_path = os.path.join(
                src_results_root_dir, src_result_path_within_repo
            )
            with open(src_result_local_path) as f_src:
                lb_submission = LeaderboardSubmission.model_validate(json.load(f_src))
                lb_submission_with_path = LeaderboardSubmissionAndSubmissionPath(
                    lb_submission=lb_submission,
                    within_repo_path=src_result_path_within_repo,
                )
                all_lb_submissions_with_paths.append(lb_submission_with_path)

        organized_submissions_with_paths = organize_results(
            all_lb_submissions_with_paths
        )
        for hf_config in organized_submissions_with_paths:
            print(f"hf config {hf_config}")
            for split in organized_submissions_with_paths[hf_config]:
                print(f"split {split}")
