from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from pydantic import BaseModel

from agenteval.leaderboard.models import InterventionPointer, LeaderboardSubmission
from agenteval.cli_utils import RepoPathsOfInterest


EDIT_INTERVENTION_KIND = "edit"
CONVERSION_INTERVENTION_KIND = "conversion"


@dataclass
class WithinRepoPath:
    hf_config: str
    split: str
    end: str

    def submission_name(self) -> str:
        suffix = ".json"
        if self.end.endswith(suffix):
            submission_name = self.end[:-len(suffix)]
        else:
            submission_name = self.end
        return submission_name

    @staticmethod
    def from_path(path: str, sep: str = "/"):
        [hf_config, split, end] = path.split(sep)
        return WithinRepoPath(
            hf_config=hf_config,
            split=split,
            end=end,
        )

    def to_path(self, sep: str = "/") -> str:
        return sep.join([self.hf_config, self.split, self.end])

    def with_different_hf_config(self, new_hf_config: str):
        return WithinRepoPath(
            hf_config=new_hf_config,
            split=self.split,
            end=self.end,
        )


class LbSubmissionWithDetails(BaseModel):
    lb_submission: LeaderboardSubmission
    submission_path: WithinRepoPath

    @staticmethod
    def mk(lb_submission: LeaderboardSubmission, submission_path: str):
        return LbSubmissionWithDetails(
            lb_submission=lb_submission,
            submission_path=WithinRepoPath.from_path(submission_path),
        )


class Intervention:
    def __init__(
        self,
        eligible: Callable[[LbSubmissionWithDetails], bool],
        transform: Callable[[LeaderboardSubmission], bool]
    ):
        self._eligible = eligible
        self._transform = transform

    def eligible(self, submission_with_details: LbSubmissionWithDetails) -> bool:
        return self._eligible(submission_with_details)

    def transform(self, lb_submission: LeaderboardSubmission) -> bool:
        return self._transform(lb_submission)


# intervention kind -> config name -> intervention name -> Intervention
INTERVENTIONS: dict[str, dict[str, dict[str, Intervention]]] = {
    EDIT_INTERVENTION_KIND: {},
    CONVERSION_INTERVENTION_KIND: {},
}


@dataclass
class RegistryPointer:
    registry: str
    name: str

    @staticmethod
    def from_str(a_str):
        sep = ":"
        [registry, name] = a_str.split(sep)
        return RegistryPointer(registry=registry, name=name)


class Registry:
    def __init__(self, registry_pointer_strs: list[str]):
        self.registry = {"agenteval": INTERVENTIONS}

        registry_pointers = [RegistryPointer.from_str(p) for p in registry_pointer_strs]
        for pointer in registry_pointers:
            assert pointer.registry not in self.registry, "Multiple registry entries with the same name."
            self.registry[pointer.registry] = import_module(pointer.name).INTERVENTIONS

    def find_intervention(self, intervention_kind: str, config_name: str, pointer: InterventionPointer):
        return self.registry.get(pointer.registry, {}).get(intervention_kind, {}).get(config_name, {}).get(pointer.name)


def edit_lb_submission(
    lb_submission_with_details: LbSubmissionWithDetails,
    intervention_pointers: list[InterventionPointer],
    registry: Registry,
) -> bool:
    edited_this_lb_submission = False
    for intervention_pointer in intervention_pointers:

        maybe_edit = registry.find_intervention(
            intervention_kind="edit",
            config_name=lb_submission_with_details.lb_submission.suite_config.version,
            pointer=intervention_pointer,
        )
        if (maybe_edit is not None) :
            if maybe_edit.eligible(lb_submission_with_details):
                applied_one_edit = maybe_edit.transform(lb_submission_with_details.lb_submission)
                if applied_one_edit:
                    lb_submission_with_details.lb_submission.add_edit(intervention_pointer)
                edited_this_lb_submission = edited_this_lb_submission or applied_one_edit
            else:
                print(f"{lb_submission_with_details.submission_path} is not eligble for the {intervention_pointer} edit.")

        else:
            print(f"Unable to find edit {intervention_pointer}.")

    return edited_this_lb_submission


def convert_lb_submission(
    lb_submission_with_details: LbSubmissionWithDetails,
    intervention_pointer: InterventionPointer,
    registry: Registry,
) -> bool:
    converted_this_lb_submission = False
    maybe_conversion = registry.find_intervention(
        intervention_kind="conversion",
        config_name=lb_submission_with_details.lb_submission.suite_config.version,
        pointer=intervention_pointer,
    )
    if (maybe_conversion is not None) :
        if maybe_conversion.eligible(lb_submission_with_details):
            converted_this_lb_submission = maybe_conversion.transform(lb_submission_with_details.lb_submission)
            if converted_this_lb_submission:
                lb_submission_with_details.lb_submission.add_conversion(intervention_pointer)
        else:
            print(f"{lb_submission_with_details.submission_path} is not eligble for the {intervention_pointer} conversion.")

    else:
        print(f"Unable to find conversion {intervention_pointer}.")

    return converted_this_lb_submission


def check_lb_submission_for_edit_eligibility(
    lb_submission_with_details: LbSubmissionWithDetails,
    intervention_pointers: list[InterventionPointer],
    registry: Registry,
) -> bool:
    for intervention_pointer in intervention_pointers:
        maybe_edit = registry.find_intervention(
            intervention_kind="edit",
            config_name=lb_submission_with_details.lb_submission.suite_config.version,
            pointer=intervention_pointer,
        )
        if (maybe_edit is not None) :
            if maybe_edit.eligible(lb_submission_with_details):
                print(f"{lb_submission_with_details.submission_path} is eligble for the {intervention_pointer} edit.")
        else:
            print(f"Unable to find edit {intervention_pointer}.")


def apply_existing_edits_to_result_files(
    repo_paths_of_interest: RepoPathsOfInterest,
    local_new_results_dir: str,
    local_existing_results_dir: str,
    registry: Registry,
):
    all_current_config_lb_submissions = []
    for current_config_result_path in repo_paths_of_interest.relative_paths:
        local_current_config_new_result_path = os.path.join(local_new_results_dir, current_config_result_path)
        with open(local_current_config_new_result_path) as f_current_config_new:
            lb_submission_current_config_new = LeaderboardSubmission.model_validate(json.load(f_current_config_new))
            lb_submission_with_details_current_config_new = LbSubmissionWithDetails.mk(lb_submission_current_config_new, current_config_result_path)

        local_current_config_existing_result_path = os.path.join(local_existing_results_dir, current_config_result_path)
        if os.path.isfile(local_current_config_existing_result_path):
            with open(local_current_config_existing_result_path) as f_current_config_existing:
                lb_submission_current_config_existing = LeaderboardSubmission.model_validate(json.load(f_current_config_existing))

            if lb_submission_current_config_existing.has_edits():
                edit_pointers = [e.pointer for e in lb_submission_current_config_existing.interventions.edits]

                # edits the lb submission in place
                edited_this_submission = edit_lb_submission(
                    lb_submission_with_details=lb_submission_with_details_current_config_new,
                    intervention_pointers=edit_pointers,
                    registry=registry,
                )
                if edited_this_submission:
                    with open(
                        local_current_config_new_result_path,
                        "w",
                        encoding="utf-8",
                    ) as f_current_config_new_post_edits:
                        f_current_config_new_post_edits.write(lb_submission_current_config_new.model_dump_json(indent=None))
                    print(lb_submission_current_config_new.model_dump_json(indent=2))

        # whether we applied edits or not, we still want to upload all these new result files
        all_current_config_lb_submissions.append(lb_submission_current_config_new)
    
    return all_current_config_lb_submissions


def apply_existing_conversions_to_result_files(
    repo_paths_of_interest: RepoPathsOfInterest,
    local_new_results_dir: str,
    local_existing_results_dir: str,
    local_converted_results_dir: str,
    registry=registry,
):
    new_config_paths_of_interest = []
    for current_config_result_path in repo_paths_of_interest.relative_paths:
        local_current_config_existing_result_path = os.path.join(local_existing_results_dir, current_config_result_path)
        if os.path.isfile(local_current_config_existing_result_path):
            with open(local_current_config_existing_result_path) as f_current_config_existing:
                lb_submission_current_config_existing = LeaderboardSubmission.model_validate(json.load(f_current_config_existing))

            if lb_submission_current_config_existing.has_conversions():
                conversion_pointers = [c.pointer for c in lb_submission_current_config_existing.interventions.edits]

                for conversion_pointer in conversion_pointers:
                    # reopen every time. don't reuse lb_submission_new_with_edits instances
                    # because they are converted to a different config in place
                    local_current_config_new_result_path = os.path.join(local_new_results_dir, current_config_result_path)
                    with open(local_current_config_new_result_path) as f_current_config_new_with_edits:
                        lb_submission_new_with_edits = LeaderboardSubmission.model_validate(json.load(f_current_config_new_with_edits))
                        lb_submission_with_details_new_with_edits = LbSubmissionWithDetails.mk(lb_submission_new_with_edits, current_config_result_path)

                    # edits the lb submission in place
                    converted_this_submission = convert_lb_submission(
                        lb_submission_with_details=lb_submission_with_details_new_with_edits,
                        intervention_pointer=conversion_pointer,
                        registry=registry,
                    )
                    if converted_this_submission::
                        new_config_result_path = lb_submission_with_details_new_with_edits.submission_path.with_different_hf_config(lb_submission_new_with_edits.suite_config.version).to_path()
                        new_config_paths_of_interest.append(new_config_result_path)

                        os.makedirs(
                            os.path.join(local_converted_results_dir, os.path.dirname(new_config_result_path)),
                            exist_ok=True,
                        )
                        with open(
                            os.path.join(local_converted_results_dir, new_config_result_path),
                            "w",
                            encoding="utf-8",
                        ) as f_new_config:
                            f_new_config.write(lb_submission_new_with_edits.model_dump_json(indent=None))
                
                        print(f"{current_config_result_path} -> {new_config_result_path}")
                        print(lb_submission_new_with_edits.model_dump_json(indent=2))

    return new_config_paths_of_interest
