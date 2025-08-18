from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from pydantic import BaseModel

from agenteval.leaderboard.models import InterventionPointer, LeaderboardSubmission


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
                print(f"{lb_submission_with_details.submission_path} is not eligble for the {intervention_pointer} change.")

        else:
            print(f"Unable to find intervention {intervention_pointer}.")

    return edited_this_lb_submission
