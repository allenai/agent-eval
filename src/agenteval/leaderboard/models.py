from pydantic import BaseModel, Field

from ..models import EvalConfig, SubmissionMetadata, SuiteConfig, TaskResult


class LeaderboardSubmission(BaseModel):
    suite_config: SuiteConfig
    """Task configuration for the results."""

    split: str
    """Split used for the results."""

    results: list[TaskResult] | None = None
    submission: SubmissionMetadata = Field(default_factory=SubmissionMetadata)

    def to_eval_config(self) -> EvalConfig:
        return EvalConfig(suite_config=self.suite_config, split=self.split)
