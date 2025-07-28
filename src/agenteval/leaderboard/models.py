from pydantic import BaseModel, Field

from ..models import SubmissionMetadata, SuiteConfig, TaskResult


class LeaderboardSubmission(BaseModel):
    suite_config: SuiteConfig
    """Task configuration for the results."""

    split: str
    """Split used for the results."""

    results: list[TaskResult] | None = None
    submission: SubmissionMetadata = Field(default_factory=SubmissionMetadata)
