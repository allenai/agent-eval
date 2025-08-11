"""
Tests for compute_summary_statistics function, specifically focusing on the stderr extraction bug fix.
"""

from unittest.mock import Mock

import pytest

from agenteval.config import Split, SuiteConfig, Task
from agenteval.score import Metric, TaskResult
from agenteval.summary import SummaryStats, compute_summary_statistics


@pytest.fixture
def core_bench_task():
    """Create a realistic core_bench task configuration."""
    return Task(
        name="core_bench_test",
        path="astabench/evals/inspect_eval_wrappers/core_bench:core_bench_test",
        primary_metric="score_with_stderr/accuracy",
        tags=["code"],
    )


@pytest.fixture
def other_task():
    """Create another task with different metric structure."""
    return Task(
        name="ds1000_test",
        path="astabench/evals/inspect_eval_wrappers/ds1000:ds1000_test",
        primary_metric="ds1000_scorer/accuracy",
        tags=["code"],
    )


@pytest.fixture
def suite_config_with_core_bench(core_bench_task, other_task):
    """Create a suite config with core_bench and other tasks."""
    split = Split(name="test", tasks=[core_bench_task, other_task])

    mock_config = Mock(spec=SuiteConfig)
    mock_config.get_tasks.return_value = [core_bench_task, other_task]
    mock_config.get_split.return_value = split

    return mock_config


@pytest.fixture
def core_bench_metrics():
    """Create realistic core_bench metrics including the problematic ones."""
    return [
        Metric(name="score_with_stderr/accuracy", value=0.324),
        Metric(name="score_with_stderr/stderr", value=0.078),
        # Add some other metrics that also contain "stderr" to test disambiguation
        Metric(name="score_rubric/stderr", value=0.045),
        Metric(name="output_match/stderr", value=0.023),
    ]


@pytest.fixture
def other_task_metrics():
    """Create metrics for the other task."""
    return [
        Metric(name="ds1000_scorer/accuracy", value=0.75),
        Metric(name="ds1000_scorer/stderr", value=0.015),
    ]


@pytest.fixture
def core_bench_task_result(core_bench_metrics):
    """Create a TaskResult for core_bench."""
    return TaskResult(
        task_name="core_bench_test",
        metrics=core_bench_metrics,
        model_costs=[1.0, 1.2, 0.8],  # Some sample costs
    )


@pytest.fixture
def other_task_result(other_task_metrics):
    """Create a TaskResult for the other task."""
    return TaskResult(
        task_name="ds1000_test", metrics=other_task_metrics, model_costs=[0.5, 0.6]
    )


class TestComputeSummaryStatistics:
    """Test suite for compute_summary_statistics function."""

    def test_stderr_extraction_bug_fix(
        self, suite_config_with_core_bench, core_bench_task_result, other_task_result
    ):
        """Test that stderr is correctly extracted and not equal to score (the main bug fix)."""
        results = [core_bench_task_result, other_task_result]

        summary_stats = compute_summary_statistics(
            suite_config_with_core_bench, "test", results
        )

        # The key assertion: stderr should NOT equal score (this was the bug)
        core_bench_stat = summary_stats.stats["task/core_bench_test"]
        assert core_bench_stat.score == 0.324
        assert core_bench_stat.score_stderr == 0.078
        assert (
            core_bench_stat.score != core_bench_stat.score_stderr
        ), "REGRESSION: stderr equals score, indicating the bug has returned!"

        # Test the other task too
        other_stat = summary_stats.stats["task/ds1000_test"]
        assert other_stat.score == 0.75
        assert other_stat.score_stderr == 0.015
        assert other_stat.score != other_stat.score_stderr

    def test_stderr_metric_disambiguation(
        self, suite_config_with_core_bench, core_bench_task_result, other_task_result
    ):
        """Test that the correct stderr metric is selected when multiple stderr metrics exist."""
        results = [core_bench_task_result, other_task_result]

        summary_stats = compute_summary_statistics(
            suite_config_with_core_bench, "test", results
        )

        # Should select "score_with_stderr/stderr" (0.078) not "score_rubric/stderr" (0.045)
        core_bench_stat = summary_stats.stats["task/core_bench_test"]
        assert (
            core_bench_stat.score_stderr == 0.078
        ), "Should select the stderr metric matching the primary metric's scorer"

        # Should select "ds1000_scorer/stderr", not any other stderr metric
        other_stat = summary_stats.stats["task/ds1000_test"]
        assert other_stat.score_stderr == 0.015

    def test_missing_stderr_metric(self, suite_config_with_core_bench):
        """Test handling when stderr metric is missing."""
        # Create task result without stderr metric
        metrics_without_stderr = [
            Metric(name="score_with_stderr/accuracy", value=0.324),
            # Note: no "score_with_stderr/stderr" metric
        ]

        task_result = TaskResult(
            task_name="core_bench_test", metrics=metrics_without_stderr
        )

        results = [task_result]

        summary_stats = compute_summary_statistics(
            suite_config_with_core_bench, "test", results
        )

        stat = summary_stats.stats["task/core_bench_test"]
        assert stat.score == 0.324
        assert stat.score_stderr is None  # Should be None when stderr missing

    def test_missing_primary_metric(self, suite_config_with_core_bench):
        """Test handling when primary metric is missing."""
        # Create task result without primary metric
        metrics_without_primary = [
            Metric(name="score_with_stderr/stderr", value=0.078),
            # Note: no "score_with_stderr/accuracy" metric
        ]

        task_result = TaskResult(
            task_name="core_bench_test", metrics=metrics_without_primary
        )

        results = [task_result]

        summary_stats = compute_summary_statistics(
            suite_config_with_core_bench, "test", results
        )

        stat = summary_stats.stats["task/core_bench_test"]
        assert stat.score is None
        assert stat.score_stderr is None

    def test_missing_task_results(self, suite_config_with_core_bench):
        """Test handling when task results are completely missing."""
        results = []  # No results at all

        summary_stats = compute_summary_statistics(
            suite_config_with_core_bench, "test", results
        )

        # All stats should be None when no results provided
        core_bench_stat = summary_stats.stats["task/core_bench_test"]
        assert core_bench_stat.score is None
        assert core_bench_stat.score_stderr is None
        assert core_bench_stat.cost is None
        assert core_bench_stat.cost_stderr is None

    def test_complex_metric_names_with_multiple_slashes(self):
        """Test rpartition handling with complex metric names."""
        task = Task(
            name="complex_task",
            path="some/path",
            primary_metric="complex/scorer/with/slashes/accuracy",  # Multiple slashes
            tags=["test"],
        )

        metrics = [
            Metric(name="complex/scorer/with/slashes/accuracy", value=0.5),
            Metric(name="complex/scorer/with/slashes/stderr", value=0.1),
        ]

        task_result = TaskResult(task_name="complex_task", metrics=metrics)

        split = Split(name="test", tasks=[task])
        mock_config = Mock(spec=SuiteConfig)
        mock_config.get_tasks.return_value = [task]
        mock_config.get_split.return_value = split

        results = [task_result]

        summary_stats = compute_summary_statistics(mock_config, "test", results)

        stat = summary_stats.stats["task/complex_task"]
        assert stat.score == 0.5
        assert stat.score_stderr == 0.1

    def test_regression_old_vs_new_logic(self, core_bench_metrics):
        """Test that demonstrates how old logic would fail vs new logic succeeds."""

        # Simulate the old buggy logic
        def old_stderr_extraction(metrics, primary_metric):
            """Simulate the old buggy stderr extraction logic."""
            primary_prefix = f"{primary_metric.split('/')[0]}/"
            for m in metrics:
                if m.name.startswith(primary_prefix) and "stderr" in m.name:
                    return m.value  # Returns first match - this was the bug!
            return None

        # Simulate the new correct logic
        def new_stderr_extraction(metrics, primary_metric):
            """Simulate the new correct stderr extraction logic."""
            expected_stderr_name = f"{primary_metric.rpartition('/')[0]}/stderr"
            metrics_by_name = {m.name: m for m in metrics}
            stderr_metric = metrics_by_name.get(expected_stderr_name)
            return stderr_metric.value if stderr_metric else None

        primary_metric = "score_with_stderr/accuracy"

        # Old logic returns wrong value (0.324 - the accuracy score!)
        old_result = old_stderr_extraction(core_bench_metrics, primary_metric)
        assert old_result == 0.324  # Wrong! This is the score, not stderr

        # New logic returns correct value
        new_result = new_stderr_extraction(core_bench_metrics, primary_metric)
        assert new_result == 0.078  # Correct! This is the actual stderr

        # The bug: old_result == score, which was the original problem
        score = next(m.value for m in core_bench_metrics if m.name == primary_metric)
        assert old_result == score, "Old logic incorrectly returns score as stderr"
        assert (
            new_result != score
        ), "New logic correctly distinguishes stderr from score"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_results_list(self, suite_config_with_core_bench):
        """Test with empty results list."""
        summary_stats = compute_summary_statistics(
            suite_config_with_core_bench, "test", []
        )

        assert isinstance(summary_stats, SummaryStats)
        assert "task/core_bench_test" in summary_stats.stats

    def test_metric_name_construction_edge_cases(self):
        """Test metric name construction with various edge cases."""
        test_cases = [
            ("simple/accuracy", "simple/stderr"),
            ("complex/path/to/accuracy", "complex/path/to/stderr"),
            ("no_slash_accuracy", "/stderr"),  # Edge case
            ("trailing/slash/accuracy/", "trailing/slash/accuracy/stderr"),
        ]

        for primary_metric, expected_stderr_name in test_cases:
            actual = f"{primary_metric.rpartition('/')[0]}/stderr"
            assert actual == expected_stderr_name, f"Failed for {primary_metric}"
