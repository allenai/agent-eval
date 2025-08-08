"""Tests for LeaderboardViewer.

Includes:
- Leaderboard webapp client requirements.
- Paper figures creation requirements.

"""

from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from agenteval.leaderboard.view import (
    LeaderboardViewer,
    _get_frontier_indices,
    _plot_combined_scatter,
)


def setup_mock_dataset(mock_load_dataset, split_name="test"):
    """Setup mock dataset for LeaderboardViewer initialization."""
    mock_dataset = Mock()
    mock_dataset.get.return_value = [
        {
            "suite_config": {
                "name": "test-suite",
                "splits": [
                    {
                        "name": split_name,
                        "tasks": [
                            {
                                "name": "task1",
                                "path": "t1",
                                "primary_metric": "score",
                                "tags": ["tag1"],
                            },
                            {
                                "name": "task2",
                                "path": "t2",
                                "primary_metric": "score",
                                "tags": ["tag1", "tag2"],
                            },
                        ],
                    }
                ],
            },
            "split": split_name,
            "results": [],
            "submission": {},
        }
    ]
    mock_load_dataset.return_value = mock_dataset
    return mock_dataset


@pytest.mark.leaderboard
class TestWebappLeaderboardViewerContract:
    """Test the minimal API contracts that the leaderboard webapp client uses."""

    @patch("agenteval.leaderboard.view.datasets.load_dataset")
    def test_initialization(self, mock_load_dataset):
        """Test LeaderboardViewer accepts parameters used by webapp.

        Webapp client: viewer = LeaderboardViewer(
            repo_id=RESULTS_DATASET,
            config=CONFIG_NAME,
            split=split,
            is_internal=IS_INTERNAL
        )
        """
        setup_mock_dataset(mock_load_dataset)

        # Pattern from webapp client - just verify it doesn't raise an error
        LeaderboardViewer(
            repo_id="allenai/asta-bench-results",
            config="1.0.0-dev1",
            split="test",
            is_internal=True,
        )

    @patch("agenteval.leaderboard.view.datasets.load_dataset")
    def test_tag_map_attribute_access(self, mock_load_dataset):
        """Test viewer.tag_map is accessible as webapp expects.

        Webapp client: create_pretty_tag_map(viewer.tag_map, ...)
        """
        setup_mock_dataset(mock_load_dataset)

        viewer = LeaderboardViewer("test-repo", "1.0.0", "test", False)

        # Webapp accesses viewer.tag_map directly
        assert hasattr(viewer, "tag_map")
        assert isinstance(viewer.tag_map, dict)
        # Based on mock data, should have these tags
        assert "tag1" in viewer.tag_map
        assert "tag2" in viewer.tag_map
        assert "task1" in viewer.tag_map["tag1"]
        assert "task2" in viewer.tag_map["tag1"]

    @patch("agenteval.leaderboard.view.datasets.load_dataset")
    def test_load_method_returns_tuple(self, mock_load_dataset):
        """Test _load() returns (DataFrame, dict) as webapp expects.

        Webapp client: raw_df, _ = viewer_or_data._load()
        """
        setup_mock_dataset(mock_load_dataset)

        with patch("agenteval.leaderboard.view._get_dataframe") as mock_get_df:
            mock_get_df.return_value = pd.DataFrame({"col": [1, 2]})

            viewer = LeaderboardViewer("test", "1.0.0", "test", False)

            # Webapp calls _load() with no parameters
            result = viewer._load()

            # Must return tuple of (DataFrame, dict)
            assert isinstance(result, tuple)
            assert len(result) == 2
            df, tag_map = result
            assert isinstance(df, pd.DataFrame)
            assert isinstance(tag_map, dict)

    @patch("agenteval.leaderboard.view.datasets.load_dataset")
    def test_webapp_expected_column_names(self, mock_load_dataset):
        """Test that the DataFrame has columns with names expected by the webapp."""
        # Webapp expects these exact column names
        webapp_expected_columns = [
            "id",
            "Agent",
            "Agent description",
            "User/organization",
            "Submission date",
            "Overall",
            "Overall cost",
            "Logs",
            "Openness",
            "Agent tooling",
            "LLM base",
        ]

        # Reuse setup_mock_dataset but extend it with more complete data
        mock_dataset = setup_mock_dataset(mock_load_dataset)

        # Update the mock to have complete submission data with results
        mock_dataset.get.return_value[0].update(
            {
                "split": "test",
                "submission": {
                    "agent_name": "Test Agent Name",
                    "agent_description": "Test Description",
                    "username": "test_user",
                    "submit_time": "2024-01-01T00:00:00Z",
                    "openness": "open",
                    "tool_usage": "basic",
                    "logs_url": "http://logs",
                    "logs_url_public": "http://logs",
                },
                "results": [
                    {
                        "task_name": "task1",
                        "metrics": [
                            {"name": "score", "value": 0.5},
                            {"name": "cost", "value": 10.0},
                        ],
                        "model_usages": [
                            [{"model": "gpt-4", "usage": {"total_tokens": 100}}]
                        ],
                    }
                ],
            }
        )

        viewer = LeaderboardViewer("test-repo", "1.0.0", "test", False)
        df, _ = viewer._load()

        # Check that all expected columns exist
        for expected_col in webapp_expected_columns:
            assert (
                expected_col in df.columns
            ), f"Expected column '{expected_col}' not found. Available: {list(df.columns)}"


@pytest.mark.leaderboard
class TestPaperWorkflowFunctionality:
    """Test core functionality used in paper_plots.sh workflow."""

    @pytest.fixture
    def paper_mock_data(self):
        """Create mock dataframe that mirrors paper data structure."""
        return pd.DataFrame(
            {
                "display_name": [
                    "ReAct (claude-sonnet-4)",
                    "ReAct (o3)",
                    "ReAct (gpt-4.1)",
                ],
                "overall/score": [0.378, 0.364, 0.283],
                "overall/cost": [0.406, 0.153, 0.128],
                "tag/lit/score": [0.472, 0.477, 0.421],
                "tag/lit/cost": [0.153, 0.217, 0.141],
            }
        )

    def test_paper_essential_features(self, paper_mock_data):
        """Test the essential features used in paper_plots.sh in one comprehensive test."""
        scatter_pairs = [
            ("overall/score", "overall/cost"),
            ("tag/lit/score", "tag/lit/cost"),
        ]

        # Test all key paper features together
        fig = _plot_combined_scatter(
            paper_mock_data,
            scatter_pairs=scatter_pairs,
            agent_col="display_name",
            use_log_scale=True,  # Key paper requirement
            figure_width=6.5,  # Paper standard width
            subplot_height=1.5,  # Paper height control
            legend_max_width=30,  # Paper legend wrapping
            subplot_spacing=0.2,  # Paper spacing
        )

        # Verify key requirements
        assert fig.get_figwidth() == 6.5, "Should use paper standard width"
        assert len(fig.axes) == 2, "Should create multiple subplots"
        assert fig.axes[0].get_xscale() == "log", "Should use log scale"

        # Verify legend exists (may be on any axis or figure-level)
        has_legend = any(ax.get_legend() is not None for ax in fig.axes)
        assert has_legend or fig.legends, "Should have legend somewhere"

        plt.close(fig)

    def test_cost_fallback_and_frontier(self, paper_mock_data):
        """Test cost fallback positioning and frontier calculation."""
        # Test frontier calculation works
        frontier_indices = _get_frontier_indices(
            paper_mock_data, "overall/cost", "overall/score"
        )
        assert len(frontier_indices) > 0, "Should find frontier points"

        # Test cost fallback with missing data
        fallback_data = paper_mock_data.copy()
        fallback_data.loc[len(fallback_data)] = {
            "display_name": "No-Cost Agent",
            "overall/score": 0.5,
            "overall/cost": None,  # Missing cost
            "tag/lit/score": 0.5,
            "tag/lit/cost": None,
        }

        fig = _plot_combined_scatter(
            fallback_data,
            scatter_pairs=[("overall/score", "overall/cost")],
            agent_col="display_name",
            use_cost_fallback=True,
            figure_width=6.5,
        )

        # Should handle missing cost data without errors
        assert fig is not None
        plt.close(fig)

    def test_figure_dimensions_scaling(self, paper_mock_data):
        """Test that figure dimensions scale correctly for different scenarios."""
        # Test single plot
        fig_single = _plot_combined_scatter(
            paper_mock_data,
            scatter_pairs=[("overall/score", "overall/cost")],
            agent_col="display_name",
            figure_width=6.5,
            subplot_height=1.5,
        )

        assert fig_single.get_figwidth() == 6.5
        assert abs(fig_single.get_figheight() - 1.5) < 0.01
        plt.close(fig_single)

        # Test multiple plots
        scatter_pairs = [
            ("overall/score", "overall/cost"),
            ("tag/lit/score", "tag/lit/cost"),
        ] * 2  # 4 plots total

        fig_multi = _plot_combined_scatter(
            paper_mock_data,
            scatter_pairs=scatter_pairs,
            agent_col="display_name",
            figure_width=6.5,
        )

        assert fig_multi.get_figwidth() == 6.5
        assert fig_multi.get_figheight() > 3.0  # Should scale with number of plots
        plt.close(fig_multi)
