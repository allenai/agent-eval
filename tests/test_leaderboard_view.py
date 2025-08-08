"""Tests for LeaderboardViewer API contracts required by the leaderboard webapp client."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from agenteval.leaderboard.view import LeaderboardViewer


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
