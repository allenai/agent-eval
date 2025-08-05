"""Test leaderboard view functionality, focusing on paper workflow requirements."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

# Skip tests if seaborn not available
pytest.importorskip("seaborn")

from agenteval.leaderboard.view import _plot_combined_scatter


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
        from agenteval.leaderboard.view import _get_frontier_indices

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
