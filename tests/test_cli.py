import os
import tempfile

from click.testing import CliRunner

from agenteval.cli import cli


def test_help_displays_usage():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


class TestEvalTaskFiltering:
    """Tests for --task and --task-category filtering in eval command."""

    def _create_test_config(self, tmpdir):
        """Create a test config file with multiple tasks and tags."""
        config_content = """
name: test-suite
version: "1.0.0"
splits:
  - name: test
    tasks:
      - name: ArxivDIGESTables_Clean_train
        path: tasks/arxiv_clean
        primary_metric: accuracy
        tags:
          - lit
          - data
      - name: CodeGenTask_v1
        path: tasks/codegen
        primary_metric: pass_rate
        tags:
          - code
      - name: DiscoveryBenchmark_2024
        path: tasks/discovery
        primary_metric: f1_score
        tags:
          - discovery
          - lit
"""
        config_path = os.path.join(tmpdir, "test_config.yml")
        with open(config_path, "w") as f:
            f.write(config_content)
        return config_path

    def test_eval_shows_task_filter_options_in_help(self):
        """Test that --task and --task-category options appear in eval help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "--task TEXT" in result.output
        assert "--task-category TEXT" in result.output
        assert "Filter to only run tasks whose name contains" in result.output
        assert "Filter to only run tasks with this tag" in result.output

    def test_eval_filter_by_task_name(self):
        """Test filtering by task name substring."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_test_config(tmpdir)
            log_dir = os.path.join(tmpdir, "logs")

            result = runner.invoke(
                cli,
                [
                    "eval",
                    "--config-path",
                    config_path,
                    "--split",
                    "test",
                    "--ignore-git",
                    "--config-only",
                    "--log-dir",
                    log_dir,
                    "--task",
                    "CodeGen",
                ],
            )

            assert result.exit_code == 0
            assert "Filtered to 1 of 3 tasks" in result.output
            assert "tasks/codegen" in result.output
            assert "tasks/arxiv_clean" not in result.output
            assert "tasks/discovery" not in result.output

    def test_eval_filter_by_task_category(self):
        """Test filtering by task category/tag."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_test_config(tmpdir)
            log_dir = os.path.join(tmpdir, "logs")

            result = runner.invoke(
                cli,
                [
                    "eval",
                    "--config-path",
                    config_path,
                    "--split",
                    "test",
                    "--ignore-git",
                    "--config-only",
                    "--log-dir",
                    log_dir,
                    "--task-category",
                    "lit",
                ],
            )

            assert result.exit_code == 0
            assert "Filtered to 2 of 3 tasks" in result.output
            assert "tasks/arxiv_clean" in result.output
            assert "tasks/discovery" in result.output
            assert "tasks/codegen" not in result.output

    def test_eval_filter_by_task_and_category_combined(self):
        """Test filtering by both task name and category (AND logic)."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_test_config(tmpdir)
            log_dir = os.path.join(tmpdir, "logs")

            result = runner.invoke(
                cli,
                [
                    "eval",
                    "--config-path",
                    config_path,
                    "--split",
                    "test",
                    "--ignore-git",
                    "--config-only",
                    "--log-dir",
                    log_dir,
                    "--task",
                    "Arxiv",
                    "--task-category",
                    "lit",
                ],
            )

            assert result.exit_code == 0
            assert "Filtered to 1 of 3 tasks" in result.output
            assert "tasks/arxiv_clean" in result.output
            # Discovery has "lit" tag but doesn't match "Arxiv"
            assert "tasks/discovery" not in result.output
            assert "tasks/codegen" not in result.output

    def test_eval_filter_multiple_task_names(self):
        """Test filtering with multiple --task options (OR logic within names)."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_test_config(tmpdir)
            log_dir = os.path.join(tmpdir, "logs")

            result = runner.invoke(
                cli,
                [
                    "eval",
                    "--config-path",
                    config_path,
                    "--split",
                    "test",
                    "--ignore-git",
                    "--config-only",
                    "--log-dir",
                    log_dir,
                    "--task",
                    "CodeGen",
                    "--task",
                    "Discovery",
                ],
            )

            assert result.exit_code == 0
            assert "Filtered to 2 of 3 tasks" in result.output
            assert "tasks/codegen" in result.output
            assert "tasks/discovery" in result.output
            assert "tasks/arxiv_clean" not in result.output

    def test_eval_filter_multiple_categories(self):
        """Test filtering with multiple --task-category options (OR logic)."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_test_config(tmpdir)
            log_dir = os.path.join(tmpdir, "logs")

            result = runner.invoke(
                cli,
                [
                    "eval",
                    "--config-path",
                    config_path,
                    "--split",
                    "test",
                    "--ignore-git",
                    "--config-only",
                    "--log-dir",
                    log_dir,
                    "--task-category",
                    "code",
                    "--task-category",
                    "discovery",
                ],
            )

            assert result.exit_code == 0
            assert "Filtered to 2 of 3 tasks" in result.output
            assert "tasks/codegen" in result.output
            assert "tasks/discovery" in result.output
            assert "tasks/arxiv_clean" not in result.output

    def test_eval_no_filter_runs_all_tasks(self):
        """Test that no filter runs all tasks."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_test_config(tmpdir)
            log_dir = os.path.join(tmpdir, "logs")

            result = runner.invoke(
                cli,
                [
                    "eval",
                    "--config-path",
                    config_path,
                    "--split",
                    "test",
                    "--ignore-git",
                    "--config-only",
                    "--log-dir",
                    log_dir,
                ],
            )

            assert result.exit_code == 0
            # Should not show "Filtered to" message
            assert "Filtered to" not in result.output
            assert "tasks/arxiv_clean" in result.output
            assert "tasks/codegen" in result.output
            assert "tasks/discovery" in result.output

    def test_eval_filter_no_matches_fails(self):
        """Test that filtering with no matches raises an error."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_test_config(tmpdir)
            log_dir = os.path.join(tmpdir, "logs")

            result = runner.invoke(
                cli,
                [
                    "eval",
                    "--config-path",
                    config_path,
                    "--split",
                    "test",
                    "--ignore-git",
                    "--config-only",
                    "--log-dir",
                    log_dir,
                    "--task",
                    "NonExistentTask",
                ],
            )

            assert result.exit_code != 0
            assert "No tasks match the specified filters" in result.output
