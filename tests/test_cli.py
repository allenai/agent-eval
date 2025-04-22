from click.testing import CliRunner

from agenteval.cli import cli


def test_help_displays_usage():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
