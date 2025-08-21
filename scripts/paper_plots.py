#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_script_dir():
    """Get the directory where this script is located."""
    return Path(__file__).parent.resolve()


def run_command(cmd, check=True):
    """Run a shell command and optionally check for errors."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=False, text=True)
    # result = subprocess.run(cmd, check=check, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"Error output: {result.stderr}")
    return result


def copy_file(src, dst):
    """Copy a file from source to destination, creating directories as needed."""
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied {src} to {dst}")


def main():
    # Check for required environment variable
    if "RESULTS_REPO_ID" not in os.environ:
        print(
            "ERROR: RESULTS_REPO_ID environment variable is not set.", file=sys.stderr
        )
        print("Please set it to your results repository ID, e.g.:", file=sys.stderr)
        print(
            "  export RESULTS_REPO_ID=allenai/asta-bench-internal-results",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate paper plots for ASTAbench")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../paper-asta",
        help="Output directory for paper figures and data (default: ../../paper-asta)",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["png", "pdf"],
        default="pdf",
        help="Format for saving plots (default: pdf)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    script_dir = get_script_dir()

    # Common options for all commands
    common_opts = [
        "--split",
        "test",
        "--config",
        "1.0.0-dev1",
        "--save-no-subdirs",
        "--preserve-none-scores",
        "--is-internal",
        "--dedup",
        "latest",
        "--scatter-show-missing-cost",
        "--scatter-x-log-scale",
        "--scatter-legend-max-width",
        "30",
        "--scatter-figure-width",
        "6.5",
        "--exclude-agent",
        "^ReAct$",
        "--exclude-agent",
        "^Smolagents Coder",
        "--exclude-agent",
        "^Asta Code:Claude",
        "--exclude-agent",
        "^Asta Code:Gemini",
        "--exclude-agent",
        "^Asta Code:Llama",
        "--exclude-agent",
        "logs_url:hf://datasets/allenai/asta-bench-internal-submissions/1.0.0-dev1/test/miked-ai_Asta_DataVoyager_2025-07-11T19-46-16",
        "--group-agent-fixed-colors",
        "3",
        "--group-agent",
        "^ReAct-:React",
        "--group-agent",
        "^Smolagents-:Smolagents",
        "--group-agent",
        "^Asta-v0:Asta v0",
        "--group-agent",
        "^Asta_Panda:Asta Panda",
        "--group-agent",
        "^Asta DataVoyager:Asta DataVoyager",
        "--group-agent",
        "^Asta Code:Asta Code",
        "--group-agent",
        "^Asta Scholar QA.*No Tables:Asta Scholar QA (No Tables)",
        "--group-agent",
        "^Asta Scholar QA$:Asta Scholar QA",
        "--group-agent",
        "^Asta Table Synthesis:Asta Table Synthesis",
        "--group-agent",
        "codescientist",
        "--group-agent",
        "faker",
        "--group-agent",
        "^futurehouse crow",
        "--group-agent",
        "^futurehouse falcon",
        "--model-name-mapping-file",
        str(script_dir / "paper_plots_model_mapping.json"),
        "--save-format",
        args.save_format,
    ]

    # Exclude these from the overall plots
    overall_excludes = [
        "--exclude-agent",
        "^Asta DataVoyager",  # Exclude DataVoyager from overall plots
        "--exclude-agent",
        "^Asta_Panda",
        "--exclude-agent",
        "^codescientist",
    ]

    # ======== Blog ======== #

    # Overall figure with larger vertical proportions
    cmd = (
        ["agenteval", "lb", "view"]
        + common_opts
        # + overall_excludes
        + [
            "--scatter-subplot-height",
            "3.1",
            "--include-tag",
            "lit:Literature Understanding",
            "--include-tag",
            "code:Code & Execution",
            "--include-tag",
            "data:Data Analysis",
            "--include-tag",
            "discovery:End-to-End Discovery",
            "--save-dir",
            "plots/blog-fig-overall",
        ]
    )
    run_command(cmd)

    # ======= Paper ======== #

    # Overall figure
    cmd = (
        ["agenteval", "lb", "view"]
        + common_opts
        # + overall_excludes
        + [
            "--scatter-subplot-height",
            "1.5",
            "--include-tag",
            "lit:Literature Understanding",
            "--include-tag",
            "code:Code & Execution",
            "--include-tag",
            "data:Data Analysis",
            "--include-tag",
            "discovery:End-to-End Discovery",
            "--save-dir",
            "plots/paper-fig-overall",
        ]
    )
    run_command(cmd)
    copy_file(
        "plots/paper-fig-overall/data.jsonl",
        output_dir / "data/leaderboard_data/overall.jsonl",
    )
    copy_file(
        "plots/paper-fig-overall/data_pipeline_statistics.json",
        output_dir / "data/leaderboard_data/overall_data_pipeline_statistics.json",
    )
    copy_file(
        f"plots/paper-fig-overall/scatter.{args.save_format}",
        output_dir / f"figures/results-overall.{args.save_format}",
    )

    # Detailed figures - Literature Search
    cmd = (
        ["agenteval", "lb", "view"]
        + common_opts
        + [
            "--tag",
            "lit",
            "--exclude-primary-metric",
            "--scatter-subplot-height",
            "3.0",
            "--scatter-subplot-spacing",
            "0.2",
            "--save-dir",
            "plots/paper-fig-lit-search",
            "--include-task",
            "paper_finder_test:PaperFindingBench",
            "--include-task",
            "^paper_finder_litqa2:LitQA2-FullText-Search",
        ]
    )
    run_command(cmd)
    copy_file(
        "plots/paper-fig-lit-search/data.jsonl",
        output_dir / "data/leaderboard_data/lit_search.jsonl",
    )
    copy_file(
        f"plots/paper-fig-lit-search/scatter.{args.save_format}",
        output_dir / f"figures/results-lit-search.{args.save_format}",
    )

    # Literature QA
    cmd = (
        ["agenteval", "lb", "view"]
        + common_opts
        + [
            "--tag",
            "lit",
            "--exclude-primary-metric",
            "--scatter-subplot-height",
            "3.0",
            "--scatter-subplot-spacing",
            "0.2",
            "--save-dir",
            "plots/paper-fig-lit-qa",
            "--include-task",
            "sqa.*:ScholarQA-CS2",
            "--include-task",
            "^litqa2:LitQA2-FullText",
        ]
    )
    run_command(cmd)
    copy_file(
        "plots/paper-fig-lit-qa/data.jsonl",
        output_dir / "data/leaderboard_data/lit_qa.jsonl",
    )
    copy_file(
        f"plots/paper-fig-lit-qa/scatter.{args.save_format}",
        output_dir / f"figures/results-lit-qa.{args.save_format}",
    )

    # Literature Table
    cmd = (
        ["agenteval", "lb", "view"]
        + common_opts
        + [
            "--tag",
            "lit",
            "--exclude-primary-metric",
            "--scatter-subplot-height",
            "3.0",
            "--scatter-subplot-spacing",
            "0.2",
            "--save-dir",
            "plots/paper-fig-lit-table",
            "--include-task",
            ".*digest.*:ArxivDIGESTables-Clean",
        ]
    )
    run_command(cmd)
    copy_file(
        "plots/paper-fig-lit-table/data.jsonl",
        output_dir / "data/leaderboard_data/lit_table.jsonl",
    )
    copy_file(
        f"plots/paper-fig-lit-table/scatter.{args.save_format}",
        output_dir / f"figures/results-lit-table.{args.save_format}",
    )

    # Code
    cmd = (
        ["agenteval", "lb", "view"]
        + common_opts
        + [
            "--tag",
            "code",
            "--exclude-primary-metric",
            "--scatter-subplot-height",
            "2.5",
            "--scatter-subplot-spacing",
            "0.3",
            "--save-dir",
            "plots/paper-fig-code",
            "--include-task",
            ".*super.*:SUPER-Expert",
            "--include-task",
            ".*core.*:CORE-Bench-Hard",
            "--include-task",
            ".*ds.*:DS-1000",
        ]
    )
    run_command(cmd)
    copy_file(
        "plots/paper-fig-code/data.jsonl",
        output_dir / "data/leaderboard_data/code_execution.jsonl",
    )
    copy_file(
        f"plots/paper-fig-code/scatter.{args.save_format}",
        output_dir / f"figures/results-coding.{args.save_format}",
    )

    # Data Analysis
    cmd = (
        ["agenteval", "lb", "view"]
        + common_opts
        + [
            "--tag",
            "data",
            "--exclude-primary-metric",
            "--scatter-subplot-height",
            "3.0",
            "--scatter-subplot-spacing",
            "0.2",
            "--save-dir",
            "plots/paper-fig-data",
            "--include-task",
            ".*:DiscoveryBench",
        ]
    )
    run_command(cmd)
    copy_file(
        "plots/paper-fig-data/data.jsonl",
        output_dir / "data/leaderboard_data/data_analysis.jsonl",
    )
    copy_file(
        f"plots/paper-fig-data/scatter.{args.save_format}",
        output_dir / f"figures/results-data.{args.save_format}",
    )

    # Discovery
    cmd = (
        ["agenteval", "lb", "view"]
        + common_opts
        + [
            "--tag",
            "discovery",
            "--exclude-primary-metric",
            "--scatter-subplot-height",
            "3.0",
            "--scatter-subplot-spacing",
            "0.2",
            "--save-dir",
            "plots/paper-fig-discovery",
            "--include-task",
            ".*discovery_test:E2E-Bench",
            "--include-task",
            ".*hard.*:E2E-Bench-Hard",
        ]
    )
    run_command(cmd)
    copy_file(
        "plots/paper-fig-discovery/data.jsonl",
        output_dir / "data/leaderboard_data/e2e_discovery.jsonl",
    )
    copy_file(
        f"plots/paper-fig-discovery/scatter.{args.save_format}",
        output_dir / f"figures/results-endtoend.{args.save_format}",
    )


if __name__ == "__main__":
    main()
