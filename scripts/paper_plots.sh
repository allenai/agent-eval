# !/usr/bin/env bash

# Clear cache
# rm -rf ~/.cache/huggingface

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Common options for all commands
COMMON_OPTS=(
    --split test 
    --config 1.0.0-dev1 
    --save-no-subdirs 
    --preserve-none-scores 
    --is-internal 
    --dedup latest
    --scatter-show-missing-cost 
    --scatter-x-log-scale 
    --scatter-legend-max-width 30 
    --scatter-figure-width 6.5 
    --exclude-agent "^ReAct$" 
    --exclude-agent "^Smolagents Coder"
    --exclude-agent "^Asta Code:Claude"
    --exclude-agent "^Asta Code:Gemini"
    --exclude-agent "^Asta Code:Llama"
    --exclude-agent "logs_url:hf://datasets/allenai/asta-bench-internal-submissions/1.0.0-dev1/test/miked-ai_Asta_DataVoyager_2025-07-11T19-46-16"
    --group-agent-fixed-colors 3
    --group-agent "^ReAct-:React"
    --group-agent "^Smolagents-:Smolagents"
    --group-agent "^Asta-v0:Asta v0"
    --group-agent "^Asta_Panda:Asta Panda"
    --group-agent "^Asta DataVoyager:Asta DataVoyager"
    --group-agent "^Asta Code:Asta Code"
    --group-agent "^Asta Scholar QA.*No Tables:Asta Scholar QA (No Tables)"
    --group-agent "^Asta Scholar QA$:Asta Scholar QA"
    --group-agent "^Asta Table Synthesis:Asta Table Synthesis"
    --group-agent "codescientist"
    --group-agent "faker"
    --group-agent "^futurehouse crow"
    --group-agent "^futurehouse falcon"
    --model-name-mapping-file "$SCRIPT_DIR/paper_plots_model_mapping.json"
)

# ======== Blog ======== #

# Overall figure with larger vertical proportions
agenteval lb view "${COMMON_OPTS[@]}" --scatter-subplot-height 3.1 --include-tag "lit:Literature Understanding" --include-tag "code:Code & Execution" --include-tag "data:Data Analysis" --include-tag "discovery:End-to-End Discovery" --save-dir plots/blog-fig-overall


# ======= Paper ======== #

# Overall figure
agenteval lb view "${COMMON_OPTS[@]}" --scatter-subplot-height 1.5 --include-tag "lit:Literature Understanding" --include-tag "code:Code & Execution" --include-tag "data:Data Analysis" --include-tag "discovery:End-to-End Discovery" --save-dir plots/paper-fig-overall
cp plots/paper-fig-overall/data.jsonl ../2025-astabench-paper/data/leaderboard_data/overall.jsonl
cp plots/paper-fig-overall/data_pipeline_statistics.json ../2025-astabench-paper/data/leaderboard_data/overall_data_pipeline_statistics.json
cp plots/paper-fig-overall/scatter.png ../2025-astabench-paper/figures/results-overall.png

# Detailed figures
agenteval lb view "${COMMON_OPTS[@]}" --tag lit --exclude-primary-metric --scatter-subplot-height 3.0 --scatter-subplot-spacing 0.2 --save-dir plots/paper-fig-lit-search --include-task "paper_finder_test:PaperFindingBench" --include-task "^paper_finder_litqa2:LitQA2-FullText-Search"
cp plots/paper-fig-lit-search/data.jsonl ../2025-astabench-paper/data/leaderboard_data/lit_search.jsonl
cp plots/paper-fig-lit-search/scatter.png ../2025-astabench-paper/figures/results-lit-search.png

agenteval lb view "${COMMON_OPTS[@]}" --tag lit --exclude-primary-metric --scatter-subplot-height 3.0 --scatter-subplot-spacing 0.2 --save-dir plots/paper-fig-lit-qa --include-task "sqa.*:ScholarQA-CS2" --include-task "^litqa2:LitQA2-FullText"
cp plots/paper-fig-lit-qa/data.jsonl ../2025-astabench-paper/data/leaderboard_data/lit_qa.jsonl
cp plots/paper-fig-lit-qa/scatter.png ../2025-astabench-paper/figures/results-lit-qa.png

agenteval lb view "${COMMON_OPTS[@]}" --tag lit --exclude-primary-metric --scatter-subplot-height 3.0 --scatter-subplot-spacing 0.2 --save-dir plots/paper-fig-lit-table  --include-task ".*digest.*:ArxivDIGESTables-Clean"
cp plots/paper-fig-lit-table/data.jsonl ../2025-astabench-paper/data/leaderboard_data/lit_table.jsonl
cp plots/paper-fig-lit-table/scatter.png ../2025-astabench-paper/figures/results-lit-table.png

agenteval lb view "${COMMON_OPTS[@]}" --tag code --exclude-primary-metric --scatter-subplot-height 2.5 --scatter-subplot-spacing 0.3 --save-dir plots/paper-fig-code --include-task ".*super.*:SUPER-Expert" --include-task ".*core.*:CORE-Bench-Hard" --include-task ".*ds.*:DS-1000"
cp plots/paper-fig-code/data.jsonl ../2025-astabench-paper/data/leaderboard_data/code_execution.jsonl
cp plots/paper-fig-code/scatter.png ../2025-astabench-paper/figures/results-coding.png

agenteval lb view "${COMMON_OPTS[@]}" --tag data --exclude-primary-metric --scatter-subplot-height 3.0 --scatter-subplot-spacing 0.2 --save-dir plots/paper-fig-data --include-task ".*:DiscoveryBench"
cp plots/paper-fig-data/data.jsonl ../2025-astabench-paper/data/leaderboard_data/data_analysis.jsonl
cp plots/paper-fig-data/scatter.png ../2025-astabench-paper/figures/results-data.png

agenteval lb view "${COMMON_OPTS[@]}" --tag discovery --exclude-primary-metric --scatter-subplot-height 3.0 --scatter-subplot-spacing 0.2 --save-dir plots/paper-fig-discovery --include-task ".*discovery_test:E2E-Bench" --include-task ".*hard.*:E2E-Bench-Hard"
cp plots/paper-fig-discovery/data.jsonl ../2025-astabench-paper/data/leaderboard_data/e2e_discovery.jsonl
cp plots/paper-fig-discovery/scatter.png ../2025-astabench-paper/figures/results-endtoend.png
