#!/usr/bin/env python3
"""Generate LaTeX table rows from JSONL files created by paper_plots.py"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Decimal places for formatting
SCORE_DECIMALS = 4  # Decimal places for score columns (matches paper format)
COST_DECIMALS = 4  # Decimal places for cost columns (matches paper format)
COMMENT_PREFIX = "%#gen:"

# Third-party agents that don't report costs
THIRD_PARTY_AGENTS = [
    "Elicit",
    "You.com Research API",
    "You.com Smart API",
    "You.com Search API",
    "OpenAI Deep Research",
    "SciSpace Deep Review",
    "Perplexity Sonar Deep Research",
]

# Agents to exclude from specific table types
TABLE_EXCLUSIONS = {
    "overall": [
        "Asta DataVoyager",
        "Asta_CodeScientist",
        "Faker",
        "Asta Panda",
    ],
    # Add other table types as needed
    # "lit_search": [...],
    # "code": [...],
}

# Agent capabilities mapping
# Maps agent names to their expected capabilities
AGENT_CAPABILITIES = {
    # Literature-specific agents
    "Elicit": {
        "can_do": "ScholarQA-CS2",
        "cannot_do": [
            "LitQA2-FullText",
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
            "E2E-Bench",
            "E2E-Bench-Hard",
        ],
    },
    "OpenSciLM": {
        "can_do": "ScholarQA-CS2",
        "cannot_do": [
            "LitQA2-FullText",
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
            "E2E-Bench",
            "E2E-Bench-Hard",
        ],
    },
    "STORM": {
        "can_do": "ScholarQA-CS2",
        "cannot_do": [
            "LitQA2-FullText",
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
            "E2E-Bench",
            "E2E-Bench-Hard",
        ],
    },
    "Asta Scholar QA": {
        "can_do": ["ScholarQA-CS2"],
        "cannot_do": [
            "LitQA2-FullText",
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
            "E2E-Bench",
            "E2E-Bench-Hard",
        ],
    },
    "Asta Scholar QA (No Tables)": {
        "can_do": ["ScholarQA-CS2"],
        "cannot_do": [
            "LitQA2-FullText",
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
            "E2E-Bench",
            "E2E-Bench-Hard",
        ],
    },
    "futurehouse crow": {
        "can_do": ["ScholarQA-CS2", "LitQA2-FullText"],
        "cannot_do": [
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
            "E2E-Bench",
            "E2E-Bench-Hard",
        ],
    },
    "futurehouse falcon": {
        "can_do": ["ScholarQA-CS2", "LitQA2-FullText"],
        "cannot_do": [
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
            "E2E-Bench",
            "E2E-Bench-Hard",
        ],
    },
    "Asta Paper Finder": {
        "can_do": ["PaperFindingBench", "LitQA2-FullText-Search"],
        "cannot_do": [
            "ScholarQA-CS2",
            "LitQA2-FullText",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
            "E2E-Bench",
            "E2E-Bench-Hard",
        ],
    },
    "Asta Table Synthesis": {
        "can_do": ["ArxivDIGESTables-Clean"],
        "cannot_do": [
            "ScholarQA-CS2",
            "LitQA2-FullText",
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
            "E2E-Bench",
            "E2E-Bench-Hard",
        ],
    },
    # Data-specific agents
    "Asta DataVoyager": {
        "can_do": ["DiscoveryBench"],
        "cannot_do": [
            "ScholarQA-CS2",
            "LitQA2-FullText",
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "E2E-Bench",
            "E2E-Bench-Hard",
        ],
    },
    # Code-specific agents
    "Asta Code": {
        "can_do": ["SUPER-Expert"],
        "cannot_do": [
            "ScholarQA-CS2",
            "LitQA2-FullText",
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
            "E2E-Bench",
            "E2E-Bench-Hard",
        ],
    },
    # End-to-end specific agents
    "Asta Panda": {
        "can_do": ["E2E-Bench", "E2E-Bench-Hard"],
        "cannot_do": [
            "ScholarQA-CS2",
            "LitQA2-FullText",
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
        ],
    },
    "codescientist": {
        "can_do": ["E2E-Bench", "E2E-Bench-Hard"],
        "cannot_do": [
            "ScholarQA-CS2",
            "LitQA2-FullText",
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
        ],
    },
    "faker": {
        "can_do": ["E2E-Bench", "E2E-Bench-Hard"],
        "cannot_do": [
            "ScholarQA-CS2",
            "LitQA2-FullText",
            "LitQA2-FullText-Search",
            "PaperFindingBench",
            "ArxivDIGESTables-Clean",
            "SUPER-Expert",
            "CORE-Bench-Hard",
            "DS-1000",
            "DiscoveryBench",
        ],
    },
    # General agents (can do everything)
    "ReAct": {"can_do": "all"},
    "React": {"can_do": "all"},
    "Smolagents": {"can_do": "all"},
    "Smolagents Coder": {"can_do": "all"},
    "Asta v0": {"can_do": "all"},
}


def is_task_expected_missing(agent_name: str, task_name: str) -> bool:
    """Check if a missing value is expected for this agent/task combination."""
    if agent_name not in AGENT_CAPABILITIES:
        return False  # Unknown agent, treat as unexpected missing

    capabilities = AGENT_CAPABILITIES[agent_name]
    if capabilities.get("can_do") == "all":
        return False  # Agent can do everything, missing is unexpected

    # Check if task is in the cannot_do list
    cannot_do = capabilities.get("cannot_do", [])
    for task_pattern in cannot_do:
        if task_pattern in task_name:
            return True

    return False


def should_exclude_agent(agent_name: str, table_type: str) -> bool:
    """Check if an agent should be excluded from a specific table type."""
    exclusions = TABLE_EXCLUSIONS.get(table_type, [])
    return agent_name in exclusions


# Mappings from agent names to LaTeX macros
AGENT_MACROS = {
    "ReAct": r"\agentReAct",
    "React": r"\agentReAct",
    "Smolagents": r"\agentSmolagents",
    "Smolagents Coder": r"\agentSmolagents",
    "Asta Code": r"\agentAstaCode",
    "Asta v0": r"\agentAsta",
    "Asta DataVoyager": r"\agentDataVoyager",
    "Asta DataVoyager (Short)": r"\agentDataVoyagerShort",
    "Asta Panda": r"\agentAutoAsta",
    "Asta Paper Finder": r"\agentPaperFinder",
    "Asta Scholar QA": r"\agentScholarQA",
    "Asta Scholar QA (w/ Tables)": r"\agentScholarQA",
    "Asta Scholar QA (No Tables)": r"\agentScholarQANoTables",
    "Asta Scholar QA (Lite)": r"\agentSQALite",
    "Asta Scholar QA (SFT)": r"\agentSQATrain",
    "Asta Table Synthesis": r"\agentAstaTableAgent",
    "Asta_CodeScientist": r"\agentCodeScientist",
    "codescientist": r"\agentCodeScientist",
    "Faker": r"\agentFaker",
    "faker": r"\agentFaker",
    "futurehouse crow": r"\agentFutureHouseCrow",
    "FutureHouse Crow": r"\agentFutureHouseCrow",
    "futurehouse falcon": r"\agentFutureHouseFalcon",
    "FutureHouse Falcon": r"\agentFutureHouseFalcon",
    "Elicit": r"\agentElicit",
    "Perplexity Deep Research": r"\agentPerplexitySQA",
    "Perplexity Sonar Deep Research": r"\agentPerplexitySQA",
    "You.com Research API": r"\agentYouComResearch",
    "You.com Smart API": r"\agentYouComSmart",
    "You.com Search API": r"\agentYouComSearch",
    "SciSpace Deep Review": r"\agentSciSpace",
    "OpenSciLM": r"\agentOpenScholar",
    "OpenAI Deep Research": r"\agentOpenAIDeepResearch",
    "Gemini Deep Research": r"\agentGeminiDeepResearch",
    "STORM": r"\agentSTORM",
    "Generate": r"\agentSingleShot",
}

# Mappings from model names to LaTeX macros
MODEL_MACROS = {
    "mixture": "mixture",
    # OpenAI models
    "gpt-5": r"\modelGPTFiveShort",
    "gpt-5-mini": r"\modelGPTFiveMiniShort",
    "gpt-4.1": r"\modelGPTFourPointOneShort",
    "gpt-4.1-nano": r"\modelGPTFourPointOneNanoShort",
    "gpt-4.1-mini": r"\modelGPTFourPointOneMiniShort",
    "gpt-4o": r"\modelGPTFourOShort",
    "gpt-4o-mini": r"\modelGPTFourOMiniShort",
    "gpt-4-turbo": r"\modelGPTFourTurboShort",
    "gpt-3.5-turbo": r"\modelGPTThreeFiveTurboShort",
    "codex-mini": r"\modelCodexMiniShort",
    "o3": r"\modelOThreeShort",
    "o3-pro": r"\modelOThreeProShort",
    "o3-mini": r"\modelOThreeMiniShort",
    "o3-deep-research": r"\modelOThreeDRShort",
    "o4-mini": r"\modelOFourMiniShort",
    # Anthropic models
    "claude-3-5-haiku": r"\modelClaudeThreeFiveHaikuShort",
    "claude-sonnet-4": r"\modelClaudeSonnetFourShort",
    "claude-opus-4": r"\modelClaudeOpusFourShort",
    "claude-3-7-sonnet": r"\modelClaudeSonnetThreeSevenShort",
    # Google models
    "gemini-2.5-flash": r"\modelGeminiTwoPointFiveFlashShort",
    "gemini-2.5-flash-lite": r"\modelGeminiTwoPointFiveFlashLiteShort",
    "gemini-2.5-pro": r"\modelGeminiTwoPointFiveProShort",
    "gemini-2-flash": r"\modelGeminiTwoFlashShort",
    "gemini-2.0-flash": r"\modelGeminiTwoFlashShort",
    "gemma-3-27b": r"\modelGemmaThreeTwentySevenShort",
    "gemma-3n-e4b-it": r"\modelGemmaThreeNShort",
    # Meta models
    "llama-4-scout": r"\modelLlamaFourScoutShort",
    "llama-4-maverick": r"\modelLlamaFourMaverickShort",
    # DeepSeek models
    "deepseek-v3": r"\modelDeepSeekVThreeShort",
    "deepseek-r1": r"\modelDeepSeekROneShort",
    # XAI models
    "grok-3": r"\modelGrokThreeShort",
    "grok-3-mini": r"\modelGrokThreeMiniShort",
    # Microsoft models
    "phi-4-reasoning": r"\modelPhiFourReasoningShort",
    "phi-4-reasoning-plus": r"\modelPhiFourReasoningPlusShort",
    # Mistral models
    "mistral-large": r"\modelMistralLargeShort",
    "mistral-medium-3": r"\modelMistralMediumThreeShort",
    "mistral-small": r"\modelMistralSmallShort",
    "mistral-codestral": r"\modelMistralCodestralShort",
    "mistral-devstral": r"\modelMistralDevstralShort",
    # Alibaba models
    "qwen3-8b": r"\modelQwenThreeEightB",
    "qwen3-235b": r"\modelQwenThreeTwoThreeFiveShort",
    "qwq-32b": r"\modelQwqThirtyTwoShort",
    # Perplexity models
    "perplexity-sonar": r"\modelPerplexitySonarShort",
    "perplexity-sonar-pro": r"\modelPerplexitySonarProShort",
    "perplexity-sonar-reasoning": r"\modelPerplexitySonarReasoningShort",
    "perplexity-sonar-reasoning-pro": r"\modelPerplexitySonarReasoningProShort",
    "sonar-deep-research": r"\modelPerplexitySonarDeepResearch",
    "llama-3.1-openscholar-8b": r"\modelOpenScholar",
}

# Mappings for openness symbols
OPENNESS_SYMBOLS = {
    "Open Source": r"\opennessSymbolOpenClosedWeight",
    "Open source & closed weights": r"\opennessSymbolOpenClosedWeight",  # Alternative format
    "Open Source + Open Weights": r"\opennessSymbolOpenOpenWeight",
    "Open source & open weights": r"\opennessSymbolOpenOpenWeight",  # Alternative format
    "Closed": r"\opennessSymbolClosed",
    "API Available": r"\opennessSymbolClosedWithApi",
}

# Mappings for tooling symbols
TOOLING_SYMBOLS = {
    "Standard": r"\toolingSymbolStandard",
    "Custom with Standard Search": r"\toolingSymbolEquivalent",
    "Custom interface": r"\toolingSymbolEquivalent",  # Alternative format
    "Custom": r"\toolingSymbolCustom",
    "Fully Custom": r"\toolingSymbolCustom",
    "Fully custom": r"\toolingSymbolCustom",  # Alternative format
}


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_value_and_ci(
    entry: Dict, base_key: str
) -> Tuple[Optional[float], Optional[float]]:
    """Extract value and confidence interval from entry.

    For a base key like "Overall", looks for:
    - "Overall" for the value
    - "Overall CI" or "Overall 95% CI" for the confidence interval

    Returns (value, ci) where ci is already the 95% CI.
    """
    value = entry.get(base_key)

    # Try different CI key formats
    ci = None
    ci_keys = [
        f"{base_key} CI",
        f"{base_key} 95% CI",
        base_key.replace(" score", "") + " 95% CI" if " score" in base_key else None,
        base_key.replace(" cost", " cost 95% CI") if " cost" in base_key else None,
    ]

    for ci_key in ci_keys:
        if ci_key and ci_key in entry:
            ci = entry.get(ci_key)
            break

    return value, ci


def format_value_with_ci(
    value: Optional[float],
    ci: Optional[float],
    decimals: int,
    is_score: bool = False,
    agent_name: str = None,
    task_name: str = None,
    on_frontier: bool = False,
) -> str:
    """Format a value with optional confidence interval.

    CI is already the 95% confidence interval from the JSON.
    If agent_name and task_name are provided, checks if missing value is expected.
    If on_frontier is True, adds asterisk to CI values to indicate Pareto frontier.
    """
    if value is None:
        if agent_name and task_name and is_task_expected_missing(agent_name, task_name):
            return "{--}"
        # For cost columns, check if this is a third-party agent that doesn't report costs
        elif (
            not is_score
            and agent_name
            and any(third_party == agent_name for third_party in THIRD_PARTY_AGENTS)
        ):
            return r"\missing"
        else:
            return r"{\red{?}}"

    # Convert to percentage if needed (scores are in 0-1 range)
    if is_score and value <= 1.0:
        value = value * 100
        if ci is not None:
            ci = ci * 100

    formatted_val = f"{value:.{decimals}f}"

    if ci is not None and ci > 0:
        formatted_ci = f"{ci:.{decimals}f}"
        formatted_val = f"{formatted_val} +- {formatted_ci}"

    if on_frontier:
        formatted_val = f"\\B {formatted_val}"

    return formatted_val


def process_single_model(model: str, full_entry: dict) -> str:
    """Process a single model name to get its LaTeX macro with appropriate suffixes."""
    # Check if model has dagger and effort specifiers
    has_dagger = "†" in model
    is_minimal_effort = ":effort=minimal" in model
    if "gpt-5" in model and "minimal_reasoning" in full_entry["Agent description"]:
        is_minimal_effort = True
    model_without_symbols = (
        model.replace("†", "").replace(":effort=minimal", "").strip().lower()
    )

    # Try to find exact match (without symbols and effort specifiers)
    if model_without_symbols in MODEL_MACROS:
        macro = MODEL_MACROS[model_without_symbols]

        # For unpinned models, replace "Short" with "Unpinned"
        if has_dagger:
            if "Short" in macro:
                base_macro = macro.replace("Short", "")
                if is_minimal_effort:
                    return f"{base_macro}UnpinnedMinimal"
                else:
                    return f"{base_macro}Unpinned"
            else:
                # Fallback if no "Short" in macro name
                if is_minimal_effort:
                    return f"{macro}UnpinnedMinimal"
                else:
                    return f"{macro}Unpinned"

        return macro

    # If no match found, return the model name as-is with warning
    print(f"Warning: No macro found for model: {model}")
    return model


def get_model_macro(entry: dict) -> str:
    """Get LaTeX macro for model(s)."""
    models = entry.get("LLM base", [])
    if not models:
        return r"\missing"

    # Handle mixture case - more than 3 models becomes "mixture"
    if len(models) > 3:
        return "mixture"
    elif len(models) > 1:
        # 2-3 models: Get macros for each model and join with comma and space
        model_macros = []
        for model in models:
            model_macros.append(process_single_model(model, full_entry=entry))

        return ", ".join(model_macros)

    # Handle "mixture" keyword
    if "mixture" in str(models).lower():
        return "mixture"

    model = models[0] if isinstance(models, list) else models
    return process_single_model(model, full_entry=entry)


def get_agent_macro(agent_name: str) -> str:
    """Get LaTeX macro for agent."""
    # Try exact match first
    if agent_name in AGENT_MACROS:
        return AGENT_MACROS[agent_name]

    # Try case-insensitive partial match
    agent_lower = agent_name.lower()
    for key, macro in AGENT_MACROS.items():
        if key.lower() in agent_lower or agent_lower in key.lower():
            return macro

    print(f"Warning: No macro found for agent: {agent_name}")
    return agent_name


def get_submission_id(entry: Dict) -> str:
    """Extract submission ID from Logs field."""
    logs = entry.get("Logs", "")
    if logs:
        # Extract the submission name from the logs path
        # Format: hf://datasets/.../test/{submission_id}
        parts = logs.split("/")
        if len(parts) > 0:
            # Get the last part which is the submission ID
            return parts[-1]
    return ""


def generate_overall_row(
    entry: Dict, include_ci: bool = False, include_submission_id: bool = True
) -> str:
    """Generate a row for the overall results table."""
    openness = OPENNESS_SYMBOLS.get(entry.get("Openness", ""), r"{\red{?}}")
    tooling = TOOLING_SYMBOLS.get(entry.get("Agent tooling", ""), r"{\red{?}}")
    agent_name = entry.get("grouped_agent_name", entry.get("Agent", ""))
    agent = get_agent_macro(agent_name)
    model = get_model_macro(entry)

    row = f"    {openness} & {tooling} & {agent} & {model}"

    # Overall - no specific task, so no expected missing check
    score, score_ci = get_value_and_ci(entry, "Overall")
    cost, cost_ci = get_value_and_ci(entry, "Overall cost")
    overall_frontier = entry["Overall frontier"]
    if not include_ci:
        score_ci = None
        cost_ci = None
    row += f" & {format_value_with_ci(score, score_ci, SCORE_DECIMALS, is_score=True, on_frontier=overall_frontier)}"
    row += f" & {format_value_with_ci(cost, cost_ci, COST_DECIMALS, on_frontier=overall_frontier)}"

    # Literature Understanding - check for expected missing
    score, score_ci = get_value_and_ci(entry, "Literature Understanding score")
    cost, cost_ci = get_value_and_ci(entry, "Literature Understanding cost")
    lit_frontier = entry["Literature Understanding frontier"]
    # For overall categories, we check if agent can do ANY task in that category
    if not include_ci:
        score_ci = None
        cost_ci = None
    row += f" & {format_value_with_ci(score, score_ci, SCORE_DECIMALS, is_score=True, agent_name=agent_name, task_name='Literature', on_frontier=lit_frontier)}"
    row += f" & {format_value_with_ci(cost, cost_ci, COST_DECIMALS, agent_name=agent_name, task_name='Literature', on_frontier=lit_frontier)}"

    # Code & Execution
    score, score_ci = get_value_and_ci(entry, "Code & Execution score")
    cost, cost_ci = get_value_and_ci(entry, "Code & Execution cost")
    code_frontier = entry["Code & Execution frontier"]
    if not include_ci:
        score_ci = None
        cost_ci = None
    row += f" & {format_value_with_ci(score, score_ci, SCORE_DECIMALS, is_score=True, agent_name=agent_name, task_name='Code', on_frontier=code_frontier)}"
    row += f" & {format_value_with_ci(cost, cost_ci, COST_DECIMALS, agent_name=agent_name, task_name='Code', on_frontier=code_frontier)}"

    # Data Analysis
    score, score_ci = get_value_and_ci(entry, "Data Analysis score")
    cost, cost_ci = get_value_and_ci(entry, "Data Analysis cost")
    data_frontier = entry["Data Analysis frontier"]
    if not include_ci:
        score_ci = None
        cost_ci = None
    row += f" & {format_value_with_ci(score, score_ci, SCORE_DECIMALS, is_score=True, agent_name=agent_name, task_name='DiscoveryBench', on_frontier=data_frontier)}"
    row += f" & {format_value_with_ci(cost, cost_ci, COST_DECIMALS, agent_name=agent_name, task_name='DiscoveryBench', on_frontier=data_frontier)}"

    # End-to-End Discovery
    score, score_ci = get_value_and_ci(entry, "End-to-End Discovery score")
    cost, cost_ci = get_value_and_ci(entry, "End-to-End Discovery cost")
    e2e_frontier = entry["End-to-End Discovery frontier"]
    if not include_ci:
        score_ci = None
        cost_ci = None
    row += f" & {format_value_with_ci(score, score_ci, SCORE_DECIMALS, is_score=True, agent_name=agent_name, task_name='E2E-Bench', on_frontier=e2e_frontier)}"
    row += f" & {format_value_with_ci(cost, cost_ci, COST_DECIMALS, agent_name=agent_name, task_name='E2E-Bench', on_frontier=e2e_frontier)}"

    row += r" \\"

    if include_submission_id:
        submission_id = get_submission_id(entry)
        if submission_id:
            row += f" {COMMENT_PREFIX} {submission_id}"

    return row


def generate_task_specific_row(
    entry: Dict,
    score_keys: List[str],
    cost_keys: List[str],
    include_ci: bool = True,
    include_submission_id: bool = True,
) -> str:
    """Generate a row for task-specific tables."""
    openness = OPENNESS_SYMBOLS.get(entry.get("Openness", ""), r"{\red{?}}")
    tooling = TOOLING_SYMBOLS.get(entry.get("Agent tooling", ""), r"{\red{?}}")
    agent_name = entry.get("grouped_agent_name", entry.get("Agent", ""))
    agent = get_agent_macro(agent_name)
    model = get_model_macro(entry)

    row = f"    {openness} & {tooling} & {agent} & {model}"

    # Add score/cost pairs for each task
    for score_key, cost_key in zip(score_keys, cost_keys):
        # Extract task name from the key (e.g., "PaperFindingBench score" -> "PaperFindingBench")
        task_name = score_key.replace(" score", "").replace(" cost", "")

        score, score_ci = get_value_and_ci(entry, score_key)
        cost, cost_ci = get_value_and_ci(entry, cost_key)

        # Check if this task is on the frontier
        frontier_key = f"{task_name} frontier"
        on_frontier = entry[frontier_key]

        if not include_ci:
            score_ci = None
            cost_ci = None
        row += f" & {format_value_with_ci(score, score_ci, SCORE_DECIMALS, is_score=True, agent_name=agent_name, task_name=task_name, on_frontier=on_frontier)}"
        row += f" & {format_value_with_ci(cost, cost_ci, COST_DECIMALS, agent_name=agent_name, task_name=task_name, on_frontier=on_frontier)}"

    row += r" \\"

    if include_submission_id:
        submission_id = get_submission_id(entry)
        if submission_id:
            row += f" {COMMENT_PREFIX} {submission_id}"

    return row


def main():
    """Generate LaTeX rows for all tables."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table rows from JSONL files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../paper-asta/data/leaderboard_data",
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--table",
        type=str,
        choices=[
            "overall",
            "lit_search",
            "lit_qa",
            "lit_table",
            "code",
            "data",
            "discovery",
            "all",
        ],
        default="all",
        help="Which table(s) to generate",
    )
    parser.add_argument(
        "--include-ci",
        action="store_true",
        help="Include confidence intervals in output (default: False for overall table, True for others)",
    )
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Exclude confidence intervals from all tables",
    )
    parser.add_argument(
        "--no-submission-id",
        action="store_true",
        help="Exclude submission ID comments from generated rows",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    tables = {
        "overall": {
            "file": "overall.jsonl",
            "name": "Overall Results",
            "generator": generate_overall_row,
            "default_ci": False,  # Overall table typically doesn't show CIs
        },
        "lit_search": {
            "file": "lit_search.jsonl",
            "name": "Literature Search",
            "score_keys": ["PaperFindingBench score", "LitQA2-FullText-Search score"],
            "cost_keys": ["PaperFindingBench cost", "LitQA2-FullText-Search cost"],
            "default_ci": True,
        },
        "lit_qa": {
            "file": "lit_qa.jsonl",
            "name": "Literature QA",
            "score_keys": ["ScholarQA-CS2 score", "LitQA2-FullText score"],
            "cost_keys": ["ScholarQA-CS2 cost", "LitQA2-FullText cost"],
            "default_ci": True,
        },
        "lit_table": {
            "file": "lit_table.jsonl",
            "name": "Literature Tables",
            "score_keys": ["ArxivDIGESTables-Clean score"],
            "cost_keys": ["ArxivDIGESTables-Clean cost"],
            "default_ci": True,
        },
        "code": {
            "file": "code_execution.jsonl",
            "name": "Coding",
            "score_keys": [
                "SUPER-Expert score",
                "CORE-Bench-Hard score",
                "DS-1000 score",
            ],
            "cost_keys": ["SUPER-Expert cost", "CORE-Bench-Hard cost", "DS-1000 cost"],
            "default_ci": True,
        },
        "data": {
            "file": "data_analysis.jsonl",
            "name": "Data Analysis",
            "score_keys": ["DiscoveryBench score"],
            "cost_keys": ["DiscoveryBench cost"],
            "default_ci": True,
        },
        "discovery": {
            "file": "e2e_discovery.jsonl",
            "name": "End-to-End Discovery",
            "score_keys": ["E2E-Bench score", "E2E-Bench-Hard score"],
            "cost_keys": ["E2E-Bench cost", "E2E-Bench-Hard cost"],
            "default_ci": True,
        },
    }

    tables_to_generate = [args.table] if args.table != "all" else tables.keys()

    for table_key in tables_to_generate:
        if table_key not in tables:
            continue

        table_info = tables[table_key]
        file_path = data_dir / table_info["file"]

        if not file_path.exists():
            print(f"Warning: {file_path} does not exist, skipping {table_info['name']}")
            continue

        data = load_jsonl(file_path)

        # Determine whether to include CIs
        if args.no_ci:
            include_ci = False
        elif args.include_ci:
            include_ci = True
        else:
            include_ci = table_info.get("default_ci", True)

        print(f"\n=== {table_info['name']} ===")
        print(f"% Generated from {file_path}")
        print(f"% Total entries: {len(data)}")
        print(f"% Confidence intervals: {'included' if include_ci else 'excluded'}")

        # Sort by agent name and model for consistent ordering
        data.sort(
            key=lambda x: (
                x.get("grouped_agent_name", x.get("Agent", "")),
                str(x.get("LLM base", [])),
            )
        )

        for entry in data:
            try:
                # Check if this agent should be excluded from this table type
                agent_name = entry.get("grouped_agent_name", entry.get("Agent", ""))
                if should_exclude_agent(agent_name, table_key):
                    continue

                include_submission_id = not args.no_submission_id
                if table_key == "overall":
                    row = table_info["generator"](
                        entry,
                        include_ci=include_ci,
                        include_submission_id=include_submission_id,
                    )
                else:
                    row = generate_task_specific_row(
                        entry,
                        table_info["score_keys"],
                        table_info["cost_keys"],
                        include_ci=include_ci,
                        include_submission_id=include_submission_id,
                    )
                print(row)
            except Exception as e:
                agent_name = entry.get("Agent", "Unknown")
                print(f"% Error generating row for {agent_name}: {e}")
                raise


if __name__ == "__main__":
    main()
