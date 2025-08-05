"""
Model name mapping for converting full model names to display names.

This mapping is used to convert verbose model names (often including paths, dates, etc.)
to clean, readable names for display in plots and leaderboards.
"""

import logging

logger = logging.getLogger(__name__)

# Model name mapping dictionary
MODEL_NAME_MAPPING = {
    # OpenAI models
    "o3-pro-2025-06-10": "o3-pro",
    "o3-2025-04-16": "o3",
    "o4-mini-2025-04-16": "o4-mini",
    "gpt-4.1-2025-04-14": "gpt-4.1",
    "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano",
    "gpt-4o-2024-11-20": "gpt-4o",
    "codex-mini-latest": "codex-mini",
    # Anthropic models
    "claude-opus-4-20250514": "claude-opus-4",
    "claude-sonnet-4-20250514": "claude-sonnet-4",
    "claude-3-5-haiku-20241022": "claude-3-5-haiku",
    "claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
    # Google models
    "gemini-2.5-pro": "gemini-2.5-pro-unpinned",
    "gemini-2.5-flash": "gemini-2.5-flash-unpinned",
    "gemini-2-flash": "gemini-2-flash-unpinned",
    "gemini-2.5-flash-lite-preview-06-17": "gemini-2.5-flash-lite",
    "gemma-3-27b": "gemma-3-27b-unpinned",
    "gemma-3n-e4b-it": "gemma-3n-e4b-it-unpinned",
    # XAI models
    "grok-3": "grok-3-unpinned",
    "grok-3-mini": "grok-3-mini-unpinned",
    # Microsoft models
    "phi-4-reasoning": "phi-4-reasoning-unpinned",
    "phi-4-reasoning-plus": "phi-4-reasoning-plus-unpinned",
    # Alibaba models
    "qwen3-8b": "qwen3-8b-unpinned",
    "qwen3-235b": "qwen3-235b-unpinned",
    "qwq-32b": "qwq-32b-unpinned",
    # DeepSeek models
    "deepseek-v3-0324": "deepseek-v3",
    "deepseek-r1-0528": "deepseek-r1",
    # Meta models
    "llama-4-scout": "llama-4-scout-unpinned",
    "llama-4-maverick": "llama-4-maverick-unpinned",
    # Mistral models
    "mistral-large-2024-11": "mistral-large",
    "mistral-medium-3-2025-05": "mistral-medium-3",
    "mistral-small-2503": "mistral-small",
    "mistral-codestral-2025-01": "mistral-codestral",
    "mistral-devstral": "mistral-devstral-unpinned",
    # Perplexity models
    "perplexity-sonar": "perplexity-sonar-unpinned",
    "perplexity-sonar-pro": "perplexity-sonar-pro-unpinned",
    "perplexity-sonar-reasoning": "perplexity-sonar-reasoning-unpinned",
    "perplexity-sonar-reasoning-pro": "perplexity-sonar-reasoning-pro-unpinned",
    "perplexity-sonar-deep-research": "perplexity-sonar-deep-research-unpinned",
    # Additional long model names found in data
    # Keep dates only for old versions to distinguish from current
    "gpt-4o-2024-08-06": "gpt-4o-old-aug24",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini-old-jul24",
    # Remove path prefixes and shorten recent versions
    "claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
    "gpt-4o-2024-11-20": "gpt-4o",
    "claude-3-5-haiku-20241022": "claude-3-5-haiku",
    "models/gemini-2.5-flash-preview-05-20": "gemini-2.5-flash",
    "gemini/gemini-2.5-flash-preview-05-20": "gemini-2.5-flash",
    "models/gemini-2.5-pro-preview-05-06": "gemini-2.5-pro",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama-4-scout",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "llama-4-maverick",
    "anthropic/claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
    "anthropic/claude-sonnet-4-20250514": "claude-sonnet-4",
    "anthropic/claude-3-5-haiku-20241022": "claude-3-5-haiku",
    # Mark unpinned versions
    "openai/gpt-4.1": "gpt-4.1-unpinned",
    "openai/gpt-4.1-2025-04-14": "gpt-4.1",
    "openai/gpt-4.1-nano-2025-04-14": "gpt-4.1-nano",
    "openai/gpt-4.1-nano": "gpt-4.1-nano-unpinned",
    "openai/gpt-4.1-mini": "gpt-4.1-mini-unpinned",
    "openai/o3-2025-04-16": "o3",
    "openai/o3-mini": "o3-mini-unpinned",
    "sonar-deep-research": "sonar-deep-unpinned",
    "perplexity/sonar-deep-research": "sonar-deep-unpinned",
    "deepseek-ai/DeepSeek-V3": "deepseek-v3-unpinned",
    "gemini/gemini-2.0-flash": "gemini-2.0-flash-unpinned",
    "gemini/gemini-2.5-pro": "gemini-2.5-pro-unpinned",
    "openai/gpt-4o": "gpt-4o-unpinned",
    "gpt-3.5-turbo-0125": "gpt-3.5-turbo",
}


def get_model_shortname(model_name: str) -> str:
    """Convert full model names to their short display names.

    Args:
        model_name: The full model name as it appears in the data

    Returns:
        The shortened display name for the model
    """
    # Log when we perform shortening for auditing
    original_name = model_name
    shortened_name = MODEL_NAME_MAPPING.get(model_name, model_name)
    if original_name != shortened_name:
        logger.info(f"Model name shortened: '{original_name}' -> '{shortened_name}'")

    return shortened_name
