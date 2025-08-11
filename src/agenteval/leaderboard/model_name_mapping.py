"""
Model name mapping for converting full model names to display names.

This mapping is used to convert verbose model names (often including paths, dates, etc.)
to clean, readable names for display in plots and leaderboards.
"""

# Display names for leaderboard
LB_MODEL_NAME_MAPPING = {
    # OpenAI models
    "o3-pro-2025-06-10": "o3 Pro (2025-06)",
    "o3-2025-04-16": "o3 (2025-04)",
    "o4-mini-2025-04-16": "o4 Mini (2025-04)",
    "gpt-4.1-2025-04-14": "GPT-4.1 (2025-04)",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 Nano (2025-04)",
    "gpt-4o-2024-11-20": "GPT-4o (2024-11)",
    "codex-mini-latest": "Codex Mini (unpinned)",
    # Anthropic models
    "claude-opus-4-20250514": "Claude Opus 4 (2025-05)",
    "claude-sonnet-4-20250514": "Claude Sonnet 4 (2025-05)",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku (2024-10)",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet (2025-02)",
    # Google models
    "gemini-2.5-pro": "Gemini 2.5 Pro (unpinned)",
    "gemini-2.5-flash": "Gemini 2.5 Flash (unpinned)",
    "gemini-2-flash": "Gemini 2 Flash (unpinned)",
    "gemini-2.5-flash-lite-preview-06-17": "Gemini 2.5 Flash Lite (2024-06)",
    "gemma-3-27b": "Gemma 3 27B (unpinned)",
    "gemma-3n-e4b-it": "Gemma 3N E4B IT (unpinned)",
    # XAI models
    "grok-3": "Grok 3 (unpinned)",
    "grok-3-mini": "Grok 3 Mini (unpinned)",
    # Microsoft models
    "phi-4-reasoning": "Phi-4 Reasoning (unpinned)",
    "phi-4-reasoning-plus": "Phi-4 Reasoning Plus (unpinned)",
    # Alibaba models
    "qwen3-8b": "Qwen 3 8B (unpinned)",
    "qwen3-235b": "Qwen 3 235B (unpinned)",
    "qwq-32b": "QwQ 32B (unpinned)",
    # DeepSeek models
    "deepseek-v3-0324": "DeepSeek V3 (2025-03)",
    "deepseek-r1-0528": "DeepSeek R1 (2025-05)",
    # Meta models
    "llama-4-scout": "Llama 4 Scout (unpinned)",
    "llama-4-maverick": "Llama 4 Maverick (unpinned)",
    # Mistral models
    "mistral-large-2024-11": "Mistral Large (2024-11)",
    "mistral-medium-3-2025-05": "Mistral Medium 3 (2025-05)",
    "mistral-small-2503": "Mistral Small (2025-03)",
    "mistral-codestral-2025-01": "Mistral Codestral (2025-01)",
    "mistral-devstral": "Mistral Devstral (unpinned)",
    # Perplexity models
    "perplexity-sonar": "Perplexity Sonar (unpinned)",
    "perplexity-sonar-pro": "Perplexity Sonar Pro (unpinned)",
    "perplexity-sonar-reasoning": "Perplexity Sonar Reasoning (unpinned)",
    "perplexity-sonar-reasoning-pro": "Perplexity Sonar Reasoning Pro (unpinned)",
    "perplexity-sonar-deep-research": "Perplexity Sonar Deep Research (unpinned)",
    # Additional long model names found in data
    # Keep dates only for old versions to distinguish from current
    "gpt-4o-2024-08-06": "GPT-4o (2024-08)",
    "gpt-4o-mini-2024-07-18": "GPT-4o Mini (2024-07)",
    # Remove path prefixes and shorten recent versions
    "models/gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash (2024-05)",
    "gemini/gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash (2024-05)",
    "models/gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro (2024-05)",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama 4 Scout 17B (unpinned)",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama 4 Maverick 17B (unpinned)",
    "anthropic/claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet (2025-02)",
    "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4 (2025-05)",
    "anthropic/claude-3-5-haiku-20241022": "Claude 3.5 Haiku (2024-10)",
    # Mark unpinned versions
    "openai/gpt-4.1": "GPT-4.1 (unpinned)",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1 (2025-04)",
    "openai/gpt-4.1-nano-2025-04-14": "GPT-4.1 Nano (2025-04)",
    "openai/gpt-4.1-nano": "GPT-4.1 Nano (unpinned)",
    "openai/gpt-4.1-mini": "GPT-4.1 Mini (unpinned)",
    "openai/o3-2025-04-16": "o3 (2025-04)",
    "openai/o3-mini": "o3 Mini (unpinned)",
    "sonar-deep-research": "Sonar Deep Research (unpinned)",
    "perplexity/sonar-deep-research": "Sonar Deep Research (unpinned)",
    "deepseek-ai/DeepSeek-V3": "DeepSeek V3 (unpinned)",
    "gemini/gemini-2.0-flash": "Gemini 2.0 Flash (unpinned)",
    "gemini/gemini-2.5-pro": "Gemini 2.5 Pro (unpinned)",
    "openai/gpt-4o": "GPT-4o (unpinned)",
    "gpt-3.5-turbo-0125": "GPT-3.5 Turbo (2025-01)",
}
