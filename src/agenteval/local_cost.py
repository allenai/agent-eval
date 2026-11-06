from litellm.utils import CostPerToken
from pydantic import BaseModel

# even where these exist in https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
# calling cost_per_token does not return a cost, perhaps due to the associated provider
# key represents model name as found in inspect model_usage
CUSTOM_PRICING = {
    # costs are from https://www.together.ai/pricing
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": CostPerToken(
        input_cost_per_token=1.8e-07, output_cost_per_token=5.9e-07
    ),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": CostPerToken(
        input_cost_per_token=2.7e-07, output_cost_per_token=8.5e-07
    ),
    "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct": CostPerToken(
        input_cost_per_token=1.8e-07, output_cost_per_token=5.9e-07
    ),
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": CostPerToken(
        input_cost_per_token=2.7e-07, output_cost_per_token=8.5e-07
    ),
    "deepseek-ai/DeepSeek-V3": CostPerToken(
        input_cost_per_token=1.25e-06, output_cost_per_token=1.25e-06
    ),
    "deepseek-ai/DeepSeek-R1": CostPerToken(
        input_cost_per_token=3e-06, output_cost_per_token=7e-06
    ),
    # cost is for xai/grok-3 https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
    "grok-3": CostPerToken(input_cost_per_token=3e-06, output_cost_per_token=1.5e-05),
    # using https://artificialanalysis.ai/models/qwen3-8b-instruct
    "Qwen3-8B-SciQA-SFT": CostPerToken(
        input_cost_per_token=1.8e-07, output_cost_per_token=7e-07
    ),
    "akariasai/os_8b": CostPerToken(
        input_cost_per_token=1.8e-07, output_cost_per_token=1.8e-07
    ),
}


class CostPerTokenWithCache(BaseModel):
    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_input_token_cost: float
    cache_write_input_token_cost: float | None = None
    # Anthropic-style tiered pricing: a request whose prompt (input + cache read
    # + cache write tokens) exceeds the threshold is billed at long_context rates.
    long_context_threshold_tokens: int | None = None
    long_context: "CostPerTokenWithCache | None" = None


CLAUDE_SONNET_5_5 = CostPerTokenWithCache(
    input_cost_per_token=2e-06,
    output_cost_per_token=1e-05,
    cache_read_input_token_cost=2e-07,
    cache_write_input_token_cost=2.5e-06,
)

CLAUDE_HAIKU_5_5 = CostPerTokenWithCache(
    input_cost_per_token=1e-07,
    output_cost_per_token=5e-07,
    cache_read_input_token_cost=1e-08,
    cache_write_input_token_cost=1.25e-07,
    long_context_threshold_tokens=100_000,
    long_context=CostPerTokenWithCache(
        input_cost_per_token=5e-07,
        output_cost_per_token=2.5e-06,
        cache_read_input_token_cost=5e-08,
        cache_write_input_token_cost=6.25e-07,
    ),
)


# Like CUSTOM_PRICING, but for models that also have a cache read discount.
# cost_per_token with usage_object doesn't work for these models in litellm 1.75.8,
# so costs are computed manually in compute_model_cost.
# key represents model name as found in inspect model_usage
CUSTOM_PRICING_WITH_CACHE = {
    # DeepSeek V4 Pro preview pricing before the GA pricing change took effect
    # on 2026-08-16 at 16:00 UTC. DeepSeek accepted `deepseek-v4-pro` as the
    # request model while Inspect logs contain this provider response model name.
    # Historical pricing:
    # https://github.com/BerriAI/litellm/blob/0059b497f44d8ad2ab8dae5e17fbd8df0cfcdb11/model_prices_and_context_window.json
    "deepseek/deepseek-v4pro-preview": CostPerTokenWithCache(
        input_cost_per_token=4.35e-07,
        output_cost_per_token=8.7e-07,
        cache_read_input_token_cost=3.625e-09,
        cache_write_input_token_cost=0.0,
    ),
    # Costs from https://platform.claude.com/docs/en/about-claude/pricing
    "claude-3-5-haiku-20241022": CostPerTokenWithCache(
        input_cost_per_token=8e-07,
        output_cost_per_token=4e-06,
        cache_read_input_token_cost=8e-08,
        cache_write_input_token_cost=1e-06,
    ),
    "anthropic/claude-3-5-haiku-20241022": CostPerTokenWithCache(
        input_cost_per_token=8e-07,
        output_cost_per_token=4e-06,
        cache_read_input_token_cost=8e-08,
        cache_write_input_token_cost=1e-06,
    ),
    # Costs from https://platform.claude.com/docs/en/about-claude/pricing
    # Inspect usage does not separate 5-minute from 1-hour cache writes, so cache
    # writes are priced at the 5-minute (1.25x input) rate.
    **{
        name: CLAUDE_SONNET_5_5
        for name in ("claude-sonnet-5-5", "anthropic/claude-sonnet-5-5")
    },
    **{
        name: CLAUDE_HAIKU_5_5
        for name in ("claude-haiku-5-5", "anthropic/claude-haiku-5-5")
    },
    # costs from https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart
    "moonshotai/kimi-k2.5-0127": CostPerTokenWithCache(
        input_cost_per_token=6e-07,
        output_cost_per_token=3e-06,
        cache_read_input_token_cost=1e-07,
    ),
}
