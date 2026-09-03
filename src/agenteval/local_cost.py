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
    # costs from https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart
    "moonshotai/kimi-k2.5-0127": CostPerTokenWithCache(
        input_cost_per_token=6e-07,
        output_cost_per_token=3e-06,
        cache_read_input_token_cost=1e-07,
    ),
}
