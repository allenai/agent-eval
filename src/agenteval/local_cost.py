from litellm.utils import CostPerToken

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
}
