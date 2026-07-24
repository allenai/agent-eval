import importlib
from importlib.metadata import version as get_version

__version__ = get_version("agent-eval")

# The scoring/leaderboard entrypoints pull in inspect-ai (and other heavy,
# scoring-only deps). Expose them lazily so that merely importing `agenteval`
# — or an inspect-free submodule like `agenteval.config` — does not force
# inspect to be installed. See allenai/gas2own#346.
_LAZY_ATTRS = {
    "process_eval_logs": "agenteval.score",
    "compute_summary_statistics": "agenteval.summary",
    "upload_folder_to_hf": "agenteval.leaderboard.upload",
}

__all__ = ["process_eval_logs", "compute_summary_statistics", "upload_folder_to_hf"]


def __getattr__(name: str):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)


def __dir__():
    return sorted(list(globals()) + __all__)
