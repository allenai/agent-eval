from importlib.metadata import version as get_version

__version__ = get_version("agent-eval")

from .leaderboard.upload import upload_folder_to_hf
from .score import process_eval_logs
from .summary import compute_summary_statistics

__all__ = [
    "process_eval_logs",
    "compute_summary_statistics",
    "upload_folder_to_hf",
]
