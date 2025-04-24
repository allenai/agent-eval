from importlib.metadata import version as get_version

__version__ = get_version("agent-eval")

from .score import process_eval_logs
from .summary import compute_summary_statistics
from .upload import upload_folder_to_hf, upload_summary_to_hf

__all__ = [
    "process_eval_logs",
    "compute_summary_statistics",
    "upload_folder_to_hf",
    "upload_summary_to_hf",
]
