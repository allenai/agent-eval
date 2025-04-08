from importlib.metadata import version as get_version

__version__ = get_version("agent-eval")

from .processor import score_directory
from .summary import compute_summary_statistics
from .upload import upload_folder_to_hf, upload_summary_to_hf

__all__ = [
    "score_directory",
    "compute_summary_statistics",
    "upload_folder_to_hf",
    "upload_summary_to_hf",
]
