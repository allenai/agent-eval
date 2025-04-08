from pathlib import Path

from .config import load_suite_config
from .models import EvalResult
from .score import process_eval_logs


def score_directory(
    log_dir: str,
    config_path: str | None = None,
    split: str | None = None,
    eval_filename: str = "agenteval.json",
) -> EvalResult:
    """
    Load or create an EvalResult for the given log directory, process the
    logs, and optionally save the updated JSON back to disk.

    Args:
        log_dir: path to the folder containing logs and existing JSON.
        config_path: path to suite config (required if no existing JSON).
        split: data split name (required if no existing JSON).
        eval_filename: JSON filename in log_dir (default: 'agenteval.json').

    Returns:
        EvalResult: the processed result object with results and specs populated.

    Raises:
        ValueError: if no existing JSON and config_path or split is None.
    """
    json_path = Path(log_dir) / eval_filename
    eval_result: EvalResult | None = None

    if json_path.exists():
        try:
            raw = json_path.read_text(encoding="utf-8")
            eval_result = EvalResult.model_validate_json(raw)
        except Exception as e:
            raise ValueError(
                f"Failed to load existing '{eval_filename}' at {json_path}: {e}"
            )

    if not eval_result:
        if config_path is None or split is None:
            raise ValueError(
                "config_path and split must be provided when no existing result JSON"
            )
        eval_result = EvalResult(
            suite_config=load_suite_config(config_path),
            split=split,
        )

    # process evaluation logs
    task_results, eval_specs = process_eval_logs(log_dir)
    eval_result.eval_specs = eval_specs
    eval_result.results = task_results

    return eval_result
