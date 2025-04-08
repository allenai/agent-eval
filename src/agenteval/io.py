import os
import tempfile
from pathlib import Path


def atomic_write_file(
    path: str | Path,
    content: str,
    encoding: str = "utf-8",
) -> None:
    """
    Write the given content string to `path` atomically.

    Writes to a temporary file in the same directory, fsyncs, then replaces.
    """
    p = Path(path)
    # write to temp file in same directory
    tmp = tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding=encoding, dir=str(p.parent)
    )
    try:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    # replace target atomically; if successful, attempt to delete any leftover temp
    try:
        os.replace(tmp.name, str(p))
    except Exception:
        # keep temp file for debugging if replace fails
        raise
    else:
        # atomic replace removed tmp.name; if it still exists, remove it
        try:
            os.remove(tmp.name)
        except FileNotFoundError:
            pass
