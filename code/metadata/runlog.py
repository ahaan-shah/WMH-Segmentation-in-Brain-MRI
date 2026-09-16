"""Run logging to console and file (CLAUDE.md Section 5.6).

Every pipeline script logs one line per subject to both stdout and
`<stage>/outputs/logs/<script>_<date>.log`. The rule exists because when
something looks wrong in Week 6, the Week 2 logs are the only evidence of what
each subject actually did — and `print()` leaves none.

Log files are named by date, not by timestamp, so re-running a script the same
day overwrites rather than accumulating (Section 5.4, idempotence). Nothing
here appends across runs.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logging(script_name: str, output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure a logger writing to both the console and a dated log file.

    `script_name` should be the bare module name (e.g. "run_head_mask"), and
    `output_dir` the stage's outputs/ directory; the log lands in
    `output_dir/logs/<script_name>_<YYYY-MM-DD>.log`.
    """
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{script_name}_{date.today().isoformat()}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(level)
    logger.handlers.clear()  # idempotent: re-running in one session must not double-log
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    file_handler = logging.FileHandler(log_path, mode="w")  # overwrite, never append
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("logging to %s", log_path)
    return logger
