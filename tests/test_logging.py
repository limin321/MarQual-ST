"""Terminal: progress + warnings only. Log file: everything, including DEBUG details and library warnings."""
import subprocess
import sys

SCRIPT = """
import sys, warnings
from marqual_st._logging import configure_logging, get_logger
configure_logging("INFO", sys.argv[1])
log = get_logger("marqual_st.test")
log.info("step header")
log.debug("full result table")
log.warning("needs attention")
warnings.warn("a library deprecation notice", FutureWarning)
"""


def test_terminal_short_log_file_complete(tmp_path):
    log_file = tmp_path / "marqual_st.log"
    run = subprocess.run([sys.executable, "-c", SCRIPT, str(log_file)], capture_output=True, text=True, check=True)
    terminal = run.stdout + run.stderr
    assert "step header" in terminal and "WARNING: needs attention" in terminal
    assert "full result table" not in terminal and "deprecation notice" not in terminal
    text = log_file.read_text()
    for s in ("step header", "full result table", "needs attention", "FutureWarning: a library deprecation notice"):
        assert s in text, s


def test_reconfigure_no_duplicate_handlers(tmp_path):
    import logging

    from marqual_st._logging import configure_logging
    configure_logging("INFO", tmp_path / "a.log")
    configure_logging("INFO")
    handlers = logging.getLogger("marqual_st").handlers
    assert sum(isinstance(h, logging.FileHandler) for h in handlers) == 0
    assert sum(isinstance(h, logging.StreamHandler) for h in handlers) == 1
    logging.captureWarnings(False)
