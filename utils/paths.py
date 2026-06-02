"""Project root resolution — all paths relative to zeroshift package root."""
import os


def project_root() -> str:
    """Return absolute path to zeroshift/ (parent of utils/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
