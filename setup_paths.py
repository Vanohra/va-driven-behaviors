"""
Path Configuration — VA Driven Behaviors Project

Place the following files in the project root (same directory as run_offline.py):
  - test_emotions.py      (JointCAM inference functions)
  - models/               (directory containing the .pt model file)

No external path configuration is needed. Python will find test_emotions.py
automatically when scripts are run from the project root.
"""

# This file is kept for backwards compatibility but does nothing.

def setup() -> None:
    """No-op — kept for backwards compatibility."""
    pass


def check() -> bool:
    """Return True if test_emotions can be imported."""
    try:
        import test_emotions  # noqa: F401
        return True
    except ImportError:
        return False
