"""Task-local guards for optional plotting packages on heterogeneous GPU images."""
from __future__ import annotations

import importlib.machinery
import sys
import types


def guard_optional_visualization_imports() -> bool:
    """Stub unused pandas/seaborn only when the remote pandas wheel is broken.

    This experiment never uses pandas or seaborn.  Some unrelated modules import
    them eagerly while Ultralytics builds its registry.  The guard is local to
    this Python process and leaves the environment/repository dependencies alone.
    Returns True when the compatibility path was needed.
    """
    try:
        import pandas  # noqa: F401
        return False
    except (ImportError, ModuleNotFoundError):
        for name in list(sys.modules):
            if name == "pandas" or name.startswith("pandas.") or name == "seaborn" or name.startswith("seaborn."):
                sys.modules.pop(name, None)
        for name in ("pandas", "seaborn"):
            placeholder = types.ModuleType(name)
            placeholder.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            placeholder.__rs_mid_optional_placeholder__ = True
            sys.modules[name] = placeholder
        return True


__all__ = ["guard_optional_visualization_imports"]
