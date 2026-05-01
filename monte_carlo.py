from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SOURCE = Path(__file__).with_name("monte_carlo-2.py")
_SPEC = importlib.util.spec_from_file_location("_monte_carlo_impl", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load Monte Carlo implementation from {_SOURCE}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

InputSpec = _MODULE.InputSpec
MCResult = _MODULE.MCResult
run_monte_carlo = _MODULE.run_monte_carlo

__all__ = ["InputSpec", "MCResult", "run_monte_carlo"]
