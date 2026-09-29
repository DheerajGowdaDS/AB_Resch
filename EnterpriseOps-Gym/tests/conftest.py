"""Test configuration for the gym test suite.

Puts the gym root on ``sys.path`` so ``ablation``, ``benchmark``,
``orchestrators``, ``csm_env`` and ``csm_integration`` import identically no
matter which directory pytest is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

GYM_ROOT = Path(__file__).resolve().parents[1]
if str(GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(GYM_ROOT))
