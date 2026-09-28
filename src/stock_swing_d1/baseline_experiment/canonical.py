"""Canonical semantic serialization for the Phase 16C experiment layer.

Phase 16C introduces no serialization mechanism of its own: it reuses the one
already owned by the Phase 15D audit layer, so a Phase 16C fingerprint and an
upstream fingerprint are computed over byte-identical canonical JSON rules.
"""

from stock_swing_d1.backtest_results.canonical import (
    canonicalize_semantic_value,
    semantic_json_bytes,
)


__all__ = ["canonicalize_semantic_value", "semantic_json_bytes"]
