"""External earnings-provider sample acceptance scaffold.

No provider-specific parsing is implemented in Phase 6C.5.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.provider_acceptance
_BLOCKED = (
    "BLOCKED_BY_PROVIDER: external earnings-provider historical sample not yet available"
)


@pytest.fixture
def external_earnings_sample_path() -> Path:
    """Locate the future licensed sample without assuming its fields or format."""

    configured = os.environ.get("EXTERNAL_EARNINGS_SAMPLE_PATH")
    if not configured:
        pytest.skip(_BLOCKED)
    path = Path(configured)
    if not path.is_file():
        pytest.fail(f"configured EXTERNAL_EARNINGS_SAMPLE_PATH does not exist: {path}")
    pytest.fail(
        "External earnings-provider sample is configured, but its provider adapter is "
        "intentionally not implemented in Phase 6C.5"
    )


def test_external_earnings_f01_current_company_full_revision_chain(
    external_earnings_sample_path: Path,
) -> None:
    """A current company sample must expose its complete revision chain."""


def test_external_earnings_f02_ticker_change_identity_mapping(
    external_earnings_sample_path: Path,
) -> None:
    """A ticker change must retain stable canonical identity."""


def test_external_earnings_f03_former_or_delisted_security_is_present(
    external_earnings_sample_path: Path,
) -> None:
    """The historical sample must include a former or delisted security."""


def test_external_earnings_f04_first_known_future_date_is_reconstructable(
    external_earnings_sample_path: Path,
) -> None:
    """The first future schedule known at T must be reconstructable."""


def test_external_earnings_f05_historical_revision_chain_is_preserved(
    external_earnings_sample_path: Path,
) -> None:
    """Earlier provider revisions must remain append-only records."""


def test_external_earnings_f06_timing_has_point_in_time_history(
    external_earnings_sample_path: Path,
) -> None:
    """Timing changes must carry their own historical knowledge times."""


def test_external_earnings_f07_timestamp_precision_is_explicit(
    external_earnings_sample_path: Path,
) -> None:
    """Timestamp versus date-only precision must be explicit."""


def test_external_earnings_f08_revision_order_is_deterministic(
    external_earnings_sample_path: Path,
) -> None:
    """The sample must supply enough information for deterministic replay."""


def test_external_earnings_f09_cancellation_and_reinstatement_are_representable(
    external_earnings_sample_path: Path,
) -> None:
    """Cancellation and reinstatement must normalize to canonical transitions."""


def test_external_earnings_f10_provider_corrections_are_distinguishable(
    external_earnings_sample_path: Path,
) -> None:
    """Provider corrections must remain distinct from schedule revisions."""
