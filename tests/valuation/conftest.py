"""Deterministic fixtures for the shared allocation-boundary valuation."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_swing_d1.portfolio.portfolio_state_models import (
    OpenPosition,
    PendingSettlement,
    PortfolioState,
)
from stock_swing_d1.portfolio.portfolio_events import (
    AppliedEventFingerprint,
    PortfolioEventKind,
)
from stock_swing_d1.provenance import ArtifactRef
from stock_swing_d1.valuation import PortfolioValuationMark


SESSION = date(2026, 8, 3)
ASSET_A = "NORGATE:1001"
ASSET_B = "NORGATE:2002"


@pytest.fixture
def artifact_ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_type="market_data",
        schema_version="stock_bars_v0_1",
        content_sha256="b" * 64,
    )


@pytest.fixture
def make_mark(artifact_ref):
    def factory(
        *,
        security_id: str = ASSET_A,
        session: date = SESSION,
        close: Decimal = Decimal("100"),
        source_artifact_ref: ArtifactRef | None = None,
        **overrides,
    ) -> PortfolioValuationMark:
        return PortfolioValuationMark(
            security_id=security_id,
            session=session,
            close=close,
            source_artifact_ref=(
                artifact_ref
                if source_artifact_ref is None
                else source_artifact_ref
            ),
            **overrides,
        )

    return factory


@pytest.fixture
def make_position():
    def factory(
        *,
        asset_id: str = ASSET_A,
        quantity: int = 10,
        entry_price: Decimal = Decimal("90"),
        entry_session: date = date(2026, 7, 31),
    ) -> OpenPosition:
        return OpenPosition(
            asset_id=asset_id,
            quantity=quantity,
            entry_session=entry_session,
            entry_price=entry_price,
            entry_execution_id=f"ENTRY-{asset_id}",
            entry_execution_cost=Decimal("0"),
            cost_basis=entry_price * quantity,
        )

    return factory


@pytest.fixture
def make_settlement():
    def factory(
        *,
        settlement_id: str = "S1",
        asset_id: str = ASSET_B,
        amount: Decimal = Decimal("500"),
    ) -> PendingSettlement:
        return PendingSettlement(
            settlement_id=settlement_id,
            source_execution_id=f"EXEC-{settlement_id}",
            asset_id=asset_id,
            amount=amount,
            trade_session=date(2026, 7, 31),
            settlement_session=date(2026, 8, 4),
        )

    return factory


@pytest.fixture
def make_state():
    def factory(
        *,
        settled_cash: Decimal = Decimal("1000"),
        open_positions: tuple[OpenPosition, ...] = (),
        pending_settlements: tuple[PendingSettlement, ...] = (),
        as_of_session: date | None = SESSION,
    ) -> PortfolioState:
        applied = tuple(
            AppliedEventFingerprint(
                event_kind=PortfolioEventKind.EXECUTION,
                event_id=position.entry_execution_id,
                payload_sha256="a" * 64,
            )
            for position in open_positions
        )
        return PortfolioState(
            settled_cash=settled_cash,
            open_positions=open_positions,
            pending_settlements=pending_settlements,
            applied_events=applied,
            as_of_session=as_of_session,
            state_version=1 if as_of_session is not None else 0,
        )

    return factory
