"""Production Phase 15C execution-event adapters for historical backtests."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date
from decimal import Decimal
from typing import TypeVar

from pydantic import ValidationError

from stock_swing_d1.execution.costs import (
    AdministrativeExitPriceQuote,
    AdministrativeExitPricingService,
    BacktestExecutionCostService,
    ExecutionCostPolicyRef,
    ExecutionCostQuote,
    ExecutionCostSide,
    ExecutionCostValidationError,
    ExecutionIdentifierService,
    HistoricalUsEquitySettlementResolver,
    SettlementPolicyModel,
    SettlementResolution,
)
from stock_swing_d1.execution.entry import (
    EntryExecutionDecision,
    EntryExecutionStatus,
)
from stock_swing_d1.execution.open_position_exit import (
    ExitBoundary,
    ExitPrerequisiteStatus,
    OpenPositionExitDecision,
    OpenPositionExitReason,
)
from stock_swing_d1.portfolio.portfolio_events import (
    ExecutionSide,
    PortfolioExecutionEvent,
)
from stock_swing_d1.portfolio.portfolio_state_models import OpenPosition

from stock_swing_d1.backtester.models import HistoricalBacktestEntryIntent


_PROTECTIVE_EXIT_REASONS = frozenset(
    {
        OpenPositionExitReason.GAP_THROUGH_STOP,
        OpenPositionExitReason.GAP_THROUGH_TARGET,
        OpenPositionExitReason.STOP_LOSS,
        OpenPositionExitReason.TAKE_PROFIT,
    }
)
_ADMINISTRATIVE_EXIT_REASONS = frozenset(
    {
        OpenPositionExitReason.EARNINGS_FORCED_EXIT,
        OpenPositionExitReason.MAX_HOLDING,
    }
)
_DataclassT = TypeVar("_DataclassT")


class BacktestExecutionAdapterValidationError(ValueError):
    """A production Phase 15C event-adapter contract was violated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _canonical_dataclass(
    value: object, expected_type: type[_DataclassT]
) -> _DataclassT:
    if type(value) is not expected_type:
        raise TypeError(f"must be exactly {expected_type.__name__}")
    return expected_type(
        **{
            descriptor.name: getattr(value, descriptor.name)
            for descriptor in fields(expected_type)
        }
    )


def _canonical_policy_ref(value: object) -> ExecutionCostPolicyRef:
    return _canonical_dataclass(value, ExecutionCostPolicyRef)


def _canonical_cost_quote(value: object) -> ExecutionCostQuote:
    return _canonical_dataclass(value, ExecutionCostQuote)


def _current_cost_policy_ref(
    service: BacktestExecutionCostService,
) -> ExecutionCostPolicyRef:
    return _canonical_policy_ref(service.policy_ref)


def _current_pricing_policy_ref(
    service: AdministrativeExitPricingService,
) -> ExecutionCostPolicyRef:
    return _canonical_policy_ref(service.policy_ref)


def _canonical_event(event: PortfolioExecutionEvent) -> PortfolioExecutionEvent:
    rebuilt = PortfolioExecutionEvent.model_validate(
        event.model_dump(mode="python")
    )
    if rebuilt != event:
        raise ValueError("event is not canonical")
    return rebuilt


@dataclass(frozen=True, slots=True)
class BacktestBuyExecutionEventAdapter:
    """Bind an authoritative Phase 9 BUY decision to one Phase 13 event."""

    execution_cost_service: BacktestExecutionCostService
    identifier_service: ExecutionIdentifierService
    policy_ref: ExecutionCostPolicyRef = field(init=False)

    def __post_init__(self) -> None:
        try:
            if not isinstance(
                self.execution_cost_service, BacktestExecutionCostService
            ):
                raise TypeError(
                    "execution_cost_service must be BacktestExecutionCostService"
                )
            if not callable(getattr(self.execution_cost_service, "quote", None)):
                raise TypeError("execution_cost_service must provide quote")
            policy_ref = _current_cost_policy_ref(self.execution_cost_service)
        except (AttributeError, TypeError, ValueError) as error:
            raise BacktestExecutionAdapterValidationError(
                "INVALID_EXECUTION_COST_SERVICE",
                "BUY adapter requires a canonical Phase 15C cost service",
            ) from error

        if not isinstance(self.identifier_service, ExecutionIdentifierService) or any(
            not callable(getattr(self.identifier_service, operation, None))
            for operation in ("buy_source_order_id", "buy_execution_id")
        ):
            raise BacktestExecutionAdapterValidationError(
                "INVALID_IDENTIFIER_SERVICE",
                "BUY adapter requires the deterministic Phase 15C identifier service",
            )
        object.__setattr__(self, "policy_ref", policy_ref)

    def _assert_policy_unchanged(self) -> None:
        try:
            current = _current_cost_policy_ref(self.execution_cost_service)
        except (AttributeError, TypeError, ValueError) as error:
            raise BacktestExecutionAdapterValidationError(
                "EXECUTION_COST_POLICY_CHANGED",
                "the execution-cost service policy reference is no longer canonical",
            ) from error
        if current != self.policy_ref:
            raise BacktestExecutionAdapterValidationError(
                "EXECUTION_COST_POLICY_CHANGED",
                "the execution-cost service policy changed after adapter construction",
            )

    def build_buy_event(
        self,
        *,
        session: date,
        intent: HistoricalBacktestEntryIntent,
        entry_execution: EntryExecutionDecision,
    ) -> PortfolioExecutionEvent:
        """Return one exact-cost BUY event, failing closed on provenance drift."""

        try:
            if type(session) is not date:
                raise TypeError("session must be a genuine date")
            if type(intent) is not HistoricalBacktestEntryIntent:
                raise TypeError("intent must be HistoricalBacktestEntryIntent")
            decision = _canonical_dataclass(
                entry_execution, EntryExecutionDecision
            )
            sized = intent.candidate_decision.sized_pending_entry
            if sized is None:
                raise ValueError("intent lacks an authoritative fixed quantity")
            if (
                decision.status is not EntryExecutionStatus.EXECUTED
                or decision.planned_entry_session != session
                or intent.planned_entry_session != session
                or decision.security_id != intent.security_id
                or decision.requested_shares != sized.fixed_shares
                or decision.executed_shares != sized.fixed_shares
                or decision.execution_price is None
                or decision.execution_cost_quote is None
            ):
                raise ValueError(
                    "entry decision, intent, session, and fixed quantity disagree"
                )
            fill_price = Decimal(str(decision.execution_price))
            quote = _canonical_cost_quote(decision.execution_cost_quote)
            if (
                quote.policy_ref != self.policy_ref
                or quote.side is not ExecutionCostSide.BUY
                or quote.quantity != decision.executed_shares
                or quote.fill_price != fill_price
            ):
                raise ValueError("entry cost quote provenance does not match the BUY")
        except (AttributeError, TypeError, ValueError) as error:
            raise BacktestExecutionAdapterValidationError(
                "INVALID_BUY_EXECUTION_INPUT",
                "BUY adaptation requires one canonical executed Phase 9 decision",
            ) from error

        self._assert_policy_unchanged()
        try:
            requoted = _canonical_cost_quote(
                self.execution_cost_service.quote(
                    side=ExecutionCostSide.BUY,
                    quantity=decision.executed_shares,
                    fill_price=fill_price,
                )
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise BacktestExecutionAdapterValidationError(
                "BUY_COST_QUOTE_FAILED",
                "the Phase 15C cost service could not reproduce the BUY quote",
            ) from error
        self._assert_policy_unchanged()
        if requoted != quote:
            raise BacktestExecutionAdapterValidationError(
                "BUY_COST_QUOTE_MISMATCH",
                "the Phase 9 cost audit differs from the production BUY re-quote",
            )

        try:
            source_order_id = self.identifier_service.buy_source_order_id(
                allocation_session=intent.allocation_session,
                security_id=intent.security_id,
            )
            execution_id = self.identifier_service.buy_execution_id(
                entry_session=session,
                security_id=intent.security_id,
            )
            event = PortfolioExecutionEvent(
                execution_id=execution_id,
                source_order_id=source_order_id,
                session=session,
                asset_id=intent.security_id,
                side=ExecutionSide.BUY,
                quantity=decision.executed_shares,
                fill_price=fill_price,
                execution_cost=quote.execution_cost,
            )
            return _canonical_event(event)
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            raise BacktestExecutionAdapterValidationError(
                "INVALID_BUY_EXECUTION_EVENT",
                "the BUY could not be represented as a canonical Phase 13 event",
            ) from error


@dataclass(frozen=True, slots=True)
class TransientEntryExposure:
    """Task 5C-C: the Phase 15C-owned view of an entry executed on session T.

    Runtime-only and immutable.  It carries exactly the facts the SELL
    pipeline needs to adapt a same-session protective exit -- identity,
    quantity, entry session, entry execution ID and the entry fill price --
    and is constructed only from the authoritative BUY
    ``PortfolioExecutionEvent`` Phase 15C itself produced.  It is not an
    ``OpenPosition`` and carries no cost basis, realized P&L, or any other
    Phase 13 accounting fact: authoritative state changes only inside the
    single Phase 13 transition that applies the BUY and its SELL together.
    """

    security_id: str
    quantity: int
    entry_session: date
    entry_execution_id: str
    entry_fill_price: Decimal

    def __post_init__(self) -> None:
        if (
            type(self.security_id) is not str
            or not self.security_id
            or self.security_id != self.security_id.strip()
            or type(self.quantity) is not int
            or self.quantity <= 0
            or type(self.entry_session) is not date
            or type(self.entry_execution_id) is not str
            or not self.entry_execution_id
            or type(self.entry_fill_price) is not Decimal
            or not self.entry_fill_price.is_finite()
            or self.entry_fill_price <= 0
        ):
            raise BacktestExecutionAdapterValidationError(
                "INVALID_TRANSIENT_ENTRY_EXPOSURE",
                "a transient entry exposure requires canonical identity, "
                "positive quantity, a genuine entry session and a positive "
                "Decimal entry fill price",
            )

    @classmethod
    def from_buy_event(cls, event: PortfolioExecutionEvent) -> TransientEntryExposure:
        """Derive the exposure solely from one canonical authoritative BUY."""

        try:
            canonical = _canonical_event(event)
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            raise BacktestExecutionAdapterValidationError(
                "INVALID_TRANSIENT_ENTRY_EXPOSURE",
                "a transient entry exposure requires one canonical BUY event",
            ) from error
        if canonical.side is not ExecutionSide.BUY:
            raise BacktestExecutionAdapterValidationError(
                "INVALID_TRANSIENT_ENTRY_EXPOSURE",
                "a transient entry exposure can only be derived from a BUY",
            )
        return cls(
            security_id=canonical.asset_id,
            quantity=canonical.quantity,
            entry_session=canonical.session,
            entry_execution_id=canonical.execution_id,
            entry_fill_price=canonical.fill_price,
        )


@dataclass(frozen=True, slots=True)
class BacktestSellExecutionEventAdapter:
    """Bind an authoritative Phase 15B terminal decision to a Phase 13 SELL.

    Two public paths share one private pricing/cost/settlement/identifier
    pipeline: ``build_sell_event`` for a position open entering T (Phase 13
    ``OpenPosition``) and ``build_round_trip_sell_event`` (Task 5C-C) for an
    entry executed on T itself (``TransientEntryExposure``).  Neither path
    prices, costs, settles or identifies differently from the other.
    """

    execution_cost_service: BacktestExecutionCostService
    administrative_pricing_service: AdministrativeExitPricingService
    settlement_resolver: HistoricalUsEquitySettlementResolver
    identifier_service: ExecutionIdentifierService
    policy_ref: ExecutionCostPolicyRef = field(init=False)

    def __post_init__(self) -> None:
        try:
            if not isinstance(
                self.execution_cost_service, BacktestExecutionCostService
            ) or not callable(getattr(self.execution_cost_service, "quote", None)):
                raise TypeError("invalid execution-cost service")
            if not isinstance(
                self.administrative_pricing_service,
                AdministrativeExitPricingService,
            ) or not callable(
                getattr(self.administrative_pricing_service, "price", None)
            ):
                raise TypeError("invalid administrative-pricing service")
            cost_ref = _current_cost_policy_ref(self.execution_cost_service)
            pricing_ref = _current_pricing_policy_ref(
                self.administrative_pricing_service
            )
            if cost_ref != pricing_ref:
                raise ValueError("cost and pricing policies differ")
            if (
                self.execution_cost_service.policy.settlement.model
                is not SettlementPolicyModel.HISTORICAL_US_EQUITY_STANDARD_CYCLE
            ):
                raise ValueError("policy does not select historical settlement")
        except (AttributeError, TypeError, ValueError) as error:
            raise BacktestExecutionAdapterValidationError(
                "INVALID_SELL_POLICY_SERVICES",
                "SELL cost and administrative pricing require one canonical policy",
            ) from error

        if not isinstance(
            self.settlement_resolver, HistoricalUsEquitySettlementResolver
        ) or not callable(getattr(self.settlement_resolver, "resolve", None)):
            raise BacktestExecutionAdapterValidationError(
                "INVALID_SETTLEMENT_RESOLVER",
                "SELL adapter requires the historical U.S. equity resolver",
            )
        if not isinstance(self.identifier_service, ExecutionIdentifierService) or any(
            not callable(getattr(self.identifier_service, operation, None))
            for operation in (
                "sell_source_order_id",
                "sell_execution_id",
                "settlement_id",
            )
        ):
            raise BacktestExecutionAdapterValidationError(
                "INVALID_IDENTIFIER_SERVICE",
                "SELL adapter requires the deterministic Phase 15C identifier service",
            )
        object.__setattr__(self, "policy_ref", cost_ref)

    def _assert_policies_unchanged(self) -> None:
        try:
            cost_ref = _current_cost_policy_ref(self.execution_cost_service)
            pricing_ref = _current_pricing_policy_ref(
                self.administrative_pricing_service
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise BacktestExecutionAdapterValidationError(
                "EXECUTION_COST_POLICY_CHANGED",
                "a SELL policy reference is no longer canonical",
            ) from error
        if cost_ref != self.policy_ref or pricing_ref != self.policy_ref:
            raise BacktestExecutionAdapterValidationError(
                "EXECUTION_COST_POLICY_CHANGED",
                "a SELL policy changed after adapter construction",
            )

    def build_sell_event(
        self,
        *,
        decision: OpenPositionExitDecision,
        open_position: OpenPosition,
    ) -> PortfolioExecutionEvent:
        """Return one full-quantity SELL with exact price, cost, IDs, and settlement."""

        try:
            canonical_decision = _canonical_dataclass(
                decision, OpenPositionExitDecision
            )
            if type(open_position) is not OpenPosition:
                raise TypeError("open_position must be exactly OpenPosition")
            canonical_position = OpenPosition.model_validate(
                open_position.model_dump(mode="python")
            )
            if canonical_position != open_position:
                raise ValueError("open position is not canonical")
            if (
                canonical_decision.exit_required is not True
                or canonical_decision.exit_prerequisite_status
                is not ExitPrerequisiteStatus.READY
                or canonical_decision.security_id != canonical_position.asset_id
                or canonical_decision.entry_session
                != canonical_position.entry_session
                or type(canonical_decision.session) is not date
            ):
                raise ValueError("terminal decision and open position disagree")
        except (
            AttributeError,
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            raise BacktestExecutionAdapterValidationError(
                "INVALID_SELL_EXECUTION_INPUT",
                "SELL adaptation requires a canonical ready terminal "
                "decision and position",
            ) from error

        return self._build_full_exit_sell_event(
            decision=canonical_decision,
            quantity=canonical_position.quantity,
            entry_execution_id=canonical_position.entry_execution_id,
            allow_administrative=True,
        )

    def build_round_trip_sell_event(
        self,
        *,
        decision: OpenPositionExitDecision,
        exposure: TransientEntryExposure,
    ) -> PortfolioExecutionEvent:
        """Task 5C-C: adapt a same-session protective exit of an entry on T.

        The exposure is the Phase 15C view of the BUY executed earlier on the
        same session; the resulting SELL is an ordinary full-quantity SELL
        event that Phase 13 applies after that BUY inside the same
        transition.  Only protective reasons are admissible: an
        administrative close (earnings / max holding) cannot arise on an
        entry session and is rejected.
        """

        try:
            canonical_decision = _canonical_dataclass(
                decision, OpenPositionExitDecision
            )
            if type(exposure) is not TransientEntryExposure:
                raise TypeError("exposure must be exactly TransientEntryExposure")
            canonical_exposure = _canonical_dataclass(
                exposure, TransientEntryExposure
            )
            protective = canonical_decision.protective_exit_decision
            if (
                canonical_decision.exit_required is not True
                or canonical_decision.exit_prerequisite_status
                is not ExitPrerequisiteStatus.READY
                or type(canonical_decision.session) is not date
                or canonical_decision.session
                != canonical_decision.entry_session
                or canonical_decision.entry_session
                != canonical_exposure.entry_session
                or canonical_decision.security_id
                != canonical_exposure.security_id
                or canonical_decision.holding_session_number != 1
                or Decimal(str(protective.entry_price))
                != canonical_exposure.entry_fill_price
            ):
                raise ValueError(
                    "terminal entry-session decision and exposure disagree"
                )
            if canonical_decision.selected_reason not in _PROTECTIVE_EXIT_REASONS:
                raise ValueError(
                    "a same-session round trip requires a protective reason"
                )
        except (
            AttributeError,
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            raise BacktestExecutionAdapterValidationError(
                "INVALID_ROUND_TRIP_SELL_INPUT",
                "round-trip SELL adaptation requires a canonical ready "
                "protective entry-session decision and its transient exposure",
            ) from error

        return self._build_full_exit_sell_event(
            decision=canonical_decision,
            quantity=canonical_exposure.quantity,
            entry_execution_id=canonical_exposure.entry_execution_id,
            allow_administrative=False,
        )

    def _build_full_exit_sell_event(
        self,
        *,
        decision: OpenPositionExitDecision,
        quantity: int,
        entry_execution_id: str,
        allow_administrative: bool,
    ) -> PortfolioExecutionEvent:
        """The one authoritative SELL pricing/cost/settlement/ID pipeline."""

        canonical_decision = decision
        self._assert_policies_unchanged()
        selected_reason = canonical_decision.selected_reason
        if selected_reason in _PROTECTIVE_EXIT_REASONS:
            if canonical_decision.final_execution_price is None:
                raise BacktestExecutionAdapterValidationError(
                    "INVALID_PROTECTIVE_SELL_PRICE",
                    "a protective exit requires the exact Phase 10 final price",
                )
            final_fill = Decimal(str(canonical_decision.final_execution_price))
        elif selected_reason in _ADMINISTRATIVE_EXIT_REASONS and allow_administrative:
            reference = canonical_decision.reference_exit_price
            if (
                reference is None
                or canonical_decision.final_execution_price is not None
                or canonical_decision.exit_boundary is not ExitBoundary.CLOSE
                or reference != canonical_decision.market_bar.close
            ):
                raise BacktestExecutionAdapterValidationError(
                    "INVALID_ADMINISTRATIVE_SELL_PRICE",
                    "an administrative exit requires the validated Phase 15B close",
                )
            reference_decimal = Decimal(str(reference))
            self._assert_policies_unchanged()
            try:
                price_quote = _canonical_dataclass(
                    self.administrative_pricing_service.price(
                        reference_exit_price=reference_decimal
                    ),
                    AdministrativeExitPriceQuote,
                )
            except (AttributeError, TypeError, ValueError) as error:
                raise BacktestExecutionAdapterValidationError(
                    "ADMINISTRATIVE_SELL_PRICING_FAILED",
                    "the Phase 15C administrative-pricing service rejected the exit",
                ) from error
            self._assert_policies_unchanged()
            if (
                price_quote.policy_ref != self.policy_ref
                or price_quote.reference_exit_price != reference_decimal
            ):
                raise BacktestExecutionAdapterValidationError(
                    "ADMINISTRATIVE_SELL_PRICE_MISMATCH",
                    "administrative price quote provenance does not match the exit",
                )
            final_fill = price_quote.final_fill_price
        else:
            raise BacktestExecutionAdapterValidationError(
                "UNSUPPORTED_SELL_REASON",
                "the terminal Phase 15B reason is not supported by Phase 15C",
            )

        self._assert_policies_unchanged()
        try:
            cost_quote = _canonical_cost_quote(
                self.execution_cost_service.quote(
                    side=ExecutionCostSide.SELL,
                    quantity=quantity,
                    fill_price=final_fill,
                )
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise BacktestExecutionAdapterValidationError(
                "SELL_COST_QUOTE_FAILED",
                "the Phase 15C cost service could not quote the SELL",
            ) from error
        self._assert_policies_unchanged()
        if (
            cost_quote.policy_ref != self.policy_ref
            or cost_quote.side is not ExecutionCostSide.SELL
            or cost_quote.quantity != quantity
            or cost_quote.fill_price != final_fill
        ):
            raise BacktestExecutionAdapterValidationError(
                "SELL_COST_QUOTE_MISMATCH",
                "SELL cost quote provenance does not match the execution",
            )

        try:
            resolution = _canonical_dataclass(
                self.settlement_resolver.resolve(
                    trade_session=canonical_decision.session
                ),
                SettlementResolution,
            )
            if (
                resolution.trade_session != canonical_decision.session
                or resolution.settlement_session <= canonical_decision.session
            ):
                raise ValueError("settlement resolution does not belong to the trade")
        except (AttributeError, TypeError, ValueError) as error:
            raise BacktestExecutionAdapterValidationError(
                "SELL_SETTLEMENT_RESOLUTION_FAILED",
                "historical settlement could not be resolved canonically",
            ) from error

        try:
            source_order_id = self.identifier_service.sell_source_order_id(
                entry_execution_id=entry_execution_id,
                exit_session=canonical_decision.session,
                security_id=canonical_decision.security_id,
            )
            execution_id = self.identifier_service.sell_execution_id(
                entry_execution_id=entry_execution_id,
                exit_session=canonical_decision.session,
                security_id=canonical_decision.security_id,
            )
            settlement_id = self.identifier_service.settlement_id(
                sell_execution_id=execution_id
            )
            event = PortfolioExecutionEvent(
                execution_id=execution_id,
                source_order_id=source_order_id,
                session=canonical_decision.session,
                asset_id=canonical_decision.security_id,
                side=ExecutionSide.SELL,
                quantity=quantity,
                fill_price=final_fill,
                execution_cost=cost_quote.execution_cost,
                settlement_id=settlement_id,
                settlement_session=resolution.settlement_session,
            )
            return _canonical_event(event)
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            raise BacktestExecutionAdapterValidationError(
                "INVALID_SELL_EXECUTION_EVENT",
                "the SELL could not be represented as a canonical Phase 13 event",
            ) from error


__all__ = [
    "BacktestBuyExecutionEventAdapter",
    "BacktestExecutionAdapterValidationError",
    "BacktestSellExecutionEventAdapter",
    "TransientEntryExposure",
]
