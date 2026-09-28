"""Lossless canonical persistence for authoritative Phase 16B results.

This module is an adaptation boundary only.  It copies already-authoritative
metrics and context into a versioned artifact, verifies their identities, and
encodes or decodes that artifact.  It never imports the calculation layer and
never consumes an equity series.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from stock_swing_d1.backtester.decision_interval import HistoricalDecisionInterval
from stock_swing_d1.research_metrics.errors import ResearchMetricsPersistenceError
from stock_swing_d1.research_metrics.hashing import semantic_sha256
from stock_swing_d1.research_metrics.models import (
    BenchmarkPerformanceMetrics,
    RelativePerformanceMetrics,
    ResearchMetricValue,
    ResearchMetricsProvenance,
    ResearchMetricsResult,
    StrategyPerformanceMetrics,
    TradePerformanceMetrics,
)
from stock_swing_d1.research_metrics.policy import (
    PERFORMANCE_MEASUREMENT_POLICY_ID,
    PERFORMANCE_MEASUREMENT_POLICY_SCHEMA_VERSION,
    PERFORMANCE_MEASUREMENT_POLICY_VERSION,
    PerformanceMeasurementPolicy,
    UndefinedMetricReason,
)


# The Phase 16B Amendment v0.1 advances both identities to v0.2. The wrapper's
# own field inventory is unchanged, but the canonical payload it hashes embeds
# the amended result, and the strict decoder below now requires the amended
# trade-metric field set -- so a v0.1 identity must never be able to stand for
# an amended artifact, in either direction.
RESEARCH_METRICS_PERSISTENCE_SCHEMA_VERSION = (
    "research_metrics_persistence.v0.2"
)
RESEARCH_METRICS_PERSISTENCE_HASH_DOMAIN = (
    "research_metrics_persistence.v0.2"
)
_CANONICAL_STARTING_CAPITAL = Decimal("100000")


def _require_canonical_text(value: object) -> object:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("must be canonical non-empty text")
    return value


def _require_starting_capital(value: object) -> object:
    if type(value) is not Decimal:
        raise ValueError("must be an exact Decimal")
    if not value.is_finite() or value != _CANONICAL_STARTING_CAPITAL:
        raise ValueError("must equal the frozen USD 100,000 capital anchor")
    return value


def _require_non_negative_count(value: object) -> object:
    if type(value) is not int or value < 0:
        raise ValueError("must be an exact non-negative int")
    return value


def _require_positive_count(value: object) -> object:
    if type(value) is not int or value <= 0:
        raise ValueError("must be an exact positive int")
    return value


def _require_exact_model(model_type: type[BaseModel]):
    def validate(value: object) -> object:
        if type(value) is not model_type:
            raise ValueError(f"must be an exact {model_type.__name__}")
        return value

    return validate


_CanonicalText = Annotated[str, BeforeValidator(_require_canonical_text)]
_StartingCapital = Annotated[
    Decimal, BeforeValidator(_require_starting_capital)
]
_NonNegativeCount = Annotated[int, BeforeValidator(_require_non_negative_count)]
_PositiveCount = Annotated[int, BeforeValidator(_require_positive_count)]
_Sha256 = Annotated[
    str,
    BeforeValidator(_require_canonical_text),
    Field(pattern=r"^[0-9a-f]{64}$", strict=True),
]
_DecisionInterval = Annotated[
    HistoricalDecisionInterval,
    BeforeValidator(_require_exact_model(HistoricalDecisionInterval)),
]
_ResearchMetricsResult = Annotated[
    ResearchMetricsResult,
    BeforeValidator(_require_exact_model(ResearchMetricsResult)),
]


class _ImmutablePersistenceModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=False,
        validate_default=True,
    )


class ResearchMetricsPolicyRef(_ImmutablePersistenceModel):
    """Minimum stable identity of the policy that produced the result."""

    schema_version: Literal[
        "performance_measurement_policy.v0.3"
    ] = PERFORMANCE_MEASUREMENT_POLICY_SCHEMA_VERSION
    policy_id: Literal[
        "explicit_research_performance_measurement_v0.3"
    ] = PERFORMANCE_MEASUREMENT_POLICY_ID
    policy_version: Literal["0.3"] = PERFORMANCE_MEASUREMENT_POLICY_VERSION
    policy_fingerprint: _Sha256


class ResearchMetricsPersistenceArtifact(_ImmutablePersistenceModel):
    """Versioned, fingerprint-bound representation of one Phase 16B result."""

    schema_version: Literal[
        "research_metrics_persistence.v0.2"
    ] = RESEARCH_METRICS_PERSISTENCE_SCHEMA_VERSION
    decision_interval: _DecisionInterval
    starting_capital: _StartingCapital
    equity_observation_count: _PositiveCount
    periodic_return_observation_count: _NonNegativeCount
    policy_ref: ResearchMetricsPolicyRef
    research_metrics_result: _ResearchMetricsResult
    artifact_fingerprint: _Sha256

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        try:
            rebuilt_interval = _rebuild_model_value(self.decision_interval)
            rebuilt_policy_ref = _rebuild_model_value(self.policy_ref)
            rebuilt_result = _rebuild_model_value(self.research_metrics_result)
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            raise ValueError("nested authoritative content is invalid") from error
        if rebuilt_interval != self.decision_interval:
            raise ValueError("decision_interval is not canonical")
        if rebuilt_policy_ref != self.policy_ref:
            raise ValueError("policy_ref is not canonical")
        if rebuilt_result != self.research_metrics_result:
            raise ValueError("research_metrics_result is not canonical")
        if (
            self.periodic_return_observation_count + 1
            != self.equity_observation_count
        ):
            raise ValueError(
                "N equity observations must have exactly N-1 periodic returns"
            )
        if (
            self.policy_ref.policy_fingerprint
            != self.research_metrics_result.provenance.performance_measurement_policy_fingerprint
        ):
            raise ValueError("policy reference does not match result provenance")
        expected = compute_research_metrics_persistence_fingerprint(self)
        if self.artifact_fingerprint != expected:
            raise ValueError("artifact_fingerprint does not match artifact content")
        return self


def _rebuild_model_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return type(value)(
            **{
                field_name: _rebuild_model_value(getattr(value, field_name))
                for field_name in type(value).model_fields
            }
        )
    if type(value) is tuple:
        return tuple(_rebuild_model_value(item) for item in value)
    return value


def _exact_decimal_string(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("persistence Decimal values must be finite")
    return str(value)


def canonicalize_persistence_value(value: object) -> Any:
    """Convert supported values to deterministic, lossless JSON data."""

    if isinstance(value, BaseModel):
        return _canonical_mapping(
            {name: getattr(value, name) for name in type(value).model_fields}
        )
    if isinstance(value, Enum):
        return canonicalize_persistence_value(value.value)
    if isinstance(value, Decimal):
        return _exact_decimal_string(value)
    if isinstance(value, datetime):
        try:
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("persistence datetimes must be timezone-aware")
        except (OverflowError, TypeError) as error:
            raise ValueError(
                "persistence datetimes must be timezone-aware"
            ) from error
        return value.astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    if type(value) is date:
        return value.isoformat()
    if isinstance(value, Mapping):
        return _canonical_mapping(value)
    if type(value) in (tuple, list):
        return [canonicalize_persistence_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("sets are not canonical persistence values")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        raise TypeError("binary floats are forbidden in persistence artifacts")
    raise TypeError(f"unsupported persistence value: {type(value).__name__}")


def _canonical_mapping(value: Mapping[object, object]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, item in value.items():
        if type(key) is not str or not key or key != key.strip():
            raise TypeError("persistence mappings require canonical string keys")
        converted[key] = canonicalize_persistence_value(item)
    return {key: converted[key] for key in sorted(converted)}


def persistence_json_bytes(value: object) -> bytes:
    """Return compact deterministic UTF-8 JSON with lossless Decimals."""

    return json.dumps(
        canonicalize_persistence_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _artifact_payload(
    *,
    schema_version: object,
    decision_interval: object,
    starting_capital: object,
    equity_observation_count: object,
    periodic_return_observation_count: object,
    policy_ref: object,
    research_metrics_result: object,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "decision_interval": decision_interval,
        "starting_capital": starting_capital,
        "equity_observation_count": equity_observation_count,
        "periodic_return_observation_count": periodic_return_observation_count,
        "policy_ref": policy_ref,
        "research_metrics_result": research_metrics_result,
    }


def _compute_research_metrics_persistence_fingerprint(**values: object) -> str:
    payload = canonicalize_persistence_value(_artifact_payload(**values))
    return semantic_sha256(
        {
            "domain": RESEARCH_METRICS_PERSISTENCE_HASH_DOMAIN,
            "payload": payload,
        }
    )


def compute_research_metrics_persistence_fingerprint(
    artifact: ResearchMetricsPersistenceArtifact,
) -> str:
    """Hash canonical artifact content, excluding the fingerprint itself."""

    if type(artifact) is not ResearchMetricsPersistenceArtifact:
        raise TypeError("artifact must be a ResearchMetricsPersistenceArtifact")
    return _compute_research_metrics_persistence_fingerprint(
        schema_version=artifact.schema_version,
        decision_interval=artifact.decision_interval,
        starting_capital=artifact.starting_capital,
        equity_observation_count=artifact.equity_observation_count,
        periodic_return_observation_count=(
            artifact.periodic_return_observation_count
        ),
        policy_ref=artifact.policy_ref,
        research_metrics_result=artifact.research_metrics_result,
    )


def build_research_metrics_persistence_artifact(
    *,
    result: ResearchMetricsResult,
    decision_interval: HistoricalDecisionInterval,
    starting_capital: Decimal,
    equity_observation_count: int,
    periodic_return_observation_count: int,
    policy: PerformanceMeasurementPolicy,
) -> ResearchMetricsPersistenceArtifact:
    """Copy authoritative result/context into one fingerprint-bound artifact."""

    if type(result) is not ResearchMetricsResult:
        raise TypeError("result must be a ResearchMetricsResult")
    if type(decision_interval) is not HistoricalDecisionInterval:
        raise TypeError("decision_interval must be a HistoricalDecisionInterval")
    if type(policy) is not PerformanceMeasurementPolicy:
        raise TypeError("policy must be a PerformanceMeasurementPolicy")
    try:
        canonical_result = _rebuild_model_value(result)
        canonical_interval = _rebuild_model_value(decision_interval)
        canonical_policy = _rebuild_model_value(policy)
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise ResearchMetricsPersistenceError(
            "INVALID_AUTHORITATIVE_INPUT",
            "result, interval, or policy fails canonical validation",
        ) from error
    policy_ref = ResearchMetricsPolicyRef(
        schema_version=canonical_policy.schema_version,
        policy_id=canonical_policy.policy_id,
        policy_version=canonical_policy.policy_version,
        policy_fingerprint=canonical_policy.policy_fingerprint,
    )
    values = _artifact_payload(
        schema_version=RESEARCH_METRICS_PERSISTENCE_SCHEMA_VERSION,
        decision_interval=canonical_interval,
        starting_capital=starting_capital,
        equity_observation_count=equity_observation_count,
        periodic_return_observation_count=periodic_return_observation_count,
        policy_ref=policy_ref,
        research_metrics_result=canonical_result,
    )
    try:
        fingerprint = _compute_research_metrics_persistence_fingerprint(**values)
        return ResearchMetricsPersistenceArtifact(
            **values, artifact_fingerprint=fingerprint
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise ResearchMetricsPersistenceError(
            "INVALID_PERSISTENCE_CONTEXT",
            "persistence context is invalid or inconsistent with the result",
        ) from error


def serialize_research_metrics_artifact(
    artifact: ResearchMetricsPersistenceArtifact,
) -> bytes:
    """Serialize one validated artifact to its canonical JSON bytes."""

    if type(artifact) is not ResearchMetricsPersistenceArtifact:
        raise TypeError("artifact must be a ResearchMetricsPersistenceArtifact")
    try:
        rebuilt = ResearchMetricsPersistenceArtifact(
            **{
                name: getattr(artifact, name)
                for name in ResearchMetricsPersistenceArtifact.model_fields
            }
        )
        return persistence_json_bytes(rebuilt)
    except (TypeError, ValueError, ValidationError) as error:
        raise ResearchMetricsPersistenceError(
            "INVALID_ARTIFACT", "artifact fails canonical validation"
        ) from error


def _require_object(
    value: object, required_fields: tuple[str, ...], *, location: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{location} must be a JSON object")
    actual = set(value)
    expected = set(required_fields)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{location} has invalid fields; missing={missing}, extra={extra}"
        )
    return value


def _decode_decimal(value: object, *, location: str) -> Decimal:
    if type(value) is not str or not value:
        raise ValueError(f"{location} must be a canonical Decimal string")
    try:
        decoded = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{location} is not a valid Decimal") from error
    if not decoded.is_finite() or _exact_decimal_string(decoded) != value:
        raise ValueError(f"{location} is not a canonical finite Decimal")
    return decoded


def _decode_date(value: object, *, location: str) -> date:
    if type(value) is not str:
        raise ValueError(f"{location} must be a canonical ISO date")
    try:
        decoded = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{location} is not a valid ISO date") from error
    if decoded.isoformat() != value:
        raise ValueError(f"{location} is not a canonical ISO date")
    return decoded


def _decode_metric(value: object, *, location: str) -> ResearchMetricValue:
    payload = _require_object(
        value,
        ("schema_version", "undefined_reason", "value"),
        location=location,
    )
    raw_reason = payload["undefined_reason"]
    if raw_reason is None:
        reason = None
    else:
        if type(raw_reason) is not str:
            raise ValueError(f"{location}.undefined_reason must be a string")
        try:
            reason = UndefinedMetricReason(raw_reason)
        except ValueError as error:
            raise ValueError(
                f"{location}.undefined_reason is not a known reason code"
            ) from error
    raw_value = payload["value"]
    decoded_value = (
        None
        if raw_value is None
        else _decode_decimal(raw_value, location=f"{location}.value")
    )
    return ResearchMetricValue(
        schema_version=payload["schema_version"],
        value=decoded_value,
        undefined_reason=reason,
    )


def _decode_strategy_metrics(value: object) -> StrategyPerformanceMetrics:
    fields = (
        "schema_version",
        "ending_equity",
        "net_pnl",
        "total_return",
        "cagr",
        "annualized_volatility",
        "maximum_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
    )
    payload = _require_object(value, fields, location="strategy_metrics")
    return StrategyPerformanceMetrics(
        schema_version=payload["schema_version"],
        ending_equity=_decode_decimal(
            payload["ending_equity"], location="strategy_metrics.ending_equity"
        ),
        net_pnl=_decode_decimal(
            payload["net_pnl"], location="strategy_metrics.net_pnl"
        ),
        **{
            name: _decode_metric(
                payload[name], location=f"strategy_metrics.{name}"
            )
            for name in fields[3:]
        },
    )


def _decode_trade_metrics(value: object) -> TradePerformanceMetrics:
    fields = (
        "schema_version",
        "completed_trade_count",
        "win_count",
        "loss_count",
        "breakeven_count",
        "gross_profit",
        "gross_loss",
        "win_rate",
        "loss_rate",
        "breakeven_rate",
        "profit_factor",
        "expectancy",
        "average_winner_usd",
        "average_loser_usd",
        "worst_trade_usd",
        "average_holding_sessions",
    )
    payload = _require_object(value, fields, location="trade_metrics")
    return TradePerformanceMetrics(
        schema_version=payload["schema_version"],
        completed_trade_count=payload["completed_trade_count"],
        win_count=payload["win_count"],
        loss_count=payload["loss_count"],
        breakeven_count=payload["breakeven_count"],
        gross_profit=_decode_decimal(
            payload["gross_profit"], location="trade_metrics.gross_profit"
        ),
        gross_loss=_decode_decimal(
            payload["gross_loss"], location="trade_metrics.gross_loss"
        ),
        **{
            name: _decode_metric(payload[name], location=f"trade_metrics.{name}")
            for name in fields[7:]
        },
    )


def _decode_benchmark_metrics(value: object) -> BenchmarkPerformanceMetrics:
    fields = (
        "schema_version",
        "ending_equity",
        "total_return",
        "cagr",
        "annualized_volatility",
        "maximum_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
    )
    payload = _require_object(value, fields, location="benchmark_metrics")
    return BenchmarkPerformanceMetrics(
        schema_version=payload["schema_version"],
        ending_equity=_decode_decimal(
            payload["ending_equity"], location="benchmark_metrics.ending_equity"
        ),
        **{
            name: _decode_metric(
                payload[name], location=f"benchmark_metrics.{name}"
            )
            for name in fields[2:]
        },
    )


def _decode_relative_metrics(value: object) -> RelativePerformanceMetrics:
    fields = (
        "schema_version",
        "strategy_minus_benchmark_total_return",
        "strategy_minus_benchmark_cagr",
        "ending_wealth_ratio",
    )
    payload = _require_object(value, fields, location="relative_metrics")
    return RelativePerformanceMetrics(
        schema_version=payload["schema_version"],
        **{
            name: _decode_metric(payload[name], location=f"relative_metrics.{name}")
            for name in fields[1:]
        },
    )


def _decode_result(value: object) -> ResearchMetricsResult:
    fields = (
        "schema_version",
        "provenance",
        "strategy_metrics",
        "trade_metrics",
        "benchmark_metrics",
        "relative_metrics",
        "result_fingerprint",
    )
    payload = _require_object(value, fields, location="research_metrics_result")
    provenance_payload = _require_object(
        payload["provenance"],
        (
            "schema_version",
            "source_audit_result_fingerprint",
            "performance_measurement_policy_fingerprint",
            "benchmark_series_fingerprint",
        ),
        location="provenance",
    )
    provenance = ResearchMetricsProvenance(**provenance_payload)
    benchmark_payload = payload["benchmark_metrics"]
    relative_payload = payload["relative_metrics"]
    return ResearchMetricsResult(
        schema_version=payload["schema_version"],
        provenance=provenance,
        strategy_metrics=_decode_strategy_metrics(payload["strategy_metrics"]),
        trade_metrics=_decode_trade_metrics(payload["trade_metrics"]),
        benchmark_metrics=(
            None
            if benchmark_payload is None
            else _decode_benchmark_metrics(benchmark_payload)
        ),
        relative_metrics=(
            None
            if relative_payload is None
            else _decode_relative_metrics(relative_payload)
        ),
        result_fingerprint=payload["result_fingerprint"],
    )


def _decode_policy_ref(value: object) -> ResearchMetricsPolicyRef:
    payload = _require_object(
        value,
        ("schema_version", "policy_id", "policy_version", "policy_fingerprint"),
        location="policy_ref",
    )
    return ResearchMetricsPolicyRef(**payload)


def _decode_interval(value: object) -> HistoricalDecisionInterval:
    payload = _require_object(
        value,
        ("decision_start_date", "decision_end_date"),
        location="decision_interval",
    )
    return HistoricalDecisionInterval(
        decision_start_date=_decode_date(
            payload["decision_start_date"],
            location="decision_interval.decision_start_date",
        ),
        decision_end_date=_decode_date(
            payload["decision_end_date"],
            location="decision_interval.decision_end_date",
        ),
    )


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"prohibited JSON constant: {value}")


def _reject_json_float(value: str) -> object:
    raise ValueError(f"JSON floating-point numbers are prohibited: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def deserialize_research_metrics_artifact(
    data: bytes | str,
) -> ResearchMetricsPersistenceArtifact:
    """Load and fully verify one canonical Phase 16B.3 JSON artifact."""

    if type(data) is bytes:
        raw_bytes = data
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ResearchMetricsPersistenceError(
                "INVALID_UTF8", "artifact is not valid UTF-8"
            ) from error
    elif type(data) is str:
        text = data
        try:
            raw_bytes = data.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ResearchMetricsPersistenceError(
                "INVALID_UTF8", "artifact is not valid UTF-8"
            ) from error
    else:
        raise TypeError("data must be exact bytes or str")
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
        fields = (
            "schema_version",
            "decision_interval",
            "starting_capital",
            "equity_observation_count",
            "periodic_return_observation_count",
            "policy_ref",
            "research_metrics_result",
            "artifact_fingerprint",
        )
        payload = _require_object(raw, fields, location="artifact")
        if payload["schema_version"] != RESEARCH_METRICS_PERSISTENCE_SCHEMA_VERSION:
            raise ResearchMetricsPersistenceError(
                "UNSUPPORTED_SCHEMA_VERSION",
                "schema_version must be research_metrics_persistence.v0.2",
            )
        decoded_values = _artifact_payload(
            schema_version=payload["schema_version"],
            decision_interval=_decode_interval(payload["decision_interval"]),
            starting_capital=_decode_decimal(
                payload["starting_capital"], location="starting_capital"
            ),
            equity_observation_count=payload["equity_observation_count"],
            periodic_return_observation_count=(
                payload["periodic_return_observation_count"]
            ),
            policy_ref=_decode_policy_ref(payload["policy_ref"]),
            research_metrics_result=_decode_result(
                payload["research_metrics_result"]
            ),
        )
        supplied_fingerprint = payload["artifact_fingerprint"]
        if (
            type(supplied_fingerprint) is not str
            or len(supplied_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in supplied_fingerprint)
        ):
            raise ResearchMetricsPersistenceError(
                "INVALID_ARTIFACT_FINGERPRINT",
                "artifact_fingerprint must be a lowercase SHA-256 digest",
            )
        expected_fingerprint = _compute_research_metrics_persistence_fingerprint(
            **decoded_values
        )
        if supplied_fingerprint != expected_fingerprint:
            raise ResearchMetricsPersistenceError(
                "ARTIFACT_FINGERPRINT_MISMATCH",
                "artifact content does not match artifact_fingerprint",
            )
        artifact = ResearchMetricsPersistenceArtifact(
            **decoded_values, artifact_fingerprint=supplied_fingerprint
        )
    except ResearchMetricsPersistenceError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
        raise ResearchMetricsPersistenceError(
            "MALFORMED_ARTIFACT", "artifact fails strict schema validation"
        ) from error
    canonical_bytes = serialize_research_metrics_artifact(artifact)
    if raw_bytes != canonical_bytes:
        raise ResearchMetricsPersistenceError(
            "NONCANONICAL_JSON",
            "artifact bytes are valid JSON but not the canonical representation",
        )
    return artifact


def write_research_metrics_artifact(
    path: str | Path, artifact: ResearchMetricsPersistenceArtifact
) -> None:
    """Write canonical bytes to a caller-selected destination."""

    try:
        Path(path).write_bytes(serialize_research_metrics_artifact(artifact))
    except OSError as error:
        raise ResearchMetricsPersistenceError(
            "ARTIFACT_WRITE_FAILED", "could not write persistence artifact"
        ) from error


def read_research_metrics_artifact(
    path: str | Path,
) -> ResearchMetricsPersistenceArtifact:
    """Read and verify canonical bytes from a caller-selected path."""

    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise ResearchMetricsPersistenceError(
            "ARTIFACT_READ_FAILED", "could not read persistence artifact"
        ) from error
    return deserialize_research_metrics_artifact(data)


__all__ = [
    "RESEARCH_METRICS_PERSISTENCE_HASH_DOMAIN",
    "RESEARCH_METRICS_PERSISTENCE_SCHEMA_VERSION",
    "ResearchMetricsPersistenceArtifact",
    "ResearchMetricsPolicyRef",
    "build_research_metrics_persistence_artifact",
    "canonicalize_persistence_value",
    "compute_research_metrics_persistence_fingerprint",
    "deserialize_research_metrics_artifact",
    "persistence_json_bytes",
    "read_research_metrics_artifact",
    "serialize_research_metrics_artifact",
    "write_research_metrics_artifact",
]
