"""Dataset-level validation for canonical StockBar collections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import pandas as pd
from pydantic import ValidationError

from stock_swing_d1.models import StockBar


CANONICAL_FIELDS = (
    "security_id",
    "symbol",
    "trading_date",
    "timeframe",
    "session_type",
    "currency",
    "price_basis",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
CANONICAL_KEY = (
    "security_id",
    "trading_date",
    "timeframe",
    "session_type",
    "price_basis",
)
SECURITY_DATE_KEY = ("security_id", "trading_date")


@dataclass(frozen=True, slots=True)
class DatasetValidationIssue:
    """One dataset-level contract violation."""

    reason: str
    security_id: str | None = None
    trading_date: date | None = None

    def as_error(self) -> dict[str, object]:
        error: dict[str, object] = {
            "stage": "dataset_validation",
            "reason": self.reason,
        }
        if self.security_id is not None:
            error["security_id"] = self.security_id
        if self.trading_date is not None:
            error["trading_date"] = self.trading_date.isoformat()
        return error


class DatasetValidationError(ValueError):
    """One or more canonical dataset invariants were violated."""

    def __init__(
        self,
        issues: Sequence[DatasetValidationIssue],
        *,
        duplicate_count: int = 0,
    ) -> None:
        self.issues = tuple(issues)
        self.duplicate_count = duplicate_count
        summary = "; ".join(issue.reason for issue in self.issues)
        super().__init__(summary or "dataset validation failed")


def canonical_rows_to_dataframe(
    rows: Sequence[StockBar | Mapping[str, object]] | pd.DataFrame,
) -> pd.DataFrame:
    """Return canonical columns without silently adding missing fields."""
    if isinstance(rows, pd.DataFrame):
        return rows.copy()

    records: list[dict[str, object]] = []
    for row in rows:
        if isinstance(row, StockBar):
            records.append(row.model_dump(mode="python"))
        elif isinstance(row, Mapping):
            records.append(dict(row))
        else:
            raise TypeError("canonical rows must be StockBar or mapping values")
    if not records:
        return pd.DataFrame(columns=list(CANONICAL_FIELDS))
    return pd.DataFrame.from_records(records)


def _issue_context(row: pd.Series) -> tuple[str | None, date | None]:
    security_id = row.get("security_id")
    trading_date = row.get("trading_date")
    return (
        security_id if isinstance(security_id, str) else None,
        trading_date if type(trading_date) is date else None,
    )


def _duplicate_issues(
    frame: pd.DataFrame,
    subset: Sequence[str],
    reason: str,
) -> tuple[list[DatasetValidationIssue], int]:
    excess_mask = frame.duplicated(subset=list(subset), keep="first")
    issues: list[DatasetValidationIssue] = []
    for _, row in frame.loc[excess_mask].iterrows():
        security_id, trading_date = _issue_context(row)
        issues.append(
            DatasetValidationIssue(
                reason,
                security_id=security_id,
                trading_date=trading_date,
            )
        )
    return issues, int(excess_mask.sum())


def validate_stock_bar_dataset(
    rows: Sequence[StockBar | Mapping[str, object]] | pd.DataFrame,
) -> list[StockBar]:
    """Validate, sort, and return canonical rows as immutable StockBars.

    Symbol is informational metadata. Different symbols for one authoritative
    ``security_id`` are valid and do not create an identity conflict.
    """
    frame = canonical_rows_to_dataframe(rows)
    actual_fields = list(frame.columns)
    missing = [field for field in CANONICAL_FIELDS if field not in actual_fields]
    extra = [field for field in actual_fields if field not in CANONICAL_FIELDS]
    structural_issues: list[DatasetValidationIssue] = []
    if missing:
        structural_issues.append(
            DatasetValidationIssue(
                f"missing canonical fields: {', '.join(missing)}"
            )
        )
    if extra:
        structural_issues.append(
            DatasetValidationIssue(f"unexpected canonical fields: {', '.join(extra)}")
        )
    if structural_issues:
        raise DatasetValidationError(structural_issues)

    frame = frame.loc[:, list(CANONICAL_FIELDS)]
    issues: list[DatasetValidationIssue] = []
    for field in CANONICAL_FIELDS:
        for _, row in frame.loc[frame[field].isna()].iterrows():
            security_id, trading_date = _issue_context(row)
            issues.append(
                DatasetValidationIssue(
                    f"canonical field {field} must not be null",
                    security_id=security_id,
                    trading_date=trading_date,
                )
            )

    expected_values = {
        "timeframe": "D1",
        "session_type": "regular",
        "currency": "USD",
        "price_basis": "unadjusted",
    }
    for field, expected in expected_values.items():
        invalid = frame[field].notna() & frame[field].ne(expected)
        for _, row in frame.loc[invalid].iterrows():
            security_id, trading_date = _issue_context(row)
            issues.append(
                DatasetValidationIssue(
                    f"{field} must equal {expected!r}",
                    security_id=security_id,
                    trading_date=trading_date,
                )
            )

    canonical_duplicate_issues, canonical_duplicate_count = _duplicate_issues(
        frame,
        CANONICAL_KEY,
        "duplicate canonical uniqueness key",
    )
    security_date_issues, security_date_duplicate_count = _duplicate_issues(
        frame,
        SECURITY_DATE_KEY,
        "duplicate security_id and trading_date",
    )
    issues.extend(canonical_duplicate_issues)
    issues.extend(security_date_issues)
    duplicate_count = max(
        canonical_duplicate_count, security_date_duplicate_count
    )

    if issues:
        raise DatasetValidationError(issues, duplicate_count=duplicate_count)

    try:
        sorted_frame = frame.sort_values(
            ["security_id", "trading_date"],
            kind="mergesort",
            ignore_index=True,
        )
    except (TypeError, ValueError) as error:
        raise DatasetValidationError(
            [DatasetValidationIssue("canonical rows cannot be sorted deterministically")]
        ) from error

    increasing_issues: list[DatasetValidationIssue] = []
    for security_id, group in sorted_frame.groupby("security_id", sort=False):
        dates = group["trading_date"].tolist()
        if any(current <= previous for previous, current in zip(dates, dates[1:])):
            increasing_issues.append(
                DatasetValidationIssue(
                    "trading dates must be strictly increasing within security_id",
                    security_id=str(security_id),
                )
            )
    if increasing_issues:
        raise DatasetValidationError(
            increasing_issues, duplicate_count=duplicate_count
        )

    validated: list[StockBar] = []
    for record in sorted_frame.to_dict(orient="records"):
        try:
            validated.append(StockBar(**record))
        except ValidationError as error:
            security_id = record.get("security_id")
            trading_date = record.get("trading_date")
            raise DatasetValidationError(
                [
                    DatasetValidationIssue(
                        f"row is not a valid StockBar: {error}",
                        security_id=(
                            security_id if isinstance(security_id, str) else None
                        ),
                        trading_date=(
                            trading_date if type(trading_date) is date else None
                        ),
                    )
                ]
            ) from error
    return validated


__all__ = [
    "CANONICAL_FIELDS",
    "CANONICAL_KEY",
    "SECURITY_DATE_KEY",
    "DatasetValidationError",
    "DatasetValidationIssue",
    "canonical_rows_to_dataframe",
    "validate_stock_bar_dataset",
]
