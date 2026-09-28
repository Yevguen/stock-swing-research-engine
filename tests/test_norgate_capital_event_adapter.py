# Synthetic test data: security identifiers and market values in this module were
# invented for testing and are not provider data. Any real ticker symbol that
# appears is used only as a format example and is not paired with provider data.
from datetime import date
from math import inf, nan

import pandas as pd
import pytest

from stock_swing_d1.data.norgate_capital_event_adapter import (
    EVENT_RAW_COLUMNS,
    CapitalEventProviderValidationError,
    fetch_norgate_capital_events,
    map_norgate_capital_event_raw_frame,
    prepare_norgate_capital_event_raw_frame,
)


def event_frame(
    values: tuple[object, ...] = (0, 1),
    dates: tuple[str, ...] = ("2026-08-06", "2026-08-07"),
    *,
    column: str = "Whatever Provider Calls It",
) -> pd.DataFrame:
    return pd.DataFrame({column: list(values)}, index=pd.to_datetime(list(dates)))


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def capital_event_timeseries(
        self, *args: object, **kwargs: object
    ) -> pd.DataFrame:
        self.calls.append((args, kwargs))
        return event_frame()


def test_event_fetch_uses_exact_documented_call() -> None:
    provider = FakeProvider()

    result = fetch_norgate_capital_events(1_900_000_001, provider=provider)

    assert isinstance(result, pd.DataFrame)
    assert provider.calls == [
        ((1_900_000_001,), {"timeseriesformat": "pandas-dataframe"})
    ]


@pytest.mark.parametrize("asset_id", [True, 0, -1, 1.0, "1"])
def test_event_fetch_rejects_invalid_asset_ids(asset_id: object) -> None:
    with pytest.raises(ValueError):
        fetch_norgate_capital_events(asset_id, provider=FakeProvider())  # type: ignore[arg-type]


def test_event_normalization_resolves_one_column_and_filters_locally() -> None:
    raw = prepare_norgate_capital_event_raw_frame(
        1,
        "aaa",
        event_frame(
            (1, 0, 1),
            ("2026-08-06", "2026-08-07", "2026-08-08"),
        ),
        start_date=date(2026, 8, 7),
        end_date=date(2026, 8, 7),
    )

    assert list(raw.columns) == list(EVENT_RAW_COLUMNS)
    assert raw.to_dict(orient="records") == [
        {
            "provider_asset_id": 1,
            "provider_symbol": "aaa",
            "event_date": date(2026, 8, 7),
            "capital_event_flag": 0,
        }
    ]


def test_event_response_requires_exactly_one_unambiguous_value_column() -> None:
    with pytest.raises(CapitalEventProviderValidationError, match="no usable"):
        prepare_norgate_capital_event_raw_frame(
            1,
            "AAA",
            pd.DataFrame(index=pd.to_datetime(["2026-08-07"])),
            start_date=date(2026, 8, 7),
            end_date=date(2026, 8, 7),
        )
    with pytest.raises(CapitalEventProviderValidationError, match="ambiguous"):
        prepare_norgate_capital_event_raw_frame(
            1,
            "AAA",
            pd.DataFrame(
                {"one": [0], "two": [1]},
                index=pd.to_datetime(["2026-08-07"]),
            ),
            start_date=date(2026, 8, 7),
            end_date=date(2026, 8, 7),
        )


def test_event_response_requires_dataframe_and_valid_symbol() -> None:
    with pytest.raises(CapitalEventProviderValidationError, match="DataFrame"):
        prepare_norgate_capital_event_raw_frame(
            1,
            "AAA",
            None,  # type: ignore[arg-type]
            start_date=date(2026, 8, 7),
            end_date=date(2026, 8, 7),
        )
    with pytest.raises(CapitalEventProviderValidationError, match="non-empty"):
        prepare_norgate_capital_event_raw_frame(
            1,
            " ",
            event_frame(),
            start_date=date(2026, 8, 6),
            end_date=date(2026, 8, 7),
        )


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0])
def test_event_flags_accept_only_lossless_zero_or_one(value: object) -> None:
    raw = prepare_norgate_capital_event_raw_frame(
        1,
        "AAA",
        event_frame((value,), ("2026-08-07",)),
        start_date=date(2026, 8, 7),
        end_date=date(2026, 8, 7),
    )

    assert raw["capital_event_flag"].tolist() == [int(value)]


@pytest.mark.parametrize("value", [-1, 2, 0.5, nan, inf, -inf, "1", True])
def test_invalid_event_flags_are_rejected(value: object) -> None:
    with pytest.raises(CapitalEventProviderValidationError, match="exactly 0 or 1"):
        prepare_norgate_capital_event_raw_frame(
            1,
            "AAA",
            event_frame((value,), ("2026-08-07",)),
            start_date=date(2026, 8, 7),
            end_date=date(2026, 8, 7),
        )


def test_duplicate_in_range_event_dates_are_rejected() -> None:
    with pytest.raises(CapitalEventProviderValidationError, match="duplicate"):
        prepare_norgate_capital_event_raw_frame(
            1,
            "AAA",
            event_frame((0, 1), ("2026-08-07", "2026-08-07")),
            start_date=date(2026, 8, 7),
            end_date=date(2026, 8, 7),
        )


def test_malformed_event_date_index_is_rejected() -> None:
    frame = pd.DataFrame({"value": [1]}, index=["not-a-date"])

    with pytest.raises(CapitalEventProviderValidationError, match="malformed"):
        prepare_norgate_capital_event_raw_frame(
            1,
            "AAA",
            frame,
            start_date=date(2026, 8, 7),
            end_date=date(2026, 8, 7),
        )


def test_zero_flag_emits_no_event_and_one_maps_exactly() -> None:
    raw = prepare_norgate_capital_event_raw_frame(
        1_900_000_001,
        "syntha",
        event_frame((0, 1), ("2026-08-06", "2026-08-07")),
        start_date=date(2026, 8, 6),
        end_date=date(2026, 8, 7),
    )

    [event] = map_norgate_capital_event_raw_frame(raw)

    assert event.model_dump() == {
        "security_id": "NORGATE:1900000001",
        "symbol": "SYNTHA",
        "event_date": date(2026, 8, 7),
        "date_semantics": "entitlement_close",
        "event_type": "unknown_capital_event",
        "terms_verified": False,
        "new_shares": None,
        "old_shares": None,
        "source_provider": "Norgate Data",
        "source_asset_id": 1_900_000_001,
    }


def normalized_raw_frame(flag: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider_asset_id": [1_900_000_001],
            "provider_symbol": ["SYNTHA"],
            "event_date": [date(2026, 8, 7)],
            "capital_event_flag": [flag],
        },
        columns=list(EVENT_RAW_COLUMNS),
    )


def test_raw_mapper_rejects_invalid_normalized_flag_instead_of_ignoring() -> None:
    with pytest.raises(CapitalEventProviderValidationError, match="exactly 0 or 1"):
        map_norgate_capital_event_raw_frame(normalized_raw_frame(2))


def test_raw_mapper_valid_zero_flag_emits_no_event() -> None:
    assert map_norgate_capital_event_raw_frame(normalized_raw_frame(0)) == []


def test_raw_mapper_valid_one_flag_maps_normally() -> None:
    [event] = map_norgate_capital_event_raw_frame(normalized_raw_frame(1))

    assert event.security_id == "NORGATE:1900000001"
    assert event.event_date == date(2026, 8, 7)
    assert event.event_type == "unknown_capital_event"


def test_zero_in_range_rows_is_structurally_valid() -> None:
    raw = prepare_norgate_capital_event_raw_frame(
        1,
        "AAA",
        event_frame((1,), ("2020-01-01",)),
        start_date=date(2026, 8, 7),
        end_date=date(2026, 8, 7),
    )

    assert raw.empty
    assert map_norgate_capital_event_raw_frame(raw) == []
