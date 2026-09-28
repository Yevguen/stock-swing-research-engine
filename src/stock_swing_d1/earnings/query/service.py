"""Verified point-in-time queries over one published earnings snapshot."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Final

from stock_swing_d1.earnings.models import (
    EarningsScheduleRevision,
    EarningsScheduleStatus,
    EarningsStateAsOf,
    EarningsValidationError,
    LifecycleState,
    TimingClass,
)
from stock_swing_d1.earnings.persistence import (
    EarningsPersistencePaths,
    EarningsPublicationManifest,
    event_history_key,
    read_canonical_revisions,
    read_publication_manifest,
)
from stock_swing_d1.earnings.pit import reconstruct_earnings_state
from stock_swing_d1.earnings.validation import validate_revision_history


_CANONICAL_ASSET_ID: Final[re.Pattern[str]] = re.compile(
    r"^NORGATE:[1-9][0-9]*$"
)
_ACTIVE_LIFECYCLE_STATES: Final[frozenset[LifecycleState]] = frozenset(
    {LifecycleState.ESTIMATED, LifecycleState.CONFIRMED}
)


def _fail(code: str, message: str) -> None:
    raise EarningsValidationError(code, message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publication_metrics(
    revisions: tuple[EarningsScheduleRevision, ...],
) -> tuple[int, int, datetime, datetime]:
    event_count = len({event_history_key(revision) for revision in revisions})
    asset_count = len({revision.canonical_asset_id for revision in revisions})
    effective_times = tuple(
        revision.strategy_effective_at.astimezone(timezone.utc)
        for revision in revisions
        if revision.strategy_effective_at is not None
    )
    if not revisions or not effective_times or len(effective_times) != len(revisions):
        _fail(
            "INVALID_PUBLISHED_ARTIFACT",
            "published revisions require complete strategy-effective coverage",
        )
    return event_count, asset_count, min(effective_times), max(effective_times)


def _load_verified_snapshot(
    published_parquet_path: Path,
    manifest_path: Path,
) -> tuple[EarningsPublicationManifest, tuple[EarningsScheduleRevision, ...]]:
    """Read and verify one exact publication pair without repairing it."""

    try:
        manifest = read_publication_manifest(manifest_path)
        checksum_before = _file_sha256(published_parquet_path)
        revisions = read_canonical_revisions(published_parquet_path)
        checksum_after = _file_sha256(published_parquet_path)
        event_count, asset_count, minimum, maximum = _publication_metrics(revisions)
        provider_histories: dict[
            tuple[str | None, str | None, str | None],
            list[EarningsScheduleRevision],
        ] = defaultdict(list)
        for revision in revisions:
            provider_histories[
                (
                    revision.provider_name,
                    revision.canonical_asset_id,
                    revision.event_instance_id,
                )
            ].append(revision)
        for history in provider_histories.values():
            validate_revision_history(history)
        if checksum_before != checksum_after:
            _fail(
                "INVALID_PUBLISHED_ARTIFACT",
                "published Parquet changed while the snapshot was loading",
            )
        if (
            manifest.output_file != published_parquet_path.name
            or manifest.output_sha256 != checksum_after
            or manifest.published_record_count != len(revisions)
            or manifest.published_event_count != event_count
            or manifest.published_asset_count != asset_count
            or manifest.minimum_strategy_effective_at != minimum
            or manifest.maximum_strategy_effective_at != maximum
        ):
            _fail(
                "INVALID_PUBLISHED_ARTIFACT",
                "published manifest does not match its canonical Parquet artifact",
            )
        return manifest, revisions
    except EarningsValidationError as exc:
        if exc.code == "INVALID_PUBLISHED_ARTIFACT":
            raise
        raise EarningsValidationError(
            "INVALID_PUBLISHED_ARTIFACT", f"canonical publication is invalid: {exc}"
        ) from exc
    except Exception as exc:
        raise EarningsValidationError(
            "INVALID_PUBLISHED_ARTIFACT", f"cannot verify canonical publication: {exc}"
        ) from exc


def _validate_query_inputs(
    *,
    canonical_asset_id: str,
    provider_name: str | None,
    as_of: datetime,
    decision_session: date,
) -> None:
    try:
        timestamp_is_aware = (
            isinstance(as_of, datetime)
            and as_of.tzinfo is not None
            and as_of.utcoffset() is not None
        )
    except Exception:
        timestamp_is_aware = False
    if not timestamp_is_aware:
        _fail(
            "INVALID_QUERY_TIMESTAMP",
            "as_of must be an explicitly timezone-aware datetime",
        )
    if type(decision_session) is not date:
        _fail(
            "INVALID_QUERY_SESSION",
            "decision_session must be a genuine date",
        )
    if (
        not isinstance(canonical_asset_id, str)
        or _CANONICAL_ASSET_ID.fullmatch(canonical_asset_id) is None
    ):
        _fail(
            "INVALID_QUERY_ASSET_ID",
            "canonical_asset_id must have the form NORGATE:<positive AssetId>",
        )
    if (
        not isinstance(provider_name, str)
        or not provider_name
        or provider_name != provider_name.strip()
    ):
        _fail(
            "INVALID_QUERY_PROVIDER",
            "provider_name must be an explicit non-empty provider value",
        )


def _unknown_state(
    *, canonical_asset_id: str, as_of: datetime
) -> EarningsStateAsOf:
    return EarningsStateAsOf(
        as_of=as_of,
        canonical_asset_id=canonical_asset_id,
        event_instance_id=None,
        earnings_schedule_known=False,
        lifecycle_state=LifecycleState.UNKNOWN,
        scheduled_date=None,
        timing_class=TimingClass.UNKNOWN,
        scheduled_at=None,
        knowledge_effective_at=None,
        schedule_status=EarningsScheduleStatus.UNKNOWN,
    )


@dataclass(frozen=True, slots=True, init=False, repr=False)
class PublishedEarningsPITQuery:
    """Immutable query view bound to one verified publication snapshot.

    Construction eagerly verifies and materializes the selected publication.
    Later filesystem replacements therefore cannot alter this instance's query
    results or cause it to switch snapshots.
    """

    _published_parquet_path: Path
    _manifest_path: Path
    _manifest: EarningsPublicationManifest
    _revisions: tuple[EarningsScheduleRevision, ...]

    def __init__(
        self,
        published_parquet_path: str | Path,
        manifest_path: str | Path,
    ) -> None:
        try:
            parquet = Path(published_parquet_path).resolve(strict=True)
            manifest_file = Path(manifest_path).resolve(strict=True)
        except Exception as exc:
            raise EarningsValidationError(
                "INVALID_PUBLISHED_ARTIFACT",
                f"published artifact paths are invalid: {exc}",
            ) from exc
        verified_manifest, revisions = _load_verified_snapshot(
            parquet, manifest_file
        )
        object.__setattr__(self, "_published_parquet_path", parquet)
        object.__setattr__(self, "_manifest_path", manifest_file)
        object.__setattr__(self, "_manifest", verified_manifest)
        object.__setattr__(self, "_revisions", revisions)

    def __repr__(self) -> str:
        return (
            "PublishedEarningsPITQuery("
            f"build_id={self.build_id!r}, output_sha256={self.output_sha256!r})"
        )

    @classmethod
    def from_persistence_paths(
        cls, paths: EarningsPersistencePaths
    ) -> PublishedEarningsPITQuery:
        """Bind to the published artifact pair represented by persistence paths."""

        if not isinstance(paths, EarningsPersistencePaths):
            _fail(
                "INVALID_PUBLISHED_ARTIFACT",
                "paths must be EarningsPersistencePaths",
            )
        return cls(paths.published_path, paths.manifest_path)

    @property
    def build_id(self) -> str:
        """The verified publication build identity."""

        return self._manifest.build_id

    @property
    def output_sha256(self) -> str:
        """The verified SHA-256 identity of the bound Parquet bytes."""

        return self._manifest.output_sha256

    def query(
        self,
        *,
        canonical_asset_id: str,
        as_of: datetime,
        decision_session: date,
        provider_name: str | None = None,
    ) -> EarningsStateAsOf:
        """Return the nearest qualifying earnings event visible at ``as_of``.

        The PIT cutoff is applied before event identities are grouped or
        replayed.  No later row can therefore reveal an event instance, state
        transition, or schedule correction at an earlier query timestamp.
        """

        _validate_query_inputs(
            canonical_asset_id=canonical_asset_id,
            provider_name=provider_name,
            as_of=as_of,
            decision_session=decision_session,
        )

        histories: dict[tuple[str, str], list[EarningsScheduleRevision]] = (
            defaultdict(list)
        )
        for revision in self._revisions:
            effective_at = revision.strategy_effective_at
            if (
                revision.canonical_asset_id != canonical_asset_id
                or revision.provider_name != provider_name
                or effective_at is None
                or effective_at > as_of
            ):
                continue
            histories[event_history_key(revision)].append(revision)

        candidates: list[EarningsStateAsOf] = []
        for history in histories.values():
            state = reconstruct_earnings_state(history, as_of)
            if (
                state.lifecycle_state in _ACTIVE_LIFECYCLE_STATES
                and state.earnings_schedule_known is True
                and type(state.scheduled_date) is date
                and state.scheduled_date >= decision_session
            ):
                candidates.append(state)

        if not candidates:
            return _unknown_state(
                canonical_asset_id=canonical_asset_id,
                as_of=as_of,
            )

        nearest_date = min(
            state.scheduled_date for state in candidates if state.scheduled_date
        )
        nearest = [
            state for state in candidates if state.scheduled_date == nearest_date
        ]
        if len(nearest) != 1:
            _fail(
                "AMBIGUOUS_ACTIVE_EARNINGS_EVENTS",
                "multiple active event instances share the nearest scheduled_date",
            )
        return nearest[0]
