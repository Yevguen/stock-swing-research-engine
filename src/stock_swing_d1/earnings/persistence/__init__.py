"""Public persistence API for canonical earnings schedule revisions."""

from stock_swing_d1.earnings.persistence.models import (
    CanonicalParquetArtifact,
    EarningsPersistenceError,
    EarningsPersistencePaths,
    EarningsPublicationManifest,
    EarningsPublicationResult,
    EarningsQuarantineRecord,
    QuarantineStage,
)
from stock_swing_d1.earnings.persistence.parquet import (
    read_canonical_revisions,
    read_quarantine_records,
    write_canonical_revisions,
    write_quarantine_records,
)
from stock_swing_d1.earnings.persistence.hashing import (
    canonical_record_hash,
    canonical_sort_key,
    event_history_key,
    revision_identity_key,
)
from stock_swing_d1.earnings.persistence.schema import (
    CANONICAL_EARNINGS_SCHEMA_VERSION,
    CANONICAL_RECORD_HASH_ALGORITHM,
    CANONICAL_SORT_KEY_VERSION,
    OUTPUT_HASH_ALGORITHM,
    canonical_earnings_arrow_schema,
)
from stock_swing_d1.earnings.persistence.publication import (
    publish_earnings_revisions,
    read_publication_manifest,
    write_publication_manifest,
)

__all__ = [
    "CanonicalParquetArtifact",
    "EarningsPersistenceError",
    "EarningsPersistencePaths",
    "EarningsPublicationManifest",
    "EarningsPublicationResult",
    "EarningsQuarantineRecord",
    "QuarantineStage",
    "CANONICAL_EARNINGS_SCHEMA_VERSION",
    "CANONICAL_RECORD_HASH_ALGORITHM",
    "CANONICAL_SORT_KEY_VERSION",
    "OUTPUT_HASH_ALGORITHM",
    "canonical_earnings_arrow_schema",
    "canonical_record_hash",
    "canonical_sort_key",
    "event_history_key",
    "revision_identity_key",
    "read_canonical_revisions",
    "read_quarantine_records",
    "write_canonical_revisions",
    "write_quarantine_records",
    "read_publication_manifest",
    "write_publication_manifest",
    "publish_earnings_revisions",
]
