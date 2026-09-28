"""Public Phase 6C.7A provider-neutral normalization API."""

from stock_swing_d1.earnings.normalization.identity import (
    EarningsCanonicalHistoryReader,
    EarningsSecurityIdentityResolver,
)
from stock_swing_d1.earnings.normalization.models import (
    EarningsAdapterBatch,
    EarningsDeliveryResult,
    EarningsKnowledgeCutoff,
    EarningsNormalizationFailure,
    EarningsNormalizationResult,
)
from stock_swing_d1.earnings.normalization.pipeline import (
    EarningsProviderDeliveryAdapter,
    normalize_earnings_delivery,
    process_earnings_delivery,
)

__all__ = [
    "EarningsKnowledgeCutoff",
    "EarningsNormalizationFailure",
    "EarningsAdapterBatch",
    "EarningsNormalizationResult",
    "EarningsDeliveryResult",
    "EarningsSecurityIdentityResolver",
    "EarningsCanonicalHistoryReader",
    "EarningsProviderDeliveryAdapter",
    "normalize_earnings_delivery",
    "process_earnings_delivery",
]
