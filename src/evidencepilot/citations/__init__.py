"""Citation parsing, auditing, patching, and metric helpers."""

from .audit import audit_claims
from .metrics import citation_coverage
from .parser import parse_atomic_claims
from .patch import apply_claim_patches

__all__ = ["apply_claim_patches", "audit_claims", "citation_coverage", "parse_atomic_claims"]
