"""Offset-safe citation patching."""

from typing import Any

from ..models import AtomicClaim, ClaimPatchBatch
from .parser import CitationParserMixin


class CitationPatchMixin:
    @staticmethod
    def _apply_claim_patches(
        report: str, patch_batch: ClaimPatchBatch, problem_map: dict[str, AtomicClaim],
        source_map: dict[str, dict[str, Any]],
    ) -> tuple[str, int]:
        edits: list[tuple[int, int, str]] = []
        used_claims: set[str] = set()
        for patch in patch_batch.patches:
            claim = problem_map.get(patch.claim_id)
            if not claim or patch.claim_id in used_claims:
                continue
            if report[claim.start_offset:claim.end_offset] != claim.text:
                continue
            replacement = patch.replacement.strip() if patch.action == "replace" else ""
            replacement_ids = CitationParserMixin._inline_source_ids(replacement)
            if patch.action == "replace" and (
                not replacement_ids or any(source_id not in source_map for source_id in replacement_ids)
            ):
                continue
            edits.append((claim.start_offset, claim.end_offset, replacement))
            used_claims.add(patch.claim_id)
        for start_offset, end_offset, replacement in sorted(edits, reverse=True):
            report = report[:start_offset] + replacement + report[end_offset:]
        return report, len(edits)




def apply_claim_patches(report: str, patch_batch: ClaimPatchBatch, problem_map: dict[str, AtomicClaim], source_map: dict[str, dict[str, Any]]) -> tuple[str, int]:
    return CitationPatchMixin._apply_claim_patches(report, patch_batch, problem_map, source_map)
