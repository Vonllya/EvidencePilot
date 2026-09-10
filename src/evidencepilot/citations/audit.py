"""Citation auditing and post-revision verification."""

import json
from typing import Any

from ..models import (
    AtomicClaim,
    CitationAudit,
    CitationBatch,
    CitationCheck,
    ClaimPatchBatch,
    ResearchState,
)
from ..providers import FakeLLMProvider
from .metrics import citation_coverage
from .parser import CitationParserMixin
from .patch import CitationPatchMixin


class CitationAuditMixin(CitationParserMixin, CitationPatchMixin):
    @staticmethod
    def _evidence_quality(source_ids: list[str], source_map: dict[str, dict[str, Any]]) -> str:
        if not source_ids or any(source_id not in source_map for source_id in source_ids):
            return "unavailable"
        statuses = {source_map[source_id].get("source_status", "fetched") for source_id in source_ids}
        if statuses == {"fetched"}:
            return "fetched"
        if statuses == {"snippet_fallback"}:
            return "snippet_fallback"
        return "mixed"

    async def _batch_citation_audit(
        self, claims: list[AtomicClaim], source_map: dict[str, dict[str, Any]]
    ) -> CitationAudit:
        checks: list[CitationCheck] = []
        requests: list[dict[str, str]] = []
        valid = 0
        for claim in claims:
            if not claim.source_ids:
                checks.append(CitationCheck(
                    claim_id=claim.claim_id, source_ids=[], claim=claim.text,
                    verdict="unsupported", evidence_quality="unavailable",
                    reason="The claim has no adjacent citation.",
                ))
                continue
            quality = self._evidence_quality(claim.source_ids, source_map)
            if quality == "unavailable":
                checks.append(CitationCheck(
                    claim_id=claim.claim_id, source_ids=claim.source_ids, claim=claim.text,
                    verdict="unsupported", evidence_quality=quality,
                    reason="At least one cited source_id does not exist.",
                ))
                continue
            valid += 1
            requests.append({
                "claim_id": claim.claim_id,
                "source_ids": claim.source_ids,
                "claim": claim.text,
            })
        if isinstance(self.llm, FakeLLMProvider):
            checks.extend(CitationCheck(
                claim_id=item["claim_id"], source_ids=item["source_ids"], claim=item["claim"],
                verdict="supported",
                evidence_quality=self._evidence_quality(item["source_ids"], source_map),
                reason="The cited bundled source contains the extracted passage.",
            ) for item in requests)
        elif requests:
            cited_ids = sorted({source_id for item in requests for source_id in item["source_ids"]})
            source_payload = [{
                "source_id": source_id,
                "content": source_map[source_id]["content"],
            } for source_id in cited_ids]
            prompt = (
                "Audit sentence-level atomic claims in one batch. For each claim_id, judge whether its "
                "cited source_ids AS A SET jointly support the complete claim. A source need not support "
                "the entire claim individually. Use supported only when the source set supports every "
                "material part, partially_supported for incomplete or broader-than-source claims, and "
                "unsupported for no material support or contradiction. Return exactly one check per "
                "claim_id. Evidence transport quality is evaluated separately; do not use it to change "
                "the semantic verdict. Do not rewrite claims.\nCLAIMS: "
                + json.dumps(requests, ensure_ascii=False)
                + "\nSOURCES: " + json.dumps(source_payload, ensure_ascii=False)
            )
            batch = await self.llm.structured(
                prompt + "\nReturn only JSON.", CitationBatch, node="verify_citations"
            )
            decisions = {item.claim_id: item for item in batch.checks}
            for item in requests:
                decision = decisions.get(item["claim_id"])
                checks.append(CitationCheck(
                    claim_id=item["claim_id"], source_ids=item["source_ids"], claim=item["claim"],
                    verdict=decision.verdict if decision else "unsupported",
                    evidence_quality=self._evidence_quality(item["source_ids"], source_map),
                    reason=decision.reason if decision else "The batch audit omitted this citation.",
                ))
        order = {claim.claim_id: index for index, claim in enumerate(claims)}
        checks.sort(key=lambda check: order.get(check.claim_id, len(order)))
        total = len(claims)
        return CitationAudit(
            checks=checks, valid_citations=valid, total_citations=total,
            coverage=citation_coverage(valid, total),
        )

    async def verify_citations(self, state: ResearchState) -> dict[str, Any]:
        source_map = {s["source_id"]: s for s in state.get("sources", [])}
        report = state.get("report", "")
        original_claims = self._atomic_claims(report)
        before = await self._batch_citation_audit(original_claims, source_map)
        problems = [check for check in before.checks if check.verdict != "supported"]
        revision = {
            "attempted": False, "problem_claims": len(problems), "rechecked_claims": 0,
            "patches_applied": 0,
            "before": before.model_dump(), "after": before.model_dump(),
        }
        if problems and not isinstance(self.llm, FakeLLMProvider):
            cited_ids = sorted(source_map)
            prompt = (
                "Create local structured patches only for the listed problem atomic claims. Use replace "
                "to narrow a claim to one independently verifiable sentence with citations immediately "
                "next to the supported fact, or delete when no support exists. Do not edit supported "
                "claims or other report text. Replacement source IDs must come from ALLOWED SOURCES.\n"
                "PROBLEM CLAIMS:\n" + json.dumps([c.model_dump() for c in problems], ensure_ascii=False)
                + "\nALLOWED SOURCES:\n" + json.dumps([{
                    "source_id": source_id, "title": source_map[source_id]["title"],
                    "url": str(source_map[source_id]["url"]),
                    "content": source_map[source_id]["content"],
                } for source_id in cited_ids], ensure_ascii=False)
            )
            patch_batch = await self.llm.structured(
                prompt + "\nReturn only JSON.", ClaimPatchBatch, node="revise_unsupported_claims"
            )
            problem_map = {claim.claim_id: claim for claim in original_claims if claim.claim_id in {c.claim_id for c in problems}}
            report, applied = self._apply_claim_patches(
                report, patch_batch, problem_map, source_map
            )
            final_claims = self._atomic_claims(report)
            # Claim IDs and offsets belong to a specific report revision. Re-audit the
            # complete new set instead of treating non-unique prose as identity.
            after = await self._batch_citation_audit(final_claims, source_map)
            revision.update(
                attempted=True, patches_applied=applied, rechecked_claims=len(final_claims),
                after=after.model_dump(),
            )
            return {
                "report": report, "citation_audit": after.model_dump(),
                "citation_revision": revision,
            }
        return {"citation_audit": before.model_dump(), "citation_revision": revision}




async def audit_claims(auditor: CitationAuditMixin, claims: list[AtomicClaim], source_map: dict[str, dict[str, Any]]) -> CitationAudit:
    return await auditor._batch_citation_audit(claims, source_map)
