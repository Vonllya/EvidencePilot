"""Citation metric calculations."""


def citation_coverage(valid_claims: int, total_claims: int) -> float:
    return valid_claims / total_claims if total_claims else 0.0
