import pytest
from pydantic import ValidationError

from evidencepilot.models import ResearchPlan, SubQuestion


def test_research_plan_validation():
    plan = ResearchPlan(objective="Understand a technical system", subquestions=[
        SubQuestion(question=f"Question number {i}?", search_queries=[f"query {i}"]) for i in range(3)
    ])
    assert len(plan.subquestions) == 3


def test_research_plan_rejects_too_few_subquestions():
    with pytest.raises(ValidationError):
        ResearchPlan(objective="Understand a technical system", subquestions=[])
