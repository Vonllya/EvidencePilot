import pytest
from pydantic import ValidationError

from evidencepilot.config import Settings
from evidencepilot.models import ResearchPlan, SubQuestion


def test_research_plan_validation():
    plan = ResearchPlan(objective="Understand a technical system", subquestions=[
        SubQuestion(question=f"Question number {i}?", search_queries=[f"query {i}"]) for i in range(3)
    ])
    assert len(plan.subquestions) == 3


def test_research_plan_rejects_too_few_subquestions():
    with pytest.raises(ValidationError):
        ResearchPlan(objective="Understand a technical system", subquestions=[])


def test_environment_provider_selection_is_required(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings.from_env()


def test_search_timeout_must_be_positive():
    with pytest.raises(ValueError, match="SEARCH_TIMEOUT_SECONDS"):
        Settings(search_timeout_seconds=0)
