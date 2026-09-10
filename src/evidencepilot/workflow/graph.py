"""LangGraph workflow composition."""

from .nodes import WorkflowNodes


class ResearchWorkflow(WorkflowNodes):
    """Public workflow assembled from the node implementation."""
