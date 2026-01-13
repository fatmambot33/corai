"""Planner response configuration."""

from ..structure.plan.plan import PlanStructure
from .config import ResponseConfiguration

PLANNER = ResponseConfiguration(
    name="planner",
    instructions="Generates structured prompts based on user input.",
    tools=None,
    input_structure=None,
    output_structure=PlanStructure,
)
