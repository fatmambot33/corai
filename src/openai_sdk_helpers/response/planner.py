"""Planner response configuration."""

from ..structure.plan.plan import PlanStructure
from .configuration import ResponseConfiguration

PLANNER = ResponseConfiguration(
    name="planner",
    instructions="Generates structured prompts based on user input.",
    tools=None,
    input_structure=None,
    output_structure=PlanStructure,
)
