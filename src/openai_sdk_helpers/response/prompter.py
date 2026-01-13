"""Prompter response configuration."""

from .config import ResponseConfiguration
from ..structure.prompt import PromptStructure

PROMPTER = ResponseConfiguration(
    name="prompter",
    instructions="Generates structured prompts based on user input.",
    tools=None,
    input_structure=None,
    output_structure=PromptStructure,
)
