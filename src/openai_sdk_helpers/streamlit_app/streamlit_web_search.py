"""Developer configuration for the example Streamlit chat app."""

import json
from openai_sdk_helpers.agent.search.web import WebAgentSearch
from openai_sdk_helpers.settings import OpenAISettings
from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.tools import ToolHandlerRegistration
from openai_sdk_helpers.structure.web_search import WebSearchStructure
from openai_sdk_helpers.structure.prompt import PromptStructure
from openai_sdk_helpers.tools import ToolSpec, build_tool_definition_list
from openai_sdk_helpers.utils import coerce_jsonable, customJSONEncoder
from openai_sdk_helpers.environment import DEFAULT_MODEL


class StreamlitWebSearch(ResponseBase[WebSearchStructure]):
    """Response tuned for a generic chat experience with structured output.

    Methods
    -------
    __init__()
        Configure a general-purpose response session using OpenAI settings.
    """

    def __init__(self) -> None:
        settings = OpenAISettings.from_env()
        if not settings.default_model:
            settings = settings.model_copy(update={"default_model": DEFAULT_MODEL})
        tool_spec = ToolSpec(
            input_structure=PromptStructure,
            tool_name="perform_search",
            tool_description="Tool to perform web searches and generate reports.",
            output_structure=WebSearchStructure,
        )
        super().__init__(
            name="streamlit_web_search",
            instructions="Perform web searches and generate reports.",
            tools=build_tool_definition_list([tool_spec]),
            output_structure=WebSearchStructure,
            tool_handlers={
                "perform_search": ToolHandlerRegistration(
                    handler=perform_search,
                    tool_spec=tool_spec,
                )
            },
            openai_settings=settings,
        )


async def perform_search(tool) -> str:
    """Perform a web search and return structured results."""
    structured_data = PromptStructure.from_string(tool.arguments)
    web_result = await WebAgentSearch(default_model=DEFAULT_MODEL).run_agent_async(
        structured_data.prompt
    )
    payload = coerce_jsonable(web_result)
    return json.dumps(payload, cls=customJSONEncoder)


APP_CONFIG = {
    "response": StreamlitWebSearch,
    "display_title": "Web Search Assistant",
    "description": "configuration-driven chat experience for performing web searches.",
}

if __name__ == "__main__":
    web_search_instance = StreamlitWebSearch()
    import asyncio

    result = asyncio.run(
        web_search_instance.run_async("What are the 2026 advancements in AI?")
    )
    if result:
        print(web_search_instance.get_last_tool_message())
    else:
        print("No result returned.")
    filepath = f"./data/{web_search_instance.name}.{web_search_instance.uuid}.json"
    web_search_instance.save(filepath)
    web_search_instance.close()
