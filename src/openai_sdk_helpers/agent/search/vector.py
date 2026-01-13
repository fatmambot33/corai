"""Core workflow management for ``vector search``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agents import custom_span, gen_trace_id, trace

from ...structure.prompt import PromptStructure
from ...structure.vector_search import (
    VectorSearchItemStructure,
    VectorSearchItemResultStructure,
    VectorSearchItemResultsStructure,
    VectorSearchStructure,
    VectorSearchPlanStructure,
    VectorSearchReportStructure,
)
from ...tools import tool_handler_factory
from ...vector_storage import VectorStorage
from ..config import AgentConfiguration
from ..utils import run_coroutine_agent_sync
from .base import SearchPlanner, SearchToolAgent, SearchWriter

MAX_CONCURRENT_SEARCHES = 10


class VectorAgentPlanner(SearchPlanner[VectorSearchPlanStructure]):
    """Plan vector searches to satisfy a user query.

    Parameters
    ----------
    prompt_dir : Path or None, default=None
        Directory containing prompt templates.
    default_model : str or None, default=None
        Default model identifier to use when not defined in config.

    Methods
    -------
    run_agent(query)
        Generate a vector search plan for the provided query.

    Raises
    ------
    ValueError
        If the default model is not provided.

    Examples
    --------
    >>> planner = VectorSearchPlanner(default_model="gpt-4o-mini")
    """

    def __init__(
        self, prompt_dir: Optional[Path] = None, default_model: Optional[str] = None
    ) -> None:
        """Initialize the planner agent."""
        super().__init__(prompt_dir=prompt_dir, default_model=default_model)

    def _configure_agent(self) -> AgentConfiguration:
        """Return configuration for the vector planner agent.

        Returns
        -------
        AgentConfiguration
            Configuration with name, description, and output type.
        """
        return AgentConfiguration(
            name="vector_planner",
            instructions="Agent instructions",
            description="Plan vector searches based on a user query.",
            output_structure=VectorSearchPlanStructure,
        )


class VectorSearchTool(
    SearchToolAgent[
        VectorSearchItemStructure,
        VectorSearchItemResultStructure,
        VectorSearchPlanStructure,
    ]
):
    """Execute vector searches defined in a search plan.

    Parameters
    ----------
    prompt_dir : Path or None, default=None
        Directory containing prompt templates.
    default_model : str or None, default=None
        Default model identifier to use when not defined in config.
    store_name : str or None, default=None
        Name of the vector store to query.
    max_concurrent_searches : int, default=MAX_CONCURRENT_SEARCHES
        Maximum number of concurrent vector search tasks to run.
    vector_storage : VectorStorage or None, default=None
        Optional preconfigured vector storage instance to reuse.
    vector_storage_factory : Callable or None, default=None
        Factory for constructing a VectorStorage when one is not provided.
        Receives ``store_name`` as an argument.

    Methods
    -------
    run_agent(search_plan)
        Execute searches described by the plan.
    run_search(item)
        Perform a single vector search and summarise the result.

    Raises
    ------
    ValueError
        If the default model is not provided.

    Examples
    --------
    >>> tool = VectorSearchTool(default_model="gpt-4o-mini", store_name="my_store")
    """

    def __init__(
        self,
        *,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
        store_name: Optional[str] = None,
        max_concurrent_searches: int = MAX_CONCURRENT_SEARCHES,
        vector_storage: Optional[VectorStorage] = None,
        vector_storage_factory: Optional[Callable[[str], VectorStorage]] = None,
    ) -> None:
        """Initialize the search tool agent."""
        self._vector_storage: Optional[VectorStorage] = None
        self._store_name = store_name or "editorial"
        self._vector_storage_factory = vector_storage_factory
        if vector_storage is not None:
            self._vector_storage = vector_storage
        super().__init__(
            prompt_dir=prompt_dir,
            default_model=default_model,
            max_concurrent_searches=max_concurrent_searches,
        )

    def _configure_agent(self) -> AgentConfiguration:
        """Return configuration for the vector search tool agent.

        Returns
        -------
        AgentConfiguration
            Configuration with name, description, and input type.
        """
        return AgentConfiguration(
            name="vector_search",
            instructions="Agent instructions",
            description="Perform vector searches based on a search plan.",
            input_structure=VectorSearchPlanStructure,
            output_structure=VectorSearchItemResultsStructure,
        )

    def _get_vector_storage(self) -> VectorStorage:
        """Return a cached vector storage instance.

        Returns
        -------
        VectorStorage
            Vector storage helper for executing searches.
        """
        if self._vector_storage is None:
            if self._vector_storage_factory is not None:
                self._vector_storage = self._vector_storage_factory(self._store_name)
            else:
                self._vector_storage = VectorStorage(store_name=self._store_name)
        return self._vector_storage

    async def run_search(
        self, item: VectorSearchItemStructure
    ) -> VectorSearchItemResultStructure:
        """Perform a single vector search using the search tool.

        Parameters
        ----------
        item : VectorSearchItemStructure
            Search item containing the query and reason.

        Returns
        -------
        VectorSearchItemResultStructure
            Summarized search result. The ``texts`` attribute is empty when no
            results are found.
        """
        results = self._get_vector_storage().search(item.query)
        if results is None:
            texts: List[str] = []
        else:
            texts = [
                content.text
                for result in results.data
                for content in (result.content or [])
                if getattr(content, "text", None)
            ]
        return VectorSearchItemResultStructure(texts=texts)


class VectorSearchWriter(SearchWriter[VectorSearchReportStructure]):
    """Generate reports summarizing vector search results.

    Parameters
    ----------
    prompt_dir : Path or None, default=None
        Directory containing prompt templates.
    default_model : str or None, default=None
        Default model identifier to use when not defined in config.

    Methods
    -------
    run_agent(query, search_results)
        Compile a final report from search results.

    Raises
    ------
    ValueError
        If the default model is not provided.

    Examples
    --------
    >>> writer = VectorSearchWriter(default_model="gpt-4o-mini")
    """

    def __init__(
        self, prompt_dir: Optional[Path] = None, default_model: Optional[str] = None
    ) -> None:
        """Initialize the writer agent."""
        super().__init__(prompt_dir=prompt_dir, default_model=default_model)

    def _configure_agent(self) -> AgentConfiguration:
        """Return configuration for the vector writer agent.

        Returns
        -------
        AgentConfiguration
            Configuration with name, description, and output type.
        """
        return AgentConfiguration(
            name="vector_writer",
            instructions="Agent instructions",
            description="Write a report based on search results.",
            output_structure=VectorSearchReportStructure,
        )


class VectorAgentSearch:
    """Manage the complete vector search workflow.

    This high-level agent orchestrates a multi-step research process that plans
    searches, executes them concurrently against a vector store, and generates
    comprehensive reports. It combines ``VectorSearchPlanner``,
    ``VectorSearchTool``, and ``VectorSearchWriter`` into a single workflow.

    Parameters
    ----------
    prompt_dir : Path or None, default=None
        Directory containing prompt templates.
    default_model : str or None, default=None
        Default model identifier to use when not defined in config.
    vector_store_name : str or None, default=None
        Name of the vector store to query.
    max_concurrent_searches : int, default=MAX_CONCURRENT_SEARCHES
        Maximum number of concurrent search tasks to run.
    vector_storage : VectorStorage or None, default=None
        Optional preconfigured vector storage instance to reuse.
    vector_storage_factory : callable, default=None
        Factory used to construct a VectorStorage when one is not provided.
        Receives ``vector_store_name`` as an argument.

    Examples
    --------
    Basic vector search:

    >>> from pathlib import Path
    >>> from openai_sdk_helpers.agent.search.vector import VectorSearch
    >>> prompts = Path("./prompts")
    >>> search = VectorSearch(prompt_dir=prompts, default_model="gpt-4o-mini")
    >>> result = search.run_agent_sync("What are the key findings in recent AI research?")
    >>> print(result.report.report)

    Custom vector store:

    >>> from openai_sdk_helpers.vector_storage import VectorStorage
    >>> storage = VectorStorage(store_name="research_papers")
    >>> search = VectorSearch(
    ...     default_model="gpt-4o-mini",
    ...     vector_storage=storage,
    ...     max_concurrent_searches=5
    ... )

    Methods
    -------
    run_agent(search_query)
        Execute the research workflow asynchronously.
    run_agent_sync(search_query)
        Execute the research workflow synchronously.
    as_response_tool(vector_store_name, tool_name, tool_description)
        Build a Responses API tool definition and handler.
    run_vector_agent(search_query)
        Convenience asynchronous entry point for the workflow.
    run_vector_agent_sync(search_query)
        Convenience synchronous entry point for the workflow.

    Raises
    ------
    ValueError
        If the default model is not provided.
    """

    def __init__(
        self,
        *,
        vector_store_name: str,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
        max_concurrent_searches: int = MAX_CONCURRENT_SEARCHES,
        vector_storage: Optional[VectorStorage] = None,
        vector_storage_factory: Optional[Callable[[str], VectorStorage]] = None,
    ) -> None:
        """Create the main VectorSearch agent."""
        self._prompt_dir = prompt_dir
        self._default_model = default_model
        self._vector_store_name = vector_store_name
        self._max_concurrent_searches = max_concurrent_searches
        self._vector_storage = vector_storage
        self._vector_storage_factory = vector_storage_factory

    async def run_agent(self, search_query: str) -> VectorSearchStructure:
        """Execute the entire research workflow for ``search_query``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        VectorSearchStructure
            Completed research output.
        """
        trace_id = gen_trace_id()
        with trace("VectorSearch trace", trace_id=trace_id):
            planner = VectorAgentPlanner(
                prompt_dir=self._prompt_dir, default_model=self._default_model
            )
            tool = VectorSearchTool(
                prompt_dir=self._prompt_dir,
                default_model=self._default_model,
                store_name=self._vector_store_name,
                max_concurrent_searches=self._max_concurrent_searches,
                vector_storage=self._vector_storage,
                vector_storage_factory=self._vector_storage_factory,
            )
            writer = VectorSearchWriter(
                prompt_dir=self._prompt_dir, default_model=self._default_model
            )
            with custom_span("vector_search.plan"):
                search_plan = await planner.run_agent(query=search_query)
            with custom_span("vector_search.search"):
                search_results_list = await tool.run_agent(search_plan=search_plan)
            with custom_span("vector_search.write"):
                search_report = await writer.run_agent(
                    search_query, search_results_list
                )
        search_results = VectorSearchItemResultsStructure(
            item_results=search_results_list
        )
        return VectorSearchStructure(
            query=search_query,
            plan=search_plan,
            results=search_results,
            report=search_report,
        )

    def run_agent_sync(self, search_query: str) -> VectorSearchStructure:
        """Run :meth:`run_agent` synchronously for ``search_query``.

        Parameters
        ----------
        search_query : str
            User's research query.

        Returns
        -------
        VectorSearchStructure
            Completed research output.
        """
        return run_coroutine_agent_sync(self.run_agent(search_query))

    def as_response_tool(
        self,
        *,
        tool_name: str = "vector_search",
        tool_description: str = "Run the vector search workflow.",
    ) -> tuple[dict[str, Callable[..., Any]], dict[str, Any]]:
        """Return a Responses API tool handler and definition.

        Parameters
        ----------
        vector_store_name : str
            Name of the vector store to use for the response tool.
        tool_name : str, default="vector_search"
            Name to use for the response tool.
        tool_description : str, default="Run the vector search workflow."
            Description for the response tool.

        Returns
        -------
        tuple[dict[str, Callable[..., Any]], dict[str, Any]]
            Tool handler mapping and tool definition for Responses API usage.
        """
        search = VectorAgentSearch(
            prompt_dir=self._prompt_dir,
            default_model=self._default_model,
            vector_store_name=self._vector_store_name,
            max_concurrent_searches=self._max_concurrent_searches,
            vector_storage=self._vector_storage,
            vector_storage_factory=self._vector_storage_factory,
        )

        def _run_search(prompt: str) -> VectorSearchStructure:
            return run_coroutine_agent_sync(search.run_agent(search_query=prompt))

        tool_handler = {
            tool_name: tool_handler_factory(_run_search, input_model=PromptStructure)
        }
        tool_definition = PromptStructure.response_tool_definition(
            tool_name, tool_description=tool_description
        )
        return tool_handler, tool_definition


__all__ = [
    "MAX_CONCURRENT_SEARCHES",
    "VectorAgentPlanner",
    "VectorSearchTool",
    "VectorSearchWriter",
    "VectorAgentSearch",
]
