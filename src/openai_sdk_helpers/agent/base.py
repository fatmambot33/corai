"""Base agent helpers built on the OpenAI Agents SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from agents import Agent, RunResult, RunResultStreaming, Runner
from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool, Tool
from jinja2 import Template

from .runner import run_async, run_streamed, run_sync


class AgentConfigurationLike(Protocol):
    """Protocol describing the configuration attributes for BaseAgent."""

    @property
    def name(self) -> str:
        """Agent name."""
        ...

    @property
    def description(self) -> Optional[str]:
        """Agent description."""
        ...

    @property
    def model(self) -> Optional[str]:
        """Model identifier."""
        ...

    @property
    def template_path(self) -> Optional[Any]:
        """Template path."""
        ...

    @property
    def instructions(self) -> Optional[Any]:
        """Instructions."""
        ...

    @property
    def input_type(self) -> Optional[Any]:
        """Input type."""
        ...

    @property
    def output_type(self) -> Optional[Any]:
        """Output type."""
        ...

    @property
    def tools(self) -> Optional[Any]:
        """Tools."""
        ...

    @property
    def model_settings(self) -> Optional[Any]:
        """Model settings."""
        ...

    @property
    def handoffs(self) -> Optional[Any]:
        """Handoffs."""
        ...

    @property
    def input_guardrails(self) -> Optional[Any]:
        """Input guardrails."""
        ...

    @property
    def output_guardrails(self) -> Optional[Any]:
        """Output guardrails."""
        ...

    @property
    def session(self) -> Optional[Any]:
        """Session."""
        ...


class BaseAgent:
    """Factory for creating and configuring specialized agents.

    ``BaseAgent`` provides the foundation for building OpenAI agents with support
    for Jinja2 prompt templates, custom tools, handoffs for agent delegation,
    input and output guardrails for validation, session management for
    conversation history, and both synchronous and asynchronous execution modes.
    All specialized agents in this package extend this base class.

    Examples
    --------
    Create a basic agent from configuration:

    >>> from openai_sdk_helpers.agent import BaseAgent, AgentConfig
    >>> config = AgentConfig(
    ...     name="my_agent",
    ...     description="A custom agent",
    ...     model="gpt-4o-mini"
    ... )
    >>> agent = BaseAgent(config=config, default_model="gpt-4o-mini")
    >>> result = agent.run_sync("What is 2+2?")

    Use absolute path to template:

    >>> config = AgentConfig(
    ...     name="my_agent",
    ...     template_path="/absolute/path/to/template.jinja",
    ...     model="gpt-4o-mini"
    ... )
    >>> agent = BaseAgent(config=config, default_model="gpt-4o-mini")

    Use async execution:

    >>> import asyncio
    >>> async def main():
    ...     result = await agent.run_async("Explain quantum physics")
    ...     return result
    >>> asyncio.run(main())

    Methods
    -------
    from_config(config, run_context_wrapper, prompt_dir, default_model)
        Instantiate a ``BaseAgent`` from configuration.
    build_prompt_from_jinja(run_context_wrapper)
        Render the agent prompt using Jinja and optional context.
    get_prompt(run_context_wrapper, _)
        Render the agent prompt using the provided run context.
    get_agent()
        Construct the configured :class:`agents.Agent` instance.
    run_async(input, context, output_type, session)
        Execute the agent asynchronously and optionally cast the result.
    run_sync(input, context, output_type, session)
        Execute the agent synchronously.
    run_streamed(input, context, output_type, session)
        Return a streaming result for the agent execution.
    as_tool()
        Return the agent as a callable tool.
    close()
        Clean up agent resources (can be overridden by subclasses).
    """

    def __init__(
        self,
        config: AgentConfigurationLike,
        run_context_wrapper: Optional[RunContextWrapper[Dict[str, Any]]] = None,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Initialize the BaseAgent using a configuration object.

        Parameters
        ----------
        config : AgentConfigurationLike
            Configuration describing this agent.
        run_context_wrapper : RunContextWrapper or None, default=None
            Optional wrapper providing runtime context for prompt rendering.
        prompt_dir : Path or None, default=None
            Optional directory holding prompt templates. Used when
            ``config.template_path`` is not provided or is relative. If
            ``config.template_path`` is an absolute path, this parameter is
            ignored.
        default_model : str or None, default=None
            Optional fallback model identifier if the config does not supply one.
        """
        name = config.name
        description = config.description or ""
        model = config.model or default_model
        if not model:
            raise ValueError("Model is required to construct the agent.")

        prompt_path: Optional[Path]
        if config.template_path:
            prompt_path = Path(config.template_path)
        elif prompt_dir is not None:
            prompt_path = prompt_dir / f"{name}.jinja"
        else:
            prompt_path = None

        if prompt_path is None:
            self._template = Template("")
        elif prompt_path.exists():
            self._template = Template(prompt_path.read_text())
        else:
            raise FileNotFoundError(
                f"Prompt template for agent '{name}' not found at {prompt_path}."
            )

        self.agent_name = name
        self.description = description
        self.model = model

        self._input_type = config.input_type
        self._output_type = config.output_type or config.input_type
        self._tools = config.tools
        self._model_settings = config.model_settings
        self._handoffs = config.handoffs
        self._input_guardrails = config.input_guardrails
        self._output_guardrails = config.output_guardrails
        self._session = config.session
        self._run_context_wrapper = run_context_wrapper

        # Store instructions if provided directly in config
        self._instructions = getattr(config, "instructions", None)

    @classmethod
    def from_config(
        cls,
        config: AgentConfigurationLike,
        *,
        run_context_wrapper: Optional[RunContextWrapper[Dict[str, Any]]] = None,
        prompt_dir: Optional[Path] = None,
        default_model: Optional[str] = None,
    ) -> BaseAgent:
        """Create an BaseAgent instance from configuration.

        Parameters
        ----------
        config : AgentConfigurationLike
            Configuration describing the agent.
        run_context_wrapper : RunContextWrapper or None, default=None
            Optional wrapper providing runtime context.
        prompt_dir : Path or None, default=None
            Optional directory holding prompt templates. Used when
            ``config.template_path`` is not provided or is relative. If
            ``config.template_path`` is an absolute path, this parameter is
            ignored.
        default_model : str or None, default=None
            Optional fallback model identifier.

        Returns
        -------
        BaseAgent
            Instantiated agent.
        """
        return cls(
            config=config,
            run_context_wrapper=run_context_wrapper,
            prompt_dir=prompt_dir,
            default_model=default_model,
        )

    def _build_prompt_from_jinja(self) -> str:
        """Render the instructions prompt for this agent.

        Returns
        -------
        str
            Prompt text rendered from the Jinja template.
        """
        return self.build_prompt_from_jinja(
            run_context_wrapper=self._run_context_wrapper
        )

    def build_prompt_from_jinja(
        self, run_context_wrapper: Optional[RunContextWrapper[Dict[str, Any]]] = None
    ) -> str:
        """Render the agent prompt using the provided run context.

        Parameters
        ----------
        run_context_wrapper : RunContextWrapper or None, default=None
            Wrapper whose ``context`` dictionary is used to render the Jinja
            template.

        Returns
        -------
        str
            Rendered prompt text.
        """
        context = {}
        if run_context_wrapper is not None:
            context = run_context_wrapper.context

        return self._template.render(context)

    def get_prompt(
        self, run_context_wrapper: RunContextWrapper[Dict[str, Any]], *, _: Agent
    ) -> str:
        """Render the agent prompt using the provided run context.

        Parameters
        ----------
        run_context_wrapper : RunContextWrapper
            Wrapper around the current run context whose ``context`` dictionary
            is used to render the Jinja template.
        _ : Agent
            Underlying Agent instance (ignored).

        Returns
        -------
        str
            The rendered prompt.
        """
        return self.build_prompt_from_jinja(run_context_wrapper)

    def get_agent(self) -> Agent:
        """Construct and return the configured :class:`agents.Agent` instance.

        Returns
        -------
        Agent
            Initialized agent ready for execution.
        """
        agent_config: Dict[str, Any] = {
            "name": self.agent_name,
            "instructions": self._build_prompt_from_jinja() or ".",
            "model": self.model,
        }
        if self._output_type:
            agent_config["output_type"] = self._output_type
        if self._tools:
            agent_config["tools"] = self._tools
        if self._model_settings:
            agent_config["model_settings"] = self._model_settings
        if self._handoffs:
            agent_config["handoffs"] = self._handoffs
        if self._input_guardrails:
            agent_config["input_guardrails"] = self._input_guardrails
        if self._output_guardrails:
            agent_config["output_guardrails"] = self._output_guardrails

        return Agent(**agent_config)

    async def run_async(
        self,
        input: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        output_type: Optional[Any] = None,
        session: Optional[Any] = None,
    ) -> Any:
        """Execute the agent asynchronously.

        Parameters
        ----------
        input : str
            Prompt or query for the agent.
        context : dict or None, default=None
            Optional dictionary passed to the agent.
        output_type : type or None, default=None
            Optional type used to cast the final output.
        session : Session or None, default=None
            Optional session for maintaining conversation history across runs.
            If not provided, uses the session from config if available.

        Returns
        -------
        Any
            Agent result, optionally converted to ``output_type``.
        """
        if self._output_type is not None and output_type is None:
            output_type = self._output_type
        # Use session from parameter, fall back to config session
        session_to_use = session if session is not None else self._session
        return await run_async(
            agent=self.get_agent(),
            input=input,
            context=context,
            output_type=output_type,
            session=session_to_use,
        )

    def run_sync(
        self,
        input: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        output_type: Optional[Any] = None,
        session: Optional[Any] = None,
    ) -> Any:
        """Run the agent synchronously.

        Parameters
        ----------
        input : str
            Prompt or query for the agent.
        context : dict or None, default=None
            Optional dictionary passed to the agent.
        output_type : type or None, default=None
            Optional type used to cast the final output.
        session : Session or None, default=None
            Optional session for maintaining conversation history across runs.
            If not provided, uses the session from config if available.

        Returns
        -------
        Any
            Agent result, optionally converted to ``output_type``.
        """
        # Use session from parameter, fall back to config session
        session_to_use = session if session is not None else self._session
        return run_sync(
            agent=self.get_agent(),
            input=input,
            context=context,
            output_type=output_type,
            session=session_to_use,
        )

    def run_streamed(
        self,
        input: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        output_type: Optional[Any] = None,
        session: Optional[Any] = None,
    ) -> RunResultStreaming:
        """Stream the agent execution results.

        Parameters
        ----------
        input : str
            Prompt or query for the agent.
        context : dict or None, default=None
            Optional dictionary passed to the agent.
        output_type : type or None, default=None
            Optional type used to cast the final output.
        session : Session or None, default=None
            Optional session for maintaining conversation history across runs.
            If not provided, uses the session from config if available.

        Returns
        -------
        RunResultStreaming
            Streaming output wrapper from the agent execution.
        """
        # Use session from parameter, fall back to config session
        session_to_use = session if session is not None else self._session
        result = run_streamed(
            agent=self.get_agent(),
            input=input,
            context=context,
            session=session_to_use,
        )
        if self._output_type and not output_type:
            output_type = self._output_type
        if output_type:
            return result.final_output_as(output_type)
        return result

    def as_tool(self) -> Tool:
        """Return the agent as a callable tool.

        Returns
        -------
        Tool
            Tool instance wrapping this agent.
        """
        agent = self.get_agent()
        tool_obj: Tool = agent.as_tool(
            tool_name=self.agent_name, tool_description=self.description
        )
        return tool_obj

    def __enter__(self) -> BaseAgent:
        """Enter the context manager for resource management.

        Returns
        -------
        BaseAgent
            Self reference for use in with statements.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and clean up resources.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred, otherwise None.
        exc_val : Exception or None
            Exception instance if an exception occurred, otherwise None.
        exc_tb : traceback or None
            Traceback object if an exception occurred, otherwise None.
        """
        self.close()

    def close(self) -> None:
        """Clean up agent resources.

        This method is called automatically when using the agent as a
        context manager. Override in subclasses to implement custom
        cleanup logic.

        Examples
        --------
        >>> agent = BaseAgent(config, default_model="gpt-4o-mini")
        >>> try:
        ...     result = agent.run_sync("query")
        ... finally:
        ...     agent.close()
        """
        # Base implementation does nothing, but subclasses can override
        pass


__all__ = ["AgentConfigurationLike", "BaseAgent"]
