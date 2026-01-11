"""Base agent helpers built on the OpenAI Agents SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, cast
import uuid

from agents import (
    Agent,
    Handoff,
    InputGuardrail,
    OutputGuardrail,
    RunResultStreaming,
    Session,
)
from agents.model_settings import ModelSettings
from agents.run_context import RunContextWrapper
from agents.tool import Tool
from jinja2 import Template

from ..utils.json.data_class import DataclassJSONSerializable
from ..structure.base import BaseStructure

from ..utils import (
    check_filepath,
    log,
)

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
    def template_path(self) -> Optional[str | Path]:
        """Template path."""
        ...

    def resolve_prompt_path(self, prompt_dir: Path | None = None) -> Path | None:
        """Resolve the prompt template path."""
        ...

    @property
    def instructions(self) -> str | Path:
        """Instructions."""
        ...

    @property
    def instructions_text(self) -> str:
        """Resolved instructions text."""
        ...

    @property
    def input_type(self) -> Optional[type[BaseStructure]]:
        """Input type."""
        ...

    @property
    def output_type(self) -> Optional[type[BaseStructure]]:
        """Output type."""
        ...

    @property
    def tools(self) -> Optional[list]:
        """Tools."""
        ...

    @property
    def model_settings(self) -> Optional[ModelSettings]:
        """Model settings."""
        ...

    @property
    def handoffs(self) -> Optional[list[Agent | Handoff]]:
        """Handoffs."""
        ...

    @property
    def input_guardrails(self) -> Optional[list[InputGuardrail]]:
        """Input guardrails."""
        ...

    @property
    def output_guardrails(self) -> Optional[list[OutputGuardrail]]:
        """Output guardrails."""
        ...

    @property
    def session(self) -> Optional[Session]:
        """Session."""
        ...


class BaseAgent(DataclassJSONSerializable):
    """Factory for creating and configuring specialized agents.

    ``BaseAgent`` provides the foundation for building OpenAI agents with support
    for Jinja2 prompt templates, custom tools, handoffs for agent delegation,
    input and output guardrails for validation, session management for
    conversation history, and both synchronous and asynchronous execution modes.
    All specialized agents in this package extend this base class.

    Examples
    --------
    Create a basic agent from configuration:

    >>> from openai_sdk_helpers.agent import BaseAgent, AgentConfiguration
    >>> config = AgentConfiguration(
    ...     name="my_agent",
    ...     description="A custom agent",
    ...     model="gpt-4o-mini"
    ... )
    >>> agent = BaseAgent(config=config, default_model="gpt-4o-mini")
    >>> result = agent.run_sync("What is 2+2?")

    Use absolute path to template:

    >>> config = AgentConfiguration(
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
    build_prompt_from_jinja(run_context_wrapper)
        Render the agent prompt using Jinja and optional context.
    get_prompt(run_context_wrapper, _)
        Render the agent prompt using the provided run context.
    name
        Return the name of this agent.
    instructions_text
        Return the resolved instructions for this agent.
    tools
        Return the tools configured for this agent.
    output_type
        Return the output type configured for this agent.
    model_settings
        Return the model settings configured for this agent.
    handoffs
        Return the handoff configurations for this agent.
    input_guardrails
        Return the input guardrails configured for this agent.
    output_guardrails
        Return the output guardrails configured for this agent.
    session
        Return the session configured for this agent.
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
        *,
        config: AgentConfigurationLike,
        run_context_wrapper: Optional[RunContextWrapper[Dict[str, Any]]] = None,
        data_path: Path | str | None = None,
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

        prompt_path = config.resolve_prompt_path(prompt_dir)

        # Build template from file or fall back to instructions
        if prompt_path is None:
            instructions_text = config.instructions_text
            self._template = Template(instructions_text)
            self._instructions = instructions_text
        elif prompt_path.exists():
            self._template = Template(prompt_path.read_text(encoding="utf-8"))
            self._instructions = None
        else:
            raise FileNotFoundError(
                f"Prompt template for agent '{name}' not found at {prompt_path}."
            )

        self._name = name
        self.uuid = uuid.uuid4()
        self.description = description
        self.model = model

        # Resolve data_path with class name appended
        class_name = self.__class__.__name__
        if data_path is not None:
            data_path_obj = Path(data_path)
            if data_path_obj.name == class_name:
                self._data_path = data_path_obj
            else:
                self._data_path = data_path_obj / class_name
        else:
            from ..environment import get_data_path

            self._data_path = get_data_path(self.__class__.__name__)

        self._input_type = config.input_type
        self._output_type = config.output_type or config.input_type
        self._tools = config.tools
        self._model_settings = config.model_settings
        self._handoffs = config.handoffs
        self._input_guardrails = config.input_guardrails
        self._output_guardrails = config.output_guardrails
        self._session = config.session
        self._run_context_wrapper = run_context_wrapper

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

    @property
    def name(self) -> str:
        """Return the name of this agent.

        Returns
        -------
        str
            Name used to identify the agent.
        """
        return self._name

    @property
    def instructions_text(self) -> str:
        """Return the resolved instructions for this agent.

        Returns
        -------
        str
            Rendered instructions text using the current run context.
        """
        if self._instructions is not None:
            return self._instructions
        return self._build_prompt_from_jinja()

    @property
    def tools(self) -> Optional[list]:
        """Return the tools configured for this agent.

        Returns
        -------
        list or None
            Tool definitions configured for the agent.
        """
        return self._tools

    @property
    def output_type(self) -> Optional[type[BaseStructure]]:
        """Return the output type configured for this agent.

        Returns
        -------
        type[BaseStructure] or None
            Output type used to cast responses.
        """
        return self._output_type

    @property
    def model_settings(self) -> Optional[ModelSettings]:
        """Return the model settings configured for this agent.

        Returns
        -------
        ModelSettings or None
            Model settings applied to the agent.
        """
        return self._model_settings

    @property
    def handoffs(self) -> Optional[list[Agent | Handoff]]:
        """Return the handoff configurations for this agent.

        Returns
        -------
        list[Agent or Handoff] or None
            Handoff configurations available to the agent.
        """
        return self._handoffs

    @property
    def input_guardrails(self) -> Optional[list[InputGuardrail]]:
        """Return the input guardrails configured for this agent.

        Returns
        -------
        list[InputGuardrail] or None
            Input guardrails applied to the agent.
        """
        return self._input_guardrails

    @property
    def output_guardrails(self) -> Optional[list[OutputGuardrail]]:
        """Return the output guardrails configured for this agent.

        Returns
        -------
        list[OutputGuardrail] or None
            Output guardrails applied to the agent.
        """
        return self._output_guardrails

    @property
    def session(self) -> Optional[Session]:
        """Return the session configured for this agent.

        Returns
        -------
        Session or None
            Session configuration used for maintaining conversation history.
        """
        return self._session

    def get_agent(self) -> Agent:
        """Construct and return the configured :class:`agents.Agent` instance.

        Returns
        -------
        Agent
            Initialized agent ready for execution.
        """
        agent_config: Dict[str, Any] = {
            "name": self._name,
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
        output_type: Optional[type[BaseStructure]] = None,
        session: Optional[Any] = None,
    ) -> Any:
        """Execute the agent asynchronously.

        Parameters
        ----------
        input : str
            Prompt or query for the agent.
        context : dict or None, default=None
            Optional dictionary passed to the agent.
        output_type : type[BaseStructure] or None, default=None
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
        output_type: Optional[type[BaseStructure]] = None,
        session: Optional[Any] = None,
    ) -> Any:
        """Run the agent synchronously.

        Parameters
        ----------
        input : str
            Prompt or query for the agent.
        context : dict or None, default=None
            Optional dictionary passed to the agent.
        output_type : type[BaseStructure] or None, default=None
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
        output_type: Optional[type[BaseStructure]] = None,
        session: Optional[Any] = None,
    ) -> RunResultStreaming | BaseStructure:
        """Stream the agent execution results.

        Parameters
        ----------
        input : str
            Prompt or query for the agent.
        context : dict or None, default=None
            Optional dictionary passed to the agent.
        output_type : type[BaseStructure] or None, default=None
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
        output_type_to_use = output_type or self._output_type
        result = run_streamed(
            agent=self.get_agent(),
            input=input,
            context=context,
            output_type=output_type_to_use,
            session=session_to_use,
        )
        if output_type_to_use and hasattr(result, "final_output_as"):
            return cast(Any, result).final_output_as(output_type_to_use)
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
            tool_name=self._name, tool_description=self.description
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
        log(f"Closing session {self.uuid} for {self.__class__.__name__}")
        self.save()

    def __repr__(self) -> str:
        """Return a string representation of the BaseAgent.

        Returns
        -------
        str
            String representation including agent name and model.
        """
        return f"<BaseAgent name={self._name!r} model={self.model!r}>"

    def save(self, filepath: str | Path | None = None) -> None:
        """Serialize the message history to a JSON file.

        Saves the current message history to a specified file path in JSON format.
        If no file path is provided, it saves to a default location based on
        the agent's UUID.

        Parameters
        ----------
        filepath : str | Path | None, default=None
            Optional file path to save the serialized history. If None,
            uses a default filename based on the agent name.
        """
        if filepath is not None:
            target = Path(filepath)
        else:
            filename = f"{str(self.uuid).lower()}.json"
            target = self._data_path / self._name / filename

        checked = check_filepath(filepath=target)
        self.to_json_file(filepath=checked)
        log(f"Saved messages to {target}")


__all__ = ["AgentConfigurationLike", "BaseAgent"]
