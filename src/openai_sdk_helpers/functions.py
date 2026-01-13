"""Helper functions used by the askPAT conversational agent.

This module contains tool-call utilities and orchestration helpers used by the
askPAT conversational agent. Functions follow NumPy-style docstrings and are
kept lightweight and composable.
"""

import logging
import threading
from datetime import datetime
import asyncio
from typing import Awaitable, List, Optional, Callable, Dict, TypeVar

from .config import OpenAISettings

from . import (
    StructureBase,
    AgentEnum,
    PlanStructure,
    PromptStructure,
    TaskStructure,
    VectorSearch,
    WebAgentSearch,
    WebSearchStructure,
    VectorSearchStructure,
    tool_handler_factory,
    get_model,
)
from .logging_config import log


def _resolve_task_type(task_type: AgentEnum | str) -> str:
    """Normalize task type identifiers for registry lookup."""
    if isinstance(task_type, AgentEnum):
        return task_type.value
    if isinstance(task_type, str) and task_type in AgentEnum.__members__:
        return AgentEnum.__members__[task_type].value
    try:
        return AgentEnum(task_type).value
    except ValueError:
        return str(task_type)


async def generate_prompt(prompt: str) -> PromptStructure:
    """Generate a prompt structure from a given prompt string.

    Parameters
    ----------
    prompt
        The prompt text to be structured.

    Returns
    -------
    PromptStructure
        An instance of ``PromptStructure`` containing the prompt.

    Examples
    --------
    >>> await generate_prompt("Summarize this request")  # doctest: +SKIP
    PromptStructure(...)
    """
    log("generate_prompt")
    from .response import prompter

    open_ai_settings = OpenAISettings.from_env()
    with prompter.PROMPTER.gen_response(openai_settings=open_ai_settings) as prompter:
        # Ensure we await the asynchronous response generation.
        prompter_response = await prompter.run_async(content=prompt)
        if isinstance(prompter_response, PromptStructure):
            return PromptStructure.from_raw_input(prompter_response.to_json())

    return PromptStructure(prompt=prompt)


def create_plan(prompt: PromptStructure | str) -> PlanStructure:
    """Generate a plan of agent tasks for the provided prompt.

    Parameters
    ----------
    prompt
        User request or brief to convert into a plan.

    Returns
    -------
    PlanStructure
        Structured list of tasks. Returns an empty ``PlanStructure`` when
        planning fails.
    """
    log("create_plan")
    from .response import planner

    open_ai_settings = OpenAISettings.from_env()

    try:
        prompt_text = prompt.prompt if isinstance(prompt, PromptStructure) else prompt
        with planner.PLANNER.gen_response(openai_settings=open_ai_settings) as _planner:
            plan = _planner.run_sync(content=prompt_text)
    except Exception as exc:  # noqa: BLE001
        log(f"Plan generation failed: {exc}", level=logging.ERROR)
        return PlanStructure()

    return plan if isinstance(plan, PlanStructure) else PlanStructure()


def execute_task(
    task: TaskStructure, previous_results: Optional[List[str]] = None
) -> TaskStructure:
    """Execute a single planned task using the default agent registry.

    Parameters
    ----------
    task : TaskStructure
        Structured task definition containing the agent identifier and inputs.
    previous_results : list[str], optional
        Outputs from earlier tasks, appended to the task context for chaining.

    Returns
    -------
    TaskStructure
        Task with updated status, timestamps, results, and errors reflected in
        ``task.results`` when exceptions occur.
    """
    log(f"Executing task type='{task.task_type}'")

    if previous_results:
        task.context = list(task.context or []) + list(previous_results)

    task_key = _resolve_task_type(task.task_type)
    if task_key not in REGISTRY:
        now = datetime.utcnow()
        task.start_date = now
        task.end_date = now
        task.status = "error"
        task.results = [f"Unknown task type: {task_key}"]
        return task

    plan = PlanStructure(tasks=[task])
    plan.execute(agent_registry=REGISTRY, halt_on_error=True)
    return task


REGISTRY: Dict[str, Callable[..., object]] = {
    "WebAgentSearch": lambda prompt: WebAgentSearch(
        default_model=get_model()
    ).run_agent_sync(prompt),
    "VectorSearch": lambda prompt: VectorSearch(
        default_model=get_model()
    ).run_agent_sync(prompt),
}


def execute_plan(plan: PlanStructure) -> List[str]:
    """Execute each task in a plan sequentially, chaining results.

    Parameters
    ----------
    plan : PlanStructure
        Structured plan containing ordered tasks to run.

    Returns
    -------
    list[str]
        Flattened list of results from all executed tasks.
    """
    return plan.execute(agent_registry=REGISTRY, halt_on_error=True)


TStructure = TypeVar("TStructure", bound=StructureBase)
TResult = TypeVar("TResult")


def _normalize_structure_kwargs(
    structure_cls: type[TStructure], kwargs: dict[str, object]
) -> dict[str, object]:
    """Normalize tool arguments before constructing a structure instance."""
    structure_key_matches = {
        "structure",
        "targeting_structure",
        structure_cls.__name__,
        structure_cls.__name__.lower(),
    }
    if len(kwargs) == 1:
        key, value = next(iter(kwargs.items()))
        if key in structure_key_matches and isinstance(value, dict):
            return value
    for key in structure_key_matches:
        sub_value = kwargs.get(key)
        if isinstance(sub_value, dict):
            return sub_value
    return kwargs


def _build_structure_tool(
    structure_cls: type[TStructure],
    handler: Callable[[TStructure], TResult],
) -> Callable[..., TResult]:
    """Return a tool payload handler that builds a structure then calls handler."""

    def _runner(**kwargs: object) -> TResult:
        clean_kwargs = _normalize_structure_kwargs(structure_cls, kwargs)
        structure = structure_cls.from_raw_input(clean_kwargs)
        return handler(structure)

    return _runner


def _run_awaitable(awaitable: Awaitable[TResult]) -> TResult:
    """Execute an awaitable in a synchronous context.

    When no event loop is running, ``asyncio.run`` is used. If there already is
    a running loop (e.g., inside Jupyter or another async framework), the
    awaitable executes on a fresh loop hosted in a background ``thread`` to
    avoid ``RuntimeError`` while still returning the result synchronously.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:

        async def _runner() -> TResult:
            return await awaitable

        return asyncio.run(_runner())

    result: TResult | None = None
    exception: Exception | None = None
    completed = False

    def _thread_runner() -> None:
        nonlocal result, exception, completed
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(awaitable)
            completed = True
        except Exception as exc:  # noqa: BLE001
            exception = exc
        finally:
            loop.close()

    thread = threading.Thread(target=_thread_runner)
    thread.daemon = True
    thread.start()
    thread.join()

    if exception:
        raise exception
    if not completed:
        raise RuntimeError("Async task did not complete")

    return result  # type: ignore[return-value]


def _build_async_structure_tool(
    structure_cls: type[TStructure],
    handler: Callable[[TStructure], Awaitable[TResult]],
) -> Callable[..., TResult]:
    """Return a sync handler that builds a structure then runs the async processor."""

    def _runner(**kwargs: object) -> TResult:
        clean_kwargs = _normalize_structure_kwargs(structure_cls, kwargs)
        structure = structure_cls.from_raw_input(
            clean_kwargs
        )  # convert enums and validate
        return _run_awaitable(handler(structure))

    return _runner


def _build_async_prompt_tool(
    handler: Callable[[str], Awaitable[TResult]],
) -> Callable[..., TResult]:
    """Return a sync handler that executes an async prompt processor."""

    def _runner(**kwargs: object) -> TResult:
        clean_kwargs = _normalize_structure_kwargs(PromptStructure, kwargs)
        payload = PromptStructure.from_raw_input(clean_kwargs)
        if not payload.prompt:
            raise ValueError("prompt argument is required")
        return _run_awaitable(handler(payload.prompt))

    return _runner


def _execute_task_payload(**kwargs: object) -> TaskStructure:
    """Normalize arguments and execute a task."""
    task = TaskStructure.model_validate(kwargs)
    return execute_task(task)


def _execute_plan_payload(**kwargs: object) -> PlanStructure:
    """Normalize arguments and execute a plan."""
    plan = PlanStructure.model_validate(kwargs)
    execute_plan(plan)
    return plan


async def _run_web_agent(prompt: str) -> WebSearchStructure:
    """Run the web agent for a prompt."""
    return await WebAgentSearch(default_model=get_model()).run_agent_async(
        search_query=prompt
    )


async def _run_vector_agent(prompt: str) -> VectorSearchStructure:
    """Run the vector agent for a prompt."""
    return await VectorSearch(default_model=get_model()).run_agent(search_query=prompt)


web_agent_handler = tool_handler_factory(
    _build_async_prompt_tool(_run_web_agent),
    input_model=PromptStructure,
)
vector_agent_handler = tool_handler_factory(
    _build_async_prompt_tool(_run_vector_agent),
    input_model=PromptStructure,
)


# Define the tool handlers mapping
TOOL_HANDLERS = {
    "web_search": web_agent_handler,
    "internal_search": vector_agent_handler,
    "create_plan": tool_handler_factory(
        _build_structure_tool(PromptStructure, create_plan)
    ),
    "execute_task": tool_handler_factory(_execute_task_payload),
    "execute_plan": tool_handler_factory(_execute_plan_payload),
}

__all__ = [
    "create_plan",
    "execute_task",
    "execute_plan",
    "TOOL_HANDLERS",
]
