"""Helper functions for creating and executing agent plans.

This module provides convenience functions for working with PlanStructure
and TaskStructure, simplifying common workflows like plan creation, task
execution, and result aggregation.
"""

from __future__ import annotations

from .enum import AgentEnum
from .plan import PlanStructure
from .task import TaskStructure
from .types import AgentCallable, AgentRegistry


def create_plan(*tasks: TaskStructure) -> PlanStructure:
    """Create a PlanStructure from a sequence of tasks.

    Convenience factory function that constructs a plan from individual
    tasks. Tasks are executed in the order they are provided.

    Parameters
    ----------
    *tasks : TaskStructure
        Variable number of task definitions to include in the plan.

    Returns
    -------
    PlanStructure
        New plan containing the provided tasks in order.

    Examples
    --------
    >>> task1 = TaskStructure(
    ...     task_type=AgentEnum.WEB_SEARCH,
    ...     prompt="Search for AI trends"
    ... )
    >>> task2 = TaskStructure(
    ...     task_type=AgentEnum.SUMMARIZER,
    ...     prompt="Summarize findings"
    ... )
    >>> plan = create_plan(task1, task2)
    >>> len(plan)
    2
    """
    return PlanStructure(tasks=list(tasks))


def execute_task(
    task: TaskStructure,
    agent_callable: AgentCallable,
) -> list[str]:
    """Execute a single task with an agent callable.

    Runs one task using the provided agent function. Updates task status,
    timing, and results. Context from previous tasks is not supported in this
    helper - use execute_plan() for multi-task execution with context passing.

    Parameters
    ----------
    task : TaskStructure
        Task definition containing prompt and metadata.
    agent_callable : AgentCallable
        Synchronous or asynchronous callable responsible for executing the task.
        Should accept the task prompt and an optional context keyword argument.

    Returns
    -------
    list[str]
        Normalized string results from task execution.

    Raises
    ------
    Exception
        Any exception raised by the agent_callable is propagated after
        task status is updated.

    Examples
    --------
    >>> def agent_fn(prompt, context=None):
    ...     return f"Result for {prompt}"
    >>> task = TaskStructure(prompt="Test task")
    >>> results = execute_task(task, agent_fn)
    >>> task.status
    'done'
    """
    from datetime import datetime, timezone

    task.start_date = datetime.now(timezone.utc)
    task.status = "running"

    # Build plan with single task and execute
    # Normalize task_type to string value for registry key to match PlanStructure.execute lookup
    plan = PlanStructure(tasks=[task])
    # Convert AgentEnum to its string value for registry key
    registry_key = (
        task.task_type.value
        if isinstance(task.task_type, AgentEnum)
        else task.task_type
    )
    registry: dict[str, AgentCallable] = {
        registry_key: agent_callable,
    }

    # Execute the plan - it will update task status
    aggregated = plan.execute(
        agent_registry=registry,
        halt_on_error=True,
    )

    # If task failed, raise the exception
    if task.status == "error":
        # Extract error message from results
        error_msg = task.results[0] if task.results else "Task execution failed"
        # Raise RuntimeError with the error message
        # The original exception type information is lost but the message is preserved
        raise RuntimeError(f"Task execution error: {error_msg}")

    return aggregated


def execute_plan(
    plan: PlanStructure,
    agent_registry: AgentRegistry,
    *,
    halt_on_error: bool = True,
) -> list[str]:
    """Execute a plan using registered agent callables.

    Convenience wrapper around PlanStructure.execute() for cleaner syntax.
    Runs all tasks in sequence, passing results between tasks as context.

    Parameters
    ----------
    plan : PlanStructure
        Plan containing ordered tasks to execute.
    agent_registry : AgentRegistry
        Lookup of agent identifiers to callables. Keys may be AgentEnum
        instances or their string values.
    halt_on_error : bool, default True
        Whether execution should stop when a task raises an exception.

    Returns
    -------
    list[str]
        Flattened list of normalized outputs from all executed tasks.

    Raises
    ------
    KeyError
        If a task references an agent not in the registry.

    Examples
    --------
    >>> def search_agent(prompt, context=None):
    ...     return ["search results"]
    >>> def summary_agent(prompt, context=None):
    ...     return ["summary"]
    >>> registry = {
    ...     AgentEnum.WEB_SEARCH: search_agent,
    ...     AgentEnum.SUMMARIZER: summary_agent,
    ... }
    >>> plan = PlanStructure(tasks=[...])  # doctest: +SKIP
    >>> results = execute_plan(plan, registry)  # doctest: +SKIP
    """
    return plan.execute(agent_registry, halt_on_error=halt_on_error)


__all__ = [
    "create_plan",
    "execute_task",
    "execute_plan",
]
