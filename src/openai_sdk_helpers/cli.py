"""Command-line interface for openai-sdk-helpers development.

Provides CLI commands for testing agents, validating templates,
and inspecting the response registry.

Commands
--------
agent test
    Test an agent locally with sample inputs.
template validate
    Validate Jinja2 templates for syntax errors.
registry list
    List all registered response configurations.
registry inspect
    Inspect a specific configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import openai_sdk_helpers

    __version__ = getattr(openai_sdk_helpers, "__version__", "unknown")
except ImportError:
    __version__ = "unknown"


def cmd_agent_test(args: argparse.Namespace) -> int:
    """Test an agent locally.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments containing agent_name and input.

    Returns
    -------
    int
        Exit code (0 for success).

    Raises
    ------
    NotImplementedError
        As the function is not yet implemented.

    Examples
    --------
    >>> cmd_agent_test(argparse.Namespace(agent_name="test", input="hello"))
    """
    print(f"Testing agent: {args.agent_name}")
    print(f"Input: {args.input}")
    print("\n[Not yet implemented - agent testing framework coming soon]")
    return 0


def cmd_template_validate(args: argparse.Namespace) -> int:
    """Validate Jinja2 templates.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments containing template_path.

    Returns
    -------
    int
        Exit code (0 for success, 1 for validation errors).

    Raises
    ------
    FileNotFoundError
        If the template path does not exist.

    Examples
    --------
    >>> cmd_template_validate(argparse.Namespace(template_path="."))
    """
    from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

    template_path = Path(args.template_path)

    if not template_path.exists():
        print(f"Error: Path not found: {template_path}", file=sys.stderr)
        return 1

    if template_path.is_file():
        # Validate single file
        templates = [template_path]
        base_dir = template_path.parent
    else:
        # Validate directory
        templates = list(template_path.glob("**/*.jinja"))
        base_dir = template_path

    if not templates:
        print(f"No .jinja templates found in {template_path}")
        return 0

    env = Environment(loader=FileSystemLoader(base_dir))
    errors = []

    for template_file in templates:
        relative_path = template_file.relative_to(base_dir)
        try:
            env.get_template(str(relative_path))
            print(f"✓ {relative_path}")
        except TemplateSyntaxError as e:
            errors.append((relative_path, str(e)))
            print(f"✗ {relative_path}: {e}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} template(s) with errors", file=sys.stderr)
        return 1

    print(f"\n{len(templates)} template(s) validated successfully")
    return 0


def cmd_registry_list(args: argparse.Namespace) -> int:
    """List all registered response configurations.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments.

    Returns
    -------
    int
        Exit code (0 for success).

    Raises
    ------
    ImportError
        If openai_sdk_helpers is not installed.

    Examples
    --------
    >>> cmd_registry_list(argparse.Namespace())
    """
    try:
        from openai_sdk_helpers import get_default_registry
    except ImportError:
        print("Error: openai_sdk_helpers not installed", file=sys.stderr)
        return 1

    registry = get_default_registry()
    names = registry.list_names()

    if not names:
        print("No configurations registered")
        return 0

    print("Registered configurations:")
    for name in sorted(names):
        config = registry.get(name)
        tools_count = len(config.tools) if config.tools else 0
        print(f"  - {name} ({tools_count} tools)")

    return 0


def cmd_registry_inspect(args: argparse.Namespace) -> int:
    """Inspect a specific configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Command arguments containing config_name.

    Returns
    -------
    int
        Exit code (0 for success, 1 for not found).

    Raises
    ------
    ImportError
        If openai_sdk_helpers is not installed.
    KeyError
        If the configuration is not found in the registry.

    Examples
    --------
    >>> cmd_registry_inspect(argparse.Namespace(config_name="my_config"))
    """
    try:
        from openai_sdk_helpers import get_default_registry
    except ImportError:
        print("Error: openai_sdk_helpers not installed", file=sys.stderr)
        return 1

    registry = get_default_registry()

    try:
        config = registry.get(args.config_name)
    except KeyError:
        print(f"Error: Configuration '{args.config_name}' not found", file=sys.stderr)
        print("\nAvailable configurations:")
        for name in sorted(registry.list_names()):
            print(f"  - {name}")
        return 1

    print(f"Configuration: {config.name}")
    instructions_str = str(config.instructions)
    instructions_preview = (
        instructions_str[:100] if len(instructions_str) > 100 else instructions_str
    )
    print(f"Instructions: {instructions_preview}...")
    print(f"Tools: {len(config.tools) if config.tools else 0}")

    if config.tools:
        print("\nTool names:")
        for tool in config.tools:
            tool_name = tool.get("function", {}).get("name", "unknown")
            print(f"  - {tool_name}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI interface.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code.

    Examples
    --------
    >>> main(["agent", "test", "my_agent"])
    """
    parser = argparse.ArgumentParser(
        prog="openai-helpers",
        description="OpenAI SDK Helpers CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"openai-sdk-helpers {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Agent test command
    agent_parser = subparsers.add_parser("agent", help="Agent operations")
    agent_sub = agent_parser.add_subparsers(dest="agent_command")

    test_parser = agent_sub.add_parser("test", help="Test an agent")
    test_parser.add_argument("agent_name", help="Agent name to test")
    test_parser.add_argument("--input", default="", help="Test input")

    # Template validate command
    template_parser = subparsers.add_parser("template", help="Template operations")
    template_sub = template_parser.add_subparsers(dest="template_command")

    validate_parser = template_sub.add_parser("validate", help="Validate templates")
    validate_parser.add_argument(
        "template_path",
        help="Path to template file or directory",
    )

    # Registry commands
    registry_parser = subparsers.add_parser("registry", help="Registry operations")
    registry_sub = registry_parser.add_subparsers(dest="registry_command")

    registry_sub.add_parser("list", help="List registered configurations")

    inspect_parser = registry_sub.add_parser("inspect", help="Inspect configuration")
    inspect_parser.add_argument("config_name", help="Configuration name")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # Route commands
    if args.command == "agent":
        if args.agent_command == "test":
            return cmd_agent_test(args)
    elif args.command == "template":
        if args.template_command == "validate":
            return cmd_template_validate(args)
    elif args.command == "registry":
        if args.registry_command == "list":
            return cmd_registry_list(args)
        elif args.registry_command == "inspect":
            return cmd_registry_inspect(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
