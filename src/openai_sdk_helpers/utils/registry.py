"""Base registry class for managing configuration instances."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Generic, TypeVar
from typing import Protocol, Self
from .path_utils import ensure_directory


class RegistryProtocol(Protocol):
    """Protocol describing serializable registry entries.

    Methods
    -------
    to_json_file(filepath)
        Write the instance to a JSON file path.
    from_json_file(filepath)
        Load an instance from a JSON file path.
    """

    @property
    def name(self) -> str:
        """Return the configuration name."""
        ...

    def to_json_file(self, filepath: Path | str) -> str:
        """Write serialized JSON data to a file path."""
        ...

    @classmethod
    def from_json_file(cls, filepath: Path | str) -> Self:
        """Load an instance from a JSON file."""
        ...


T = TypeVar("T", bound=RegistryProtocol)


class RegistryBase(Generic[T]):
    """Base registry for managing configuration instances.

    Provides centralized storage and retrieval of configurations,
    enabling reusable specs across the application. Configurations
    are stored by name and can be retrieved or listed as needed.

    Type Parameters
    ---------------
    T
        The configuration type this registry manages.

    Methods
    -------
    register(configuration)
        Add a configuration to the registry.
    get(name)
        Retrieve a configuration by name.
    list_names()
        Return all registered configuration names.
    clear()
        Remove all registered configurations.
    save_to_directory(path)
        Export all registered configurations to JSON files.
    load_from_directory(path)
        Load configurations from JSON files in a directory.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._configs: dict[str, T] = {}

    @property
    def configs(self) -> dict[str, T]:
        """Return the internal configuration mapping.

        Returns
        -------
        dict[str, T]
            Mapping of configuration names to instances.
        """
        return self._configs

    def register(self, configuration: T) -> None:
        """Add a configuration to the registry.

        Parameters
        ----------
        configuration : T
            Configuration to register. Must have a 'name' attribute.

        Raises
        ------
        ValueError
            If a configuration with the same name is already registered.
        """
        name = getattr(configuration, "name")
        if name in self._configs:
            raise ValueError(
                f"Configuration '{name}' is already registered. "
                "Use a unique name or clear the registry first."
            )
        self._configs[name] = configuration

    def get(self, name: str) -> T:
        """Retrieve a configuration by name.

        Parameters
        ----------
        name : str
            Configuration name to look up.

        Returns
        -------
        T
            The registered configuration.

        Raises
        ------
        KeyError
            If no configuration with the given name exists.
        """
        if name not in self._configs:
            raise KeyError(
                f"No configuration named '{name}' found. "
                f"Available: {list(self._configs.keys())}"
            )
        return self._configs[name]

    def list_names(self) -> list[str]:
        """Return all registered configuration names.

        Returns
        -------
        list[str]
            Sorted list of configuration names.
        """
        return sorted(self._configs.keys())

    def clear(self) -> None:
        """Remove all registered configurations."""
        self._configs.clear()

    def save_to_directory(self, path: Path | str) -> None:
        """Export all registered configurations to JSON files in a directory.

        Serializes each registered configuration to an individual JSON file
        named after the configuration. Creates the directory if it does not exist.

        Parameters
        ----------
        path : Path or str
            Directory path where JSON files will be saved. Will be created if
            it does not already exist.

        Raises
        ------
        OSError
            If the directory cannot be created or files cannot be written.
        """
        dir_path = ensure_directory(Path(path))
        config_names = self.configs

        if not config_names:
            return

        for config_name in config_names:
            configuration = self.get(config_name)
            filename = f"{config_name}.json"
            filepath = dir_path / configuration.__class__.__name__ / filename
            configuration.to_json_file(filepath)

    def load_from_directory(self, path: Path | str, *, config_class: type[T]) -> int:
        """Load all configurations from JSON files in a directory.

        Scans the directory for JSON files and attempts to load each as a
        configuration. Successfully loaded configurations are registered.
        If a file fails to load, a warning is issued and processing continues
        with the remaining files.

        Parameters
        ----------
        path : Path or str
            Directory path containing JSON configuration files.
        config_class : type[T]
            The configuration class to use for deserialization.

        Returns
        -------
        int
            Number of configurations successfully loaded and registered.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        NotADirectoryError
            If the path is not a directory.
        """
        dir_path = Path(path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {dir_path}")

        count = 0
        for json_file in sorted(dir_path.glob("*.json")):
            try:
                configuration = config_class.from_json_file(json_file)
                self.register(configuration)
                count += 1
            except Exception as exc:
                # Log warning but continue processing other files
                warnings.warn(
                    f"Failed to load configuration from {json_file}: {exc}",
                    stacklevel=2,
                )

        return count
