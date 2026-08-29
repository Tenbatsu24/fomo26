"""Singleton registry for models, datasets, and augs.

All lookups should go through this module to keep a single source of truth.
Modules are auto-imported on package load to trigger their registration
decorators.
"""

from __future__ import annotations

from typing import TypeVar
from importlib import import_module

T = TypeVar("T")


class InstanceRegistry:
    """Named-class registry for a single type (models, datasets, augs)."""

    def __init__(self) -> None:
        self._registry: dict[str, type] = {}

    def register(self, name: str, cls: type) -> None:
        self._registry[name] = cls

    def get(self, name: str) -> type:
        if name not in self._registry:
            raise KeyError(
                f"Unknown {name!r} in registry.\nAvailable: {list(self._registry)}"
            )
        return self._registry[name]

    def keys(self) -> list[str]:
        return list(self._registry.keys())

    def __str__(self) -> str:
        return "\n" + "\n".join(f"    {k}: {v}" for k, v in self._registry.items())


class RegistryStore:
    """Singleton that holds registries keyed by type."""

    _instance: "RegistryStore | None" = None
    _instances: dict[str, InstanceRegistry] = {}

    TYPE_AUGS = "augs"
    TYPE_MODELS = "models"
    TYPE_DATASETS = "datasets"

    STORE_TYPES = [TYPE_MODELS, TYPE_DATASETS, TYPE_AUGS]

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get(self, type_of: str) -> InstanceRegistry:
        if type_of not in self._instances:
            self._instances[type_of] = InstanceRegistry()
        return self._instances[type_of]

    def register(self, type_of: str, name: str, cls: type) -> None:
        self._get(type_of).register(name, cls)

    def get(self, type_of: str, name: str) -> type:
        return self._get(type_of).get(name)

    def reg(self, type_of: str, name: str):
        """Decorator to register a class."""

        def inner(cls: type) -> type:
            self.register(type_of, name, cls)
            return cls

        return inner

    def __str__(self) -> str:
        return f"\n{self.__class__.__name__}:\n" + "\n".join(
            f"  {k}: {v}" for k, v in self._instances.items()
        )


STORE = RegistryStore()


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def register_model(name: str):
    """Decorator to register a model builder/function under STORE.TYPE_MODELS."""
    return STORE.reg(STORE.TYPE_MODELS, name)


def register_dataset(name: str):
    """Decorator to register a dataset class under STORE.TYPE_DATASETS."""
    return STORE.reg(STORE.TYPE_DATASETS, name)


def register_aug(name: str):
    """Decorator to register an aug function under STORE.TYPE_AUGS."""
    return STORE.reg(STORE.TYPE_AUGS, name)


# ---------------------------------------------------------------------------
# Lazy auto-import to trigger registrations
# ---------------------------------------------------------------------------
# Imports are deferred until first registry access to avoid circular-import
# issues that arise when registry.py is loaded before the package tree is ready.

_pkg = "med_adapt"


def _ensure_imported(_pkg: str = _pkg) -> None:
    """Import all registry submodules to trigger decorator-based registrations."""
    for _name in STORE.STORE_TYPES:
        try:
            import_module(f".{_name}", package=_pkg)
        except ImportError:
            pass


# Patch get so the import runs before any lookup
_original_get = STORE.get


def _lazy_get(type_of: str, name: str) -> type:
    _ensure_imported()
    return _original_get(type_of, name)


STORE.get = _lazy_get  # type: ignore[assignment]
