"""
Built-in strategy templates for common trading patterns.

Usage::

    from vibetrading.templates import momentum, mean_reversion, grid, dca

    # Generate with defaults
    code = momentum.generate()

    # Customize parameters
    code = momentum.generate(asset="ETH", leverage=5, sma_fast=7, sma_slow=21)

    # List available templates
    from vibetrading.templates import list_templates, get_template
    print(list_templates())
    template = get_template("momentum")
    code = template.generate(asset="SOL")
"""

from . import dca, grid, mean_reversion, momentum

_TEMPLATES = {
    "momentum": momentum,
    "mean_reversion": mean_reversion,
    "grid": grid,
    "dca": dca,
}


def list_templates() -> list[str]:
    """Return names of all available strategy templates."""
    return list(_TEMPLATES.keys())


def get_template(name: str):
    """Get a template module by name.

    Args:
        name: Template name (e.g. 'momentum', 'mean_reversion', 'grid', 'dca').

    Returns:
        The template module with a ``generate(**kwargs) -> str`` function.

    Raises:
        KeyError: If the template name is not found.
    """
    if name not in _TEMPLATES:
        available = ", ".join(_TEMPLATES.keys())
        raise KeyError(f"Template '{name}' not found. Available: {available}")
    return _TEMPLATES[name]


__all__ = [
    "dca",
    "get_template",
    "grid",
    "list_templates",
    "mean_reversion",
    "momentum",
]
