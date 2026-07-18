"""Stable public-contract limits for bounded user-supplied collections."""

from typing import Annotated

from pydantic import StringConstraints

MAX_MATERIAL_ALIASES = 32
MAX_MATERIAL_ALIAS_CHARS = 255

MaterialAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_MATERIAL_ALIAS_CHARS,
    ),
]


__all__ = [
    "MAX_MATERIAL_ALIASES",
    "MAX_MATERIAL_ALIAS_CHARS",
    "MaterialAlias",
]
