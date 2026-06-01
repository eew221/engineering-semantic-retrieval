"""Class metadata and prompt helpers."""

from __future__ import annotations

DACL10K_DAMAGE_CLASSES = [
    "crack",
    "alligator crack",
    "wetspot",
    "efflorescence",
    "rust",
    "rockpocket",
    "hollowareas",
    "cavity",
    "spalling",
    "graffiti",
    "weathering",
    "restformwork",
    "exposed rebars",
]

# Public mirrors show slight count inconsistencies for the object-part group.
# The official toolkit lists 13 damage classes and 6 object classes.
# Keep the list configurable and tolerant to an unknown fallback component.
DACL10K_COMPONENT_CLASSES = [
    "bearing",
    "expansion joint",
    "drainage",
    "protective equipment",
    "joint tape",
    "washouts/concrete corrosion",
    "bridge component unknown",
]

CODEBRIM_CLASSES = [
    "crack",
    "spallation",
    "exposed reinforcement bar",
    "efflorescence",
    "corrosion stain",
    "background",
]


def make_damage_prompt(damage_name: str) -> str:
    return f"a bridge inspection image showing {damage_name}"


def make_component_prompt(component_name: str) -> str:
    return f"a bridge inspection image containing {component_name}"


def make_compositional_prompt(damage_name: str, component_name: str) -> str:
    if not component_name or component_name == "bridge component unknown":
        return make_damage_prompt(damage_name)
    return f"a bridge inspection image showing {damage_name} on {component_name}"
