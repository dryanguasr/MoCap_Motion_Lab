"""Canonical flat layout for generated artifacts.

Every filename carries its source, pipeline, and artifact role so files remain
identifiable after outputs are grouped by type.
"""

from __future__ import annotations

import re
from pathlib import Path


CATEGORIES = (
    "videos",
    "metrics",
    "events",
    "datasets",
    "summaries",
    "arrays",
    "diagnostics",
    "reports",
    "metadata",
    "logs",
)


def _token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._~-]+", "_", str(value).strip()).strip("._")
    if not token or token in {".", ".."}:
        raise ValueError(f"Invalid output filename token: {value!r}")
    return token


def ensure_output_layout(root: Path) -> Path:
    """Create and return the canonical processed-output root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        (root / category).mkdir(exist_ok=True)
    return root


def artifact_path(
    root: Path,
    category: str,
    source: str,
    pipeline: str,
    artifact: str,
    suffix: str,
) -> Path:
    """Return ``root/category/source__pipeline__artifact.suffix``."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown output category {category!r}; expected one of {CATEGORIES}")
    extension = suffix if suffix.startswith(".") else f".{suffix}"
    if not re.fullmatch(r"\.[A-Za-z0-9]+", extension):
        raise ValueError(f"Invalid output suffix: {suffix!r}")
    directory = ensure_output_layout(root) / category
    name = "__".join((_token(source), _token(pipeline), _token(artifact))) + extension.lower()
    return directory / name
