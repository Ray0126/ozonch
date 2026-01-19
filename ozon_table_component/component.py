"""Ozon TanStack table Streamlit component (Python wrapper).

This version is cloud-safe:
- No DataFrame truthiness checks (fixes ValueError: ambiguous truth value)
- Accepts pandas.DataFrame or list[dict]
- Signature tolerant to extra kwargs

Repo layout expected:
  ozon_table_component/
    __init__.py
    component.py
    frontend/dist/index.html
    frontend/dist/main.js
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit.components.v1 as components

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# IMPORTANT: the frontend assets MUST exist at this path at runtime
_DIST_PATH = Path(__file__).resolve().parent / "frontend" / "dist"

# Use an internal name for the component function to avoid confusing error messages
_component_func = components.declare_component(
    "ozon_table_component",
    path=str(_DIST_PATH),
)


def _to_records(data: Any) -> List[Dict[str, Any]]:
    """Convert supported inputs to list-of-dicts (JSON serializable)."""
    if data is None:
        return []

    # pandas DataFrame
    if pd is not None and hasattr(data, "to_dict") and data.__class__.__name__ == "DataFrame":
        try:
            df = data
            # replace NaN with None for JSON
            df = df.where(df.notna(), None)
            return df.to_dict(orient="records")
        except Exception:
            return []

    # already list of dicts
    if isinstance(data, list):
        out: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                out.append(item)
        return out

    # single dict
    if isinstance(data, dict):
        return [data]

    return []


def ozon_table(
    data: Any = None,
    *,
    key: str,
    columns: Optional[List[Dict[str, Any]]] = None,
    default_view: Optional[Dict[str, Any]] = None,
    height: int = 520,
    debug: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Render the TanStack table component.

    Parameters
    - data: pandas.DataFrame or list[dict]
    - key: Streamlit component key
    - columns: optional column definitions for the frontend
    - default_view: persisted view state (order, hidden, filters, etc.)
    - height: component height in px

    Returns
    - dict view state from frontend (or default_view if nothing returned)
    """

    records = _to_records(data)

    payload: Dict[str, Any] = {
        "data": records,
        "columns": columns or [],
        "defaultView": default_view or {},
        "debug": debug,
    }

    # Forward-compatible: some frontend builds expect top-level props instead of payload.
    # We'll provide both.
    result = _component_func(
        payload=payload,
        data=payload["data"],
        columns=payload["columns"],
        defaultView=payload["defaultView"],
        debug=payload["debug"],
        key=key,
        height=height,
        **kwargs,
    )

    if isinstance(result, dict):
        return result
    return default_view or {}
