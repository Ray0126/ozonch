from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIST = os.path.join(_COMPONENT_DIR, "frontend", "dist")

_ozon_table_component = components.declare_component(
    "ozon_table_component",
    path=_FRONTEND_DIST,
)


def ozon_table(
    data: Optional[List[Dict[str, Any]]] = None,
    columns: Optional[List[Dict[str, Any]]] = None,
    *,
    # Backward/compat kwargs (so app.py can evolve without breaking)
    df: Any = None,
    rows: Any = None,
    col_defs: Any = None,
    default_state: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    table_id: str = "table",
    height: int = 520,
    key: Optional[str] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Render interactive table.

    Args:
        data: list of row dicts
        columns: optional column definitions; if omitted frontend infers from data keys
        table_id: used for persistent layout in browser localStorage
        height: px
        key: Streamlit component key

    Returns:
        dict with current state (e.g., column order/hidden, filters) or empty dict.
    """

    # --- Normalize inputs ---
    # Allow passing a pandas DataFrame via df= or rows=
    if data is None:
        data = []
    if df is not None and not data:
        try:
            import pandas as pd  # type: ignore

            if isinstance(df, pd.DataFrame):
                data = df.to_dict(orient="records")
        except Exception:
            # If pandas isn't available or df isn't a DataFrame, ignore
            pass
    if rows is not None and not data:
        # rows can already be list[dict]
        if isinstance(rows, list):
            data = rows  # type: ignore

    # Column defs aliases
    if columns is None and col_defs is not None:
        if isinstance(col_defs, list):
            columns = col_defs  # type: ignore
    if columns is None:
        columns = []

    # Prefer explicit `state`, else `default_state`
    init_state = state if isinstance(state, dict) else (default_state if isinstance(default_state, dict) else {})

    payload = {
        "data": data or [],
        "columns": columns or [],
        "table_id": table_id,
        "height": int(height),
        "state": init_state,
    }

    # default={} is what Streamlit returns before the component posts back.
    return _ozon_table_component(**payload, key=key, default={})
