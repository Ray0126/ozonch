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
    data: List[Dict[str, Any]],
    columns: Optional[List[Dict[str, Any]]] = None,
    *,
    table_id: str = "table",
    height: int = 520,
    key: Optional[str] = None,
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

    payload = {
        "data": data or [],
        "columns": columns or [],
        "table_id": table_id,
        "height": int(height),
    }

    return _ozon_table_component(**payload, key=key, default={})
