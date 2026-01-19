from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"

_component = components.declare_component(
    "ozon_table_component",
    path=str(_FRONTEND_DIR),
)


def ozon_table(
    *,
    data: Optional[List[Dict[str, Any]]] = None,
    columns: Optional[List[Dict[str, Any]]] = None,
    state: Optional[Dict[str, Any]] = None,
    height: int = 520,
    key: Optional[str] = None,
) -> Dict[str, Any]:
    """Lightweight sortable/filterable table with column drag&drop.

    Parameters
    ----------
    data: list[dict]
        Rows.
    columns: list[dict]
        Column specs: {"id": <field>, "label": <title>, "width": <px optional>}.
    state: dict
        Persisted UI state from previous renders. At minimum supports:
          - column_order: list[str]
          - hidden_columns: list[str]
          - global_filter: str
    """

    payload = {
        "data": data or [],
        "columns": columns or [],
        "state": state or {},
        "height": int(height),
    }

    # Return value is the updated state dict from the frontend.
    res = _component(payload, default=payload["state"], key=key)

    # Frontend may return None during first paint.
    return res or payload["state"]
