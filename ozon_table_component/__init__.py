import os
from typing import Any, Dict, Optional

import pandas as pd
import streamlit.components.v1 as components


# NOTE: for Streamlit Cloud the built frontend must be committed.
_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")

_component_func = components.declare_component(
    "ozon_tanstack_table",
    path=_FRONTEND_DIST,
)


def tanstack_table(
    df: pd.DataFrame,
    *,
    key: str,
    state: Optional[Dict[str, Any]] = None,
    height: int = 560,
    compact: bool = True,
    show_toolbar: bool = True,
) -> Dict[str, Any]:
    """Render a lightweight TanStack Table with drag&drop column reordering.

    Returns updated view state that you can store in st.session_state (or Supabase).
    """

    if df is None:
        df = pd.DataFrame()

    df2 = df.copy()
    # Keep NaN as None for JS
    data = df2.where(pd.notnull(df2), None).to_dict(orient="records")

    cols = []
    for c in df2.columns.tolist():
        s = df2[c]
        dtype = "number" if pd.api.types.is_numeric_dtype(s) else "text"
        cols.append({"id": c, "header": c, "dtype": dtype})

    payload = {
        "data": data,
        "columns": cols,
        "state": state or {},
        "height": int(height),
        "compact": bool(compact),
        "showToolbar": bool(show_toolbar),
    }

    result = _component_func(**payload, key=key, default=state or {})
    if isinstance(result, dict):
        return result
    return state or {}
