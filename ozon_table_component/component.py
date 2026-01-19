from __future__ import annotations
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

_FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

_component = components.declare_component(
    "ozon_table_component",
    path=str(_FRONTEND_DIST),
)

def ozon_table(data, columns, state=None, key=None, height=520):
    # data может быть DataFrame -> превращаем в list[dict]
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="records")

    payload = {
        "data": data or [],
        "columns": columns or [],
        "state": state or {},
        "height": int(height),
    }
    return _component(**payload, key=key, default=state or {})
