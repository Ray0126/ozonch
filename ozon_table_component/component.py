from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from typing import Any, Dict, List, Optional

_BUILD_DIR = (Path(__file__).parent / "frontend" / "dist").resolve()

# ВАЖНО: имя компонента (первый аргумент) должно совпадать всегда
_ozon_table_component = components.declare_component(
    "ozon_table_component",
    path=str(_BUILD_DIR),
)

def ozon_table(
    *,
    data: List[Dict[str, Any]],
    columns: List[str],
    state: Optional[Dict[str, Any]] = None,
    height: int = 520,
    key: str = "ozon_table_component",
) -> Dict[str, Any]:
    # Если dist не на месте — не падаем всем приложением
    if not (_BUILD_DIR / "index.html").exists():
        st.warning("Компонент таблицы не собран: нет frontend/dist/index.html. Показываю обычную таблицу.")
        st.dataframe(data, use_container_width=True, height=height)
        return state or {}

    payload = {
        "data": data if data is not None else [],
        "columns": columns if columns is not None else [],
        "state": state or {},
    }

    result = _ozon_table_component(
        **payload,
        default=state or {},
        key=key,
        height=height,
    )

    # Streamlit иногда может вернуть None при первом рендере
    if result is None:
        return state or {}
    return result
