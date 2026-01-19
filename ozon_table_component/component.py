from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # noqa

_BUILD_DIR = (Path(__file__).parent / "frontend" / "dist").resolve()

_ozon_table_component = components.declare_component(
    "ozon_table_component",
    path=str(_BUILD_DIR),
)

def _to_records(data: Any) -> List[Dict[str, Any]]:
    # list[dict]
    if isinstance(data, list):
        return data
    # dict -> [dict]
    if isinstance(data, dict):
        return [data]
    # pandas.DataFrame
    if pd is not None:
        try:
            if isinstance(data, pd.DataFrame):
                return data.to_dict("records")
        except Exception:
            pass
    # fallback
    return []

def _infer_columns(records: List[Dict[str, Any]]) -> List[str]:
    if not records:
        return []
    # берем ключи первой строки
    try:
        return list(records[0].keys())
    except Exception:
        return []

def ozon_table(*args, **kwargs) -> Dict[str, Any]:
    """
    Универсальный враппер:
    - не падает, если app передал "лишние" kwargs
    - принимает data как DataFrame / list[dict] / dict
    """

    # Поддержим и позиционные вызовы на всякий
    data = None
    columns = None
    state = None

    if len(args) >= 1:
        data = args[0]
    if len(args) >= 2:
        columns = args[1]
    if len(args) >= 3:
        state = args[2]

    data = kwargs.get("data", data)
    columns = kwargs.get("columns", columns)
    state = kwargs.get("state", state) or kwargs.get("table_state") or kwargs.get("value") or {}
    height = int(kwargs.get("height", 520) or 520)
    key = str(kwargs.get("key", "ozon_table_component") or "ozon_table_component")

    # Если dist не на месте — не роняем приложение
    if not (_BUILD_DIR / "index.html").exists():
        st.warning("Компонент таблицы не собран: нет frontend/dist/index.html. Показываю обычную таблицу.")
        recs = _to_records(data)
        st.dataframe(recs, use_container_width=True, height=height)
        return state if isinstance(state, dict) else {}

    recs = _to_records(data)

    # columns: может прийти списком, может None
    if isinstance(columns, (list, tuple)):
        cols = [str(c) for c in columns]
    else:
        cols = _infer_columns(recs)

    payload = {
        "data": recs,
        "columns": cols,
        "state": state if isinstance(state, dict) else {},
    }

    result = _ozon_table_component(
        **payload,
        default=payload["state"],
        key=key,
        height=height,
    )

    if result is None:
        return payload["state"]
    if isinstance(result, dict):
        return result
    return payload["state"]
