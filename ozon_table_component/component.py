# ozon_table_component/component.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

# Путь к фронту (dist)
_FRONTEND_DIR = (Path(__file__).parent / "frontend" / "dist").resolve()

_component = components.declare_component(
    "ozon_table_component",
    path=str(_FRONTEND_DIR),
)

def _df_to_records(x: Any) -> List[Dict[str, Any]]:
    # pandas.DataFrame -> records
    try:
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            return x.fillna("").to_dict(orient="records")
    except Exception:
        pass

    # already list[dict]
    if isinstance(x, list):
        if not x:
            return []
        if isinstance(x[0], dict):
            return x
    return []

def _build_columns_from_data(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        return []
    keys = list(records[0].keys())
    cols = []
    for k in keys:
        cols.append({"id": k, "header": str(k), "accessorKey": k})
    return cols

def ozon_table(
    data: Any,
    columns: Optional[List[Dict[str, Any]]] = None,
    state: Optional[Dict[str, Any]] = None,
    key: Optional[str] = None,
    height: int = 520,
) -> Dict[str, Any]:
    """
    Возвращает state (например: порядок колонок, скрытые, фильтры).
    """
    records = _df_to_records(data)

    # если columns не передали — строим сами из данных
    if not columns:
        columns = _build_columns_from_data(records)
    else:
        # нормализуем, чтобы фронт точно получил header+accessorKey
        norm = []
        for c in columns:
            if not isinstance(c, dict):
                continue
            acc = c.get("accessorKey") or c.get("field") or c.get("key") or c.get("id")
            hdr = c.get("header") or c.get("Header") or c.get("label") or acc
            cid = c.get("id") or acc
            if not acc:
                continue
            norm.append({"id": str(cid), "header": str(hdr), "accessorKey": str(acc)})
        columns = norm

    payload = {
        "data": records,
        "columns": columns,
        "state": state or {},
    }

    result = _component(payload, key=key, default=state or {}, height=height)
    return result or {}
