import streamlit.components.v1 as components
from pathlib import Path

_DIST_DIR = Path(__file__).parent / "frontend" / "dist"

_component = components.declare_component(
    "ozon_table_component",
    path=str(_DIST_DIR),
)

def ozon_table(
    df=None,
    data=None,
    columns=None,
    state=None,
    key=None,
    height=520,
    **kwargs,
):
    # если передали df — конвертим в records
    if df is not None and data is None:
        data = df.to_dict("records")
        if columns is None:
            columns = list(df.columns)

    if data is None:
        data = []
    if columns is None:
        columns = []

    return _component(
        data=data,
        columns=columns,
        state=state or {},
        height=height,
        key=key,
        default=None,
        **kwargs,
    )
