"""Ozon table Streamlit component.

Renders an interactive table with:
- drag&drop column reorder
- column hide/show
- header filters
- sort
- persistent layout (localStorage key per table_id)

Frontend is shipped as static files in ozon_table_component/frontend/dist.
"""

from .component import ozon_table

__all__ = ["ozon_table"]
