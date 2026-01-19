/* global Streamlit, Tabulator */

function getLSKey(tableId) {
  return `ozon_table_component::${tableId}`;
}

function safeParseJSON(s, fallback) {
  try { return JSON.parse(s); } catch (e) { return fallback; }
}

let table = null;
let lastTableId = null;

function buildColumns(data, columnsFromPy) {
  if (Array.isArray(columnsFromPy) && columnsFromPy.length) {
    return columnsFromPy.map((c) => {
      if (typeof c === 'string') {
        return { title: c, field: c, headerFilter: true, hozAlign: 'left' };
      }
      // allow dict like {field,title,formatter,hozAlign,headerFilter}
      return {
        title: c.title || c.field,
        field: c.field,
        headerFilter: c.headerFilter !== undefined ? c.headerFilter : true,
        hozAlign: c.hozAlign || 'left',
        formatter: c.formatter || undefined,
      };
    });
  }

  // infer from data keys
  const keys = data && data.length ? Object.keys(data[0]) : [];
  return keys.map((k) => ({ title: k, field: k, headerFilter: true, hozAlign: 'left' }));
}

function applyLayoutFromStorage(tableId) {
  const raw = window.localStorage.getItem(getLSKey(tableId));
  const st = safeParseJSON(raw, null);
  if (!st || !table) return;

  // column layout
  if (Array.isArray(st.columnsLayout)) {
    try { table.setColumnLayout(st.columnsLayout); } catch (e) {}
  }

  // sort
  if (Array.isArray(st.sorters)) {
    try { table.setSort(st.sorters); } catch (e) {}
  }

  // filters
  if (Array.isArray(st.filters)) {
    try { table.setFilter(st.filters); } catch (e) {}
  }
}

function saveLayoutToStorage(tableId) {
  if (!table) return;
  const state = {
    columnsLayout: table.getColumnLayout ? table.getColumnLayout() : null,
    sorters: table.getSorters ? table.getSorters().map(s => ({ field: s.field, dir: s.dir })) : null,
    filters: table.getFilters ? table.getFilters().map(f => ({ field: f.field, type: f.type, value: f.value })) : null,
  };
  window.localStorage.setItem(getLSKey(tableId), JSON.stringify(state));

  // send to python too
  Streamlit.setComponentValue(state);
}

function clearStorage(tableId) {
  window.localStorage.removeItem(getLSKey(tableId));
}

function render({ data, columns, table_id, height }) {
  const root = document.getElementById('root');
  const tableId = table_id || 'table';

  // full rebuild if table_id changed
  const needRebuild = !table || lastTableId !== tableId;
  lastTableId = tableId;

  if (needRebuild) {
    root.innerHTML = '<div id="tbl"></div><div style="padding:6px 0; display:flex; gap:8px; justify-content:flex-end;">' +
      '<button id="saveBtn" style="font-size:12px; padding:4px 8px;">💾 Сохранить вид</button>' +
      '<button id="resetBtn" style="font-size:12px; padding:4px 8px;">↩️ Сбросить</button>' +
      '</div>';

    const cols = buildColumns(data, columns);

    table = new Tabulator('#tbl', {
      data: data || [],
      columns: cols,
      layout: 'fitDataFill',
      height: height || 520,
      movableColumns: true,
      resizableColumnFit: true,
      columnHeaderSortMulti: true,
      clipboard: true,
      clipboardCopyRowRange: 'range',
      placeholder: 'Нет данных',
    });

    // apply persisted layout after table is ready
    table.on('tableBuilt', () => {
      applyLayoutFromStorage(tableId);
      Streamlit.setFrameHeight();
    });

    // auto-save on key changes
    table.on('columnMoved', () => saveLayoutToStorage(tableId));
    table.on('columnVisibilityChanged', () => saveLayoutToStorage(tableId));
    table.on('columnResized', () => saveLayoutToStorage(tableId));
    table.on('dataFiltered', () => saveLayoutToStorage(tableId));
    table.on('dataSorted', () => saveLayoutToStorage(tableId));

    document.getElementById('saveBtn').addEventListener('click', () => saveLayoutToStorage(tableId));
    document.getElementById('resetBtn').addEventListener('click', () => {
      clearStorage(tableId);
      // soft reset: rebuild
      table.destroy();
      table = null;
      render({ data, columns, table_id: tableId, height });
    });
  } else {
    // update data fast
    table.replaceData(data || []);
  }

  Streamlit.setFrameHeight();
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, (event) => {
  const args = event.detail.args || {};
  render({
    data: args.data || [],
    columns: args.columns || [],
    table_id: args.table_id || 'table',
    height: args.height || 520,
  });
});

Streamlit.setComponentReady();
Streamlit.setFrameHeight();
