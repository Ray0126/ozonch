/* Minimal Streamlit component without npm deps.
   Supports:
   - drag&drop column order
   - per-column hide
   - simple column filters row toggle
   - sort on header click
   - returns view state to Streamlit (columnOrder, hidden, filters, sort)
*/

let ARGS = null;
let STATE = {
  columnOrder: [],
  hidden: {},
  filters: {},
  sort: { id: null, desc: false },
  showFilters: false,
};

const el = (id) => document.getElementById(id);

function postToStreamlit(type, data) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*");
}

function setFrameHeight() {
  const h = document.documentElement.scrollHeight;
  postToStreamlit("streamlit:setFrameHeight", { height: h });
}

function setValue(value) {
  postToStreamlit("streamlit:setComponentValue", { value });
}

function normalizeColumns(cols, data) {
  if (Array.isArray(cols) && cols.length) return cols;
  const first = (data && data[0]) || {};
  return Object.keys(first).map((k) => ({ id: k, header: k }));
}

function renderControls() {
  el("btn-cols").onclick = () => {
    const p = el("colsPanel");
    p.style.display = p.style.display === "none" ? "block" : "none";
    setFrameHeight();
  };
  el("btn-filters").onclick = () => {
    STATE.showFilters = !STATE.showFilters;
    render();
    setFrameHeight();
  };
  el("btn-save").onclick = () => {
    setValue({
      columnOrder: STATE.columnOrder,
      hidden: STATE.hidden,
      filters: STATE.filters,
      sort: STATE.sort,
      showFilters: STATE.showFilters,
    });
  };
  el("btn-reset").onclick = () => {
    // reset to defaults
    const cols = normalizeColumns(ARGS.columns, ARGS.data);
    STATE.columnOrder = cols.map((c) => c.id);
    STATE.hidden = {};
    STATE.filters = {};
    STATE.sort = { id: null, desc: false };
    STATE.showFilters = false;
    setValue({ __reset__: true });
    render();
    setFrameHeight();
  };
}

function buildColsPanel(cols) {
  const box = el("colsPanel");
  box.innerHTML = "";
  cols.forEach((c) => {
    const row = document.createElement("label");
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.gap = "8px";
    row.style.fontSize = "12px";
    row.style.margin = "2px 0";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !STATE.hidden[c.id];
    cb.onchange = () => {
      STATE.hidden[c.id] = !cb.checked;
      render();
      setFrameHeight();
    };
    const sp = document.createElement("span");
    sp.textContent = c.header || c.id;
    row.appendChild(cb);
    row.appendChild(sp);
    box.appendChild(row);
  });
}

function applyFilters(data, colsInOrder) {
  let out = data;
  for (const c of colsInOrder) {
    if (STATE.hidden[c.id]) continue;
    const fv = (STATE.filters[c.id] || "").toString().trim().toLowerCase();
    if (!fv) continue;
    out = out.filter((row) => {
      const v = (row[c.id] ?? "").toString().toLowerCase();
      return v.includes(fv);
    });
  }
  return out;
}

function applySort(data) {
  const { id, desc } = STATE.sort || {};
  if (!id) return data;
  const out = [...data];
  out.sort((a, b) => {
    const va = a[id];
    const vb = b[id];
    const na = Number(String(va).replace(/\s/g, "").replace(",", "."));
    const nb = Number(String(vb).replace(/\s/g, "").replace(",", "."));
    const bothNum = Number.isFinite(na) && Number.isFinite(nb);
    let cmp = 0;
    if (bothNum) cmp = na - nb;
    else cmp = String(va ?? "").localeCompare(String(vb ?? ""), "ru");
    return desc ? -cmp : cmp;
  });
  return out;
}

function renderHeader(colsInOrder) {
  const thead = el("thead");
  thead.innerHTML = "";
  const tr = document.createElement("tr");
  colsInOrder.forEach((c) => {
    if (STATE.hidden[c.id]) return;
    const th = document.createElement("th");
    th.textContent = c.header || c.id;
    th.draggable = true;
    th.dataset.colId = c.id;
    th.title = "Клик: сортировка. Перетащи: порядок колонок";
    th.onclick = () => {
      if (STATE.sort.id !== c.id) STATE.sort = { id: c.id, desc: false };
      else STATE.sort = { id: c.id, desc: !STATE.sort.desc };
      render();
    };

    th.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", c.id);
      e.dataTransfer.effectAllowed = "move";
    });
    th.addEventListener("dragover", (e) => {
      e.preventDefault();
      th.style.background = "var(--muted)";
    });
    th.addEventListener("dragleave", () => {
      th.style.background = "";
    });
    th.addEventListener("drop", (e) => {
      e.preventDefault();
      th.style.background = "";
      const fromId = e.dataTransfer.getData("text/plain");
      const toId = c.id;
      if (!fromId || fromId === toId) return;
      const order = [...STATE.columnOrder];
      const a = order.indexOf(fromId);
      const b = order.indexOf(toId);
      if (a === -1 || b === -1) return;
      order.splice(a, 1);
      order.splice(b, 0, fromId);
      STATE.columnOrder = order;
      render();
      setFrameHeight();
    });

    tr.appendChild(th);
  });
  thead.appendChild(tr);

  if (STATE.showFilters) {
    const trf = document.createElement("tr");
    colsInOrder.forEach((c) => {
      if (STATE.hidden[c.id]) return;
      const th = document.createElement("th");
      const inp = document.createElement("input");
      inp.type = "text";
      inp.placeholder = "фильтр…";
      inp.value = STATE.filters[c.id] || "";
      inp.oninput = () => {
        STATE.filters[c.id] = inp.value;
        renderBody(colsInOrder);
      };
      th.appendChild(inp);
      trf.appendChild(th);
    });
    thead.appendChild(trf);
  }
}

function renderBody(colsInOrder) {
  const tbody = el("tbody");
  tbody.innerHTML = "";
  const filtered = applySort(applyFilters(ARGS.data || [], colsInOrder));
  filtered.forEach((row) => {
    const tr = document.createElement("tr");
    colsInOrder.forEach((c) => {
      if (STATE.hidden[c.id]) return;
      const td = document.createElement("td");
      const v = row[c.id];
      td.textContent = v === null || v === undefined ? "" : String(v);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function render() {
  if (!ARGS) return;
  const cols = normalizeColumns(ARGS.columns, ARGS.data);
  if (!STATE.columnOrder.length) STATE.columnOrder = cols.map((c) => c.id);
  const colsById = new Map(cols.map((c) => [c.id, c]));
  const colsInOrder = STATE.columnOrder.map((id) => colsById.get(id)).filter(Boolean);

  buildColsPanel(cols);
  renderHeader(colsInOrder);
  renderBody(colsInOrder);
}

function initFromArgs(args) {
  ARGS = args || { data: [], columns: [] };

  // restore state passed from python
  const s = (args && args.view_state) || {};
  const cols = normalizeColumns(args.columns, args.data);

  STATE.columnOrder = Array.isArray(s.columnOrder) && s.columnOrder.length ? s.columnOrder : cols.map((c) => c.id);
  STATE.hidden = (s.hidden && typeof s.hidden === "object") ? s.hidden : {};
  STATE.filters = (s.filters && typeof s.filters === "object") ? s.filters : {};
  STATE.sort = (s.sort && typeof s.sort === "object") ? s.sort : { id: null, desc: false };
  STATE.showFilters = !!s.showFilters;

  el("colsPanel").style.display = "none";

  renderControls();
  render();
  setFrameHeight();
}

// Streamlit will send args via postMessage
window.addEventListener("message", (event) => {
  const msg = event.data;
  if (!msg) return;
  if (msg.type === "streamlit:render") {
    initFromArgs(msg.args);
  }
});

// Let Streamlit know we're ready
postToStreamlit("streamlit:componentReady", { apiVersion: 1 });
postToStreamlit("streamlit:setFrameHeight", { height: 300 });
