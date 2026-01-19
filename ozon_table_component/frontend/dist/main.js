(function () {
  const root = document.getElementById('root');
  let lastArgs = null;

  // State returned to Streamlit
  let stState = {
    column_order: null,   // array of column ids
    global_filter: '',
  };

  function postToStreamlit(type, payload) {
    const msg = Object.assign({ isStreamlitMessage: true, type: type }, payload || {});
    window.parent.postMessage(msg, '*');
  }

  function setFrameHeight() {
    const h = document.documentElement.scrollHeight;
    postToStreamlit('streamlit:setFrameHeight', { height: h });
  }

  function sendValue() {
    postToStreamlit('streamlit:setComponentValue', { value: stState });
  }

  function el(tag, attrs, children) {
    const e = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach((k) => {
        if (k === 'style') {
          Object.assign(e.style, attrs[k]);
        } else if (k.startsWith('on') && typeof attrs[k] === 'function') {
          e.addEventListener(k.substring(2).toLowerCase(), attrs[k]);
        } else {
          e.setAttribute(k, String(attrs[k]));
        }
      });
    }
    if (children) {
      children.forEach((c) => {
        if (c == null) return;
        if (typeof c === 'string') e.appendChild(document.createTextNode(c));
        else e.appendChild(c);
      });
    }
    return e;
  }

  function normalizeArgs(args) {
    const data = Array.isArray(args.data) ? args.data : [];
    const columns = Array.isArray(args.columns) ? args.columns : [];
    const stateIn = args.state && typeof args.state === 'object' ? args.state : {};

    if (Array.isArray(stateIn.column_order)) {
      stState.column_order = stateIn.column_order.slice();
    }
    if (typeof stateIn.global_filter === 'string') {
      stState.global_filter = stateIn.global_filter;
    }

    let order = stState.column_order;
    if (!order || order.length === 0) {
      order = columns.map((c) => String(c.id));
      stState.column_order = order.slice();
    }

    // keep only known columns
    order = order.filter((id) => columns.some((c) => String(c.id) === id));
    const missing = columns.map((c) => String(c.id)).filter((id) => !order.includes(id));
    order = order.concat(missing);
    stState.column_order = order.slice();

    const colById = {};
    columns.forEach((c) => { colById[String(c.id)] = c; });

    return { data, columns, order, colById };
  }

  function renderTable(args) {
    root.innerHTML = '';

    const { data, columns, order, colById } = normalizeArgs(args);

    const container = el('div', { style: { padding: '8px', fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif' } });

    const search = el('input', {
      type: 'text',
      value: stState.global_filter,
      placeholder: 'Поиск…',
      style: {
        width: '100%',
        padding: '8px 10px',
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        marginBottom: '8px'
      },
      oninput: (ev) => {
        stState.global_filter = ev.target.value || '';
        sendValue();
        renderTable(lastArgs);
      }
    });

    const table = el('table', {
      style: {
        width: '100%',
        borderCollapse: 'collapse',
        border: '1px solid #e5e7eb',
        borderRadius: '10px',
        overflow: 'hidden'
      }
    });

    const thead = el('thead');
    const trh = el('tr', null, []);

    let dragCol = null;

    function onDragStart(ev, colId) {
      dragCol = colId;
      ev.dataTransfer.effectAllowed = 'move';
      try { ev.dataTransfer.setData('text/plain', colId); } catch (_) {}
    }

    function onDrop(ev, targetId) {
      ev.preventDefault();
      const fromId = dragCol || (function(){ try { return ev.dataTransfer.getData('text/plain'); } catch (_) { return null; } })();
      dragCol = null;
      if (!fromId || fromId === targetId) return;
      const cur = stState.column_order.slice();
      const fromIdx = cur.indexOf(fromId);
      const toIdx = cur.indexOf(targetId);
      if (fromIdx < 0 || toIdx < 0) return;
      cur.splice(fromIdx, 1);
      cur.splice(toIdx, 0, fromId);
      stState.column_order = cur;
      sendValue();
      renderTable(lastArgs);
    }

    order.forEach((id) => {
      const c = colById[id] || { id, label: id };
      const th = el('th', {
        draggable: 'true',
        style: {
          textAlign: 'left',
          padding: '8px 10px',
          fontSize: '12px',
          fontWeight: '600',
          color: '#374151',
          background: '#f9fafb',
          borderBottom: '1px solid #e5e7eb',
          cursor: 'grab',
          whiteSpace: 'nowrap'
        },
        ondragstart: (ev) => onDragStart(ev, id),
        ondragover: (ev) => { ev.preventDefault(); ev.dataTransfer.dropEffect = 'move'; },
        ondrop: (ev) => onDrop(ev, id)
      }, [String(c.label || c.id)]);
      trh.appendChild(th);
    });

    thead.appendChild(trh);
    table.appendChild(thead);

    const tbody = el('tbody');

    const q = (stState.global_filter || '').toLowerCase().trim();

    const filtered = q
      ? data.filter((row) => {
          const s = order.map((id) => String(row[id] ?? '')).join(' | ').toLowerCase();
          return s.includes(q);
        })
      : data;

    filtered.forEach((row) => {
      const tr = el('tr');
      order.forEach((id) => {
        const v = row[id];
        const td = el('td', {
          style: {
            padding: '8px 10px',
            fontSize: '12px',
            borderBottom: '1px solid #f1f5f9',
            verticalAlign: 'top'
          }
        }, [v == null ? '' : String(v)]);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);

    container.appendChild(search);
    container.appendChild(table);
    root.appendChild(container);

    // Resize iframe
    setTimeout(setFrameHeight, 0);
  }

  function handleRender(event) {
    const data = event && event.data ? event.data : null;
    if (!data || data.type !== 'streamlit:render') return;

    const args = data.args || {};
    lastArgs = args;

    // Let Streamlit know we're ready
    postToStreamlit('streamlit:componentReady', { apiVersion: 1 });

    renderTable(args);
  }

  window.addEventListener('message', handleRender);

  // In case Streamlit sends the first render before our listener is attached (rare)
  postToStreamlit('streamlit:componentReady', { apiVersion: 1 });
})();
