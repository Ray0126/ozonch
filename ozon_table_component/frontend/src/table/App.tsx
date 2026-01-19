import React, { useMemo, useState } from "react";
import { Streamlit, withStreamlitConnection, ComponentProps } from "streamlit-component-lib";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
  ColumnFiltersState,
  VisibilityState,
  ColumnOrderState,
} from "@tanstack/react-table";
import {
  DndContext,
  closestCenter,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  horizontalListSortingStrategy,
  useSortable,
  arrayMove,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

type OttState = {
  columnOrder?: string[];
  columnVisibility?: Record<string, boolean>;
  sorting?: SortingState;
  columnFilters?: ColumnFiltersState;
  globalFilter?: string;
};

type OttArgs = {
  data: Record<string, any>[];
  columns: { id: string; header: string; dtype: string }[];
  state?: OttState;
  height?: number;
  theme?: string;
  compact?: boolean;
  showToolbar?: boolean;
  defaultPageSize?: number;
};

function HeaderCell({ id, label }: { id: string; label: string }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });
  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.6 : 1,
    cursor: "grab",
    userSelect: "none",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  };
  return (
    <div ref={setNodeRef} style={style} {...attributes} {...listeners} title={label}>
      {label}
    </div>
  );
}

function App(props: ComponentProps) {
  const args = props.args as OttArgs;
  const height = (args.height ?? 520) as number;
  const showToolbar = args.showToolbar ?? true;
  const compact = args.compact ?? true;

  const colsMeta = args.columns ?? [];
  const data = args.data ?? [];

  const initialState = (args.state ?? {}) as OttState;

  const [globalFilter, setGlobalFilter] = useState<string>(initialState.globalFilter ?? "");
  const [sorting, setSorting] = useState<SortingState>(initialState.sorting ?? []);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>(initialState.columnFilters ?? []);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(initialState.columnVisibility ?? {});
  const [columnOrder, setColumnOrder] = useState<ColumnOrderState>(
    initialState.columnOrder ?? colsMeta.map((c) => c.id)
  );
  const [showCols, setShowCols] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  const columns = useMemo<ColumnDef<Record<string, any>>[]>(
    () =>
      colsMeta.map((c) => ({
        id: c.id,
        accessorKey: c.id,
        header: () => c.header,
        cell: (info) => {
          const v = info.getValue();
          if (v === null || v === undefined) return "";
          return String(v);
        },
      })),
    [colsMeta]
  );

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
      globalFilter,
      columnFilters,
      columnVisibility,
      columnOrder,
    },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: (v) => setGlobalFilter(String(v ?? "")),
    onColumnVisibilityChange: setColumnVisibility,
    onColumnOrderChange: setColumnOrder,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    enableSorting: true,
    enableColumnFilters: true,
    globalFilterFn: "includesString",
  });

  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }));

  const onDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over) return;
    if (active.id !== over.id) {
      setColumnOrder((prev) => {
        const oldIndex = prev.indexOf(String(active.id));
        const newIndex = prev.indexOf(String(over.id));
        return arrayMove(prev, oldIndex, newIndex);
      });
    }
  };

  const saveView = () => {
    // Return state flat (so Streamlit gets a dict of fields directly)
    Streamlit.setComponentValue({
      globalFilter,
      sorting,
      columnFilters,
      columnVisibility,
      columnOrder,
    } as OttState);
  };

  const resetView = () => {
    setGlobalFilter("");
    setSorting([]);
    setColumnFilters([]);
    setColumnVisibility({});
    setColumnOrder(colsMeta.map((c) => c.id));
    Streamlit.setComponentValue({} as OttState);
  };

  const wrapStyle: React.CSSProperties = {
    display: "flex",
    flexDirection: "column",
    gap: 10,
  };

  const barStyle: React.CSSProperties = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 10,
    flexWrap: "wrap",
  };

  const btnStyle: React.CSSProperties = {
    border: "1px solid rgba(0,0,0,0.2)",
    background: "rgba(0,0,0,0.03)",
    padding: "6px 10px",
    borderRadius: 8,
    cursor: "pointer",
  };

  const inputStyle: React.CSSProperties = {
    border: "1px solid rgba(0,0,0,0.2)",
    padding: "6px 10px",
    borderRadius: 8,
    minWidth: 220,
  };

  const tableWrapStyle: React.CSSProperties = {
    border: "1px solid rgba(0,0,0,0.12)",
    borderRadius: 10,
    overflow: "hidden",
  };

  const scrollStyle: React.CSSProperties = {
    height,
    overflow: "auto",
  };

  const thStyle: React.CSSProperties = {
    position: "sticky",
    top: 0,
    background: "rgba(0,0,0,0.03)",
    textAlign: "left",
    padding: compact ? "8px 10px" : "10px 12px",
    borderBottom: "1px solid rgba(0,0,0,0.12)",
    fontWeight: 600,
    fontSize: 13,
  };

  const tdStyle: React.CSSProperties = {
    padding: compact ? "7px 10px" : "10px 12px",
    borderBottom: "1px solid rgba(0,0,0,0.08)",
    verticalAlign: "top",
    fontSize: 13,
  };

  const smallMuted: React.CSSProperties = { fontSize: 12, opacity: 0.7 };

  const headerIds = table.getAllLeafColumns().map((c) => c.id);

  return (
    <div style={wrapStyle}>
      {showToolbar && (
        <div style={barStyle}>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <input
              style={inputStyle}
              value={globalFilter ?? ""}
              onChange={(e) => setGlobalFilter(e.target.value)}
              placeholder="Поиск по таблице…"
            />
            <button style={btnStyle} onClick={() => setShowCols((v) => !v)}>
              Колонки
            </button>
            <button style={btnStyle} onClick={() => setShowFilters((v) => !v)}>
              Фильтры
            </button>
            <span style={smallMuted}>Строк: {table.getRowModel().rows.length}</span>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button style={btnStyle} onClick={saveView} title="Сохранить порядок/фильтры">
              💾 Сохранить вид
            </button>
            <button style={btnStyle} onClick={resetView} title="Сбросить к дефолту">
              ↩️ Сбросить
            </button>
          </div>
        </div>
      )}

      {showCols && (
        <div
          style={{
            border: "1px solid rgba(0,0,0,0.12)",
            borderRadius: 10,
            padding: 10,
            display: "flex",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          {table.getAllLeafColumns().map((col) => (
            <label key={col.id} style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={col.getIsVisible()}
                onChange={(e) => col.toggleVisibility(e.target.checked)}
              />
              <span>{colsMeta.find((c) => c.id === col.id)?.header ?? col.id}</span>
            </label>
          ))}
        </div>
      )}

      <div style={tableWrapStyle}>
        <div style={scrollStyle}>
          <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: 0 }}>
            <thead>
              {table.getHeaderGroups().map((hg) => (
                <React.Fragment key={hg.id}>
                <tr>
                  <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
                    <SortableContext items={headerIds} strategy={horizontalListSortingStrategy}>
                      {hg.headers.map((h) => {
                        const label = colsMeta.find((c) => c.id === h.column.id)?.header ?? h.column.id;
                        const sort = h.column.getIsSorted();
                        return (
                          <th
                            key={h.id}
                            style={{ ...thStyle, minWidth: 110 }}
                            onClick={h.column.getToggleSortingHandler()}
                            title="Клик = сортировка, drag = перестановка"
                          >
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                              <HeaderCell id={h.column.id} label={label} />
                              <span style={{ fontSize: 12, opacity: 0.7 }}>
                                {sort === "asc" ? "▲" : sort === "desc" ? "▼" : ""}
                              </span>
                            </div>
                          </th>
                        );
                      })}
                    </SortableContext>
                  </DndContext>
                </tr>
                {showFilters && (
                  <tr>
                    {hg.headers.map((h) => {
                      const col = h.column;
                      return (
                        <th key={h.id} style={{ ...thStyle, fontWeight: 400 }}>
                          <input
                            style={{ ...inputStyle, width: "100%", padding: "6px 8px" }}
                            value={(col.getFilterValue() as string) ?? ""}
                            onChange={(e) => col.setFilterValue(e.target.value)}
                            placeholder="фильтр…"
                          />
                        </th>
                      );
                    })}
                  </tr>
                )}
                </React.Fragment>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => (
                <tr key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} style={tdStyle}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default withStreamlitConnection(App);
