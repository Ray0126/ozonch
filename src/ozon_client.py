import time
import requests
from datetime import datetime, date, timedelta

API_BASE = "https://api-seller.ozon.ru"


class OzonAPIError(RuntimeError):
    """Ошибка запросов к Ozon Seller API.

    Используем отдельный класс, чтобы приложение Streamlit могло
    перехватывать ошибки и показывать пользователю понятное сообщение,
    вместо падения всего приложения.
    """

    def __init__(self, status_code: int, path: str, body: str):
        self.status_code = int(status_code)
        self.path = str(path)
        self.body = str(body)
        super().__init__(f"{self.status_code} {self.path}: {self.body}")

def _as_dt_str(d: str, end: bool = False) -> str:
    """
    Ozon /v3/finance/transaction/list expects google.protobuf.Timestamp (RFC3339).
    Examples: 2025-11-29T00:00:00Z
    """
    # if already RFC3339-ish, return as is
    if "T" in d and (d.endswith("Z") or "+" in d or d.endswith("00:00")):
        return d

    if len(d) == 10:  # YYYY-MM-DD
        return f"{d}T{'23:59:59' if end else '00:00:00'}Z"

    # fallback: try to normalize "YYYY-MM-DD HH:MM:SS"
    d = d.replace(" ", "T")
    if d.endswith("Z"):
        return d
    return d + "Z"

def _parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def last_closed_month(today: date | None = None) -> tuple[int,int]:
    """Returns (year, month) for the last fully closed calendar month."""
    today = today or date.today()
    first_this = today.replace(day=1)
    last_prev = first_this - timedelta(days=1)
    return last_prev.year, last_prev.month

def month_range(year: int, month: int) -> tuple[str,str]:
    start = date(year, month, 1)
    # next month
    if month == 12:
        end = date(year+1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month+1, 1) - timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


class OzonSellerClient:
    def __init__(self, client_id: str, api_key: str, timeout: int = 60):
        self.client_id = str(client_id)
        self.api_key = str(api_key)
        self.timeout = timeout

    def post(self, path: str, json_body: dict):
        url = API_BASE + path
        headers = {
            "Client-Id": self.client_id,
            "Api-Key": self.api_key,
            "Content-Type": "application/json",
        }
        r = requests.post(url, headers=headers, json=json_body, timeout=self.timeout)
        if r.status_code >= 400:
            # кидаем типизированную ошибку, чтобы UI мог показать дружелюбный алерт
            raise OzonAPIError(r.status_code, path, r.text)
        return r.json()

    def fetch_finance_transactions(self, date_from: str, date_to: str, page_size: int = 1000):
        """
        Fetches /v3/finance/transaction/list with stable pagination.

        IMPORTANT:
        - We page until page_count (if provided) OR until an empty page.
        - We do NOT rely on len(ops) < page_size, because Ozon can return
          a short page before the end (e.g. internal limits / filtering).
        """
        all_ops: list[dict] = []
        page = 1
        max_pages_guard = 5000  # safety to avoid infinite loops

        while page <= max_pages_guard:
            body = {
                "filter": {
                    "date": {
                        "from": _as_dt_str(date_from, end=False),
                        "to": _as_dt_str(date_to, end=True),
                    },
                    # keep other filters empty = all
                    "operation_type": [],
                    "posting_number": "",
                    "transaction_type": "all",
                },
                "page": page,
                "page_size": page_size,
            }

            resp = self.post("/v3/finance/transaction/list", body)
            result = resp.get("result", {}) if isinstance(resp, dict) else {}
            if not isinstance(result, dict):
                break

            ops = result.get("operations") or []
            if not isinstance(ops, list):
                ops = []

            if not ops:
                # empty page => no more data
                break

            all_ops.extend(ops)

            page_count = result.get("page_count")
            if page_count is not None:
                try:
                    page_count_int = int(page_count)
                except Exception:
                    page_count_int = None

                if page_count_int is not None and page >= page_count_int:
                    break

            page += 1
            time.sleep(0.12)

        return all_ops
    def fetch_analytics_sku_units(self, date_from: str, date_to: str, limit: int = 1000) -> list[dict]:
        """
        Fetches SKU-level quantity metrics from /v1/analytics/data.

        Why:
        - Finance transactions are NOT equal to "orders/buyout" shown in Ozon LK (юнит-экономика),
          especially on month boundaries. LK quantity metrics come from Analytics, not Finance.

        Returns list of rows:
        [{"sku": int, "ordered_units": float, "delivered_units": float, "returned_units": float, "cancelled_units": float}, ...]
        """
        def _call(dimensions: list[str], metrics: list[str], offset: int):
            body = {
                "date_from": date_from,
                "date_to": date_to,
                "dimension": dimensions,
                "metrics": metrics,
                "sort": [{"key": metrics[0], "order": "DESC"}] if metrics else [],
                "limit": int(limit),
                "offset": int(offset),
            }
            return self.post("/v1/analytics/data", body)

        # Try a couple of safe variants because Ozon accounts / API versions may differ in accepted keys.
        dim_variants = [["sku"], ["sku_id"]]
        metric_variants = [
            ["ordered_units", "delivered_units", "returned_units", "cancelled_units"],
            ["ordered_units", "delivered_units", "returned_units"],
            ["ordered_units", "delivered_units"],
            ["ordered_units"],
        ]

        last_err = None
        resp = None
        used_dims = None
        used_metrics = None
        for dims in dim_variants:
            for mets in metric_variants:
                try:
                    resp = _call(dims, mets, 0)
                    used_dims = dims
                    used_metrics = mets
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    resp = None
            if resp is not None:
                break

        if resp is None:
            raise last_err if last_err else RuntimeError("Failed to call /v1/analytics/data")

        rows: list[dict] = []
        offset = 0
        max_pages_guard = 5000

        def _parse_page(r) -> list[dict]:
            result = r.get("result", {}) if isinstance(r, dict) else {}
            data = result.get("data") or []
            out = []
            if not isinstance(data, list):
                return out
            for item in data:
                if not isinstance(item, dict):
                    continue
                dims = item.get("dimensions") or []
                mets = item.get("metrics") or []
                sku_val = None
                if isinstance(dims, list) and dims:
                    d0 = dims[0]
                    if isinstance(d0, dict):
                        sku_val = d0.get("id") or d0.get("value") or d0.get("name")
                    else:
                        sku_val = d0
                # metrics come as list aligned with requested metrics
                vals = {}
                if isinstance(mets, list):
                    for i, key in enumerate(used_metrics or []):
                        try:
                            vals[key] = float(mets[i]) if i < len(mets) and mets[i] is not None else 0.0
                        except Exception:
                            vals[key] = 0.0
                # normalize sku
                try:
                    sku_int = int(float(str(sku_val).strip()))
                except Exception:
                    continue
                out.append({
                    "sku": sku_int,
                    "ordered_units": float(vals.get("ordered_units", 0.0)),
                    "delivered_units": float(vals.get("delivered_units", 0.0)),
                    "returned_units": float(vals.get("returned_units", 0.0)),
                    "cancelled_units": float(vals.get("cancelled_units", 0.0)),
                })
            return out

        # First page already in resp
        first = _parse_page(resp)
        rows.extend(first)

        while len(first) == limit and offset/limit < max_pages_guard:
            offset += limit
            try:
                resp2 = _call(used_dims, used_metrics, offset)
            except Exception as e:
                break
            page_rows = _parse_page(resp2)
            if not page_rows:
                break
            rows.extend(page_rows)
            first = page_rows
            time.sleep(0.12)

        return rows

