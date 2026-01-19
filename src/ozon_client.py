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
        This matches the export you provided from:
        Финансы → Экономика магазина → Детализация начислений → Скачать отчёт
        (operations list).
        """
        all_ops = []
        page = 1
        while True:
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
            result = resp.get("result", {})
            ops = result.get("operations", []) if isinstance(result, dict) else []
            all_ops.extend(ops)

            if len(ops) < page_size:
                break
            page += 1
            time.sleep(0.12)
        return all_ops
