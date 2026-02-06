import os
import io
import time
import json
import requests
import streamlit as st
import pandas as pd

# ================== PERFORMANCE PRODUCTS REPORT (Spend Click by SKU) ==================
def _rfc3339_day(s: str, end: bool = False) -> str:
    s = str(s or "").strip()
    if "T" in s:
        return s
    return f"{s}T23:59:59Z" if end else f"{s}T00:00:00Z"

def _parse_ru_money(x) -> float:
    """Парсит деньги из строк вида '1 234,56', '-', '—', None."""
    if x is None:
        return 0.0
    s = str(x).strip()
    if s in ("", "-", "—", "None", "nan", "NaN"):
        return 0.0
    s = s.replace("\ufeff", "").replace("\xa0", " ").replace("₽", "").strip()
    s = s.replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0

@st.cache_data(ttl=60*30, show_spinner=False)
def load_perf_spend_click_by_sku(date_from: str, date_to: str) -> dict:
    """Возвращает {sku:int -> spend_click(float)} из отчёта Performance products (JSON).

    Поле spend_click берём напрямую из отчёта: MoneySpentFromCPC.
    Ключи берём из Secrets/.env:
      - PERF_CLIENT_ID / PERF_CLIENT_SECRET
      - fallback: OZON_PERF_CLIENT_ID / OZON_PERF_CLIENT_SECRET
      - fallback: OZON_CLIENT_ID / OZON_CLIENT_SECRET
    """
    perf_id = (st.secrets.get("PERF_CLIENT_ID", None) if hasattr(st, "secrets") else None) or os.getenv("PERF_CLIENT_ID") or os.getenv("OZON_PERF_CLIENT_ID") or os.getenv("OZON_CLIENT_ID")
    perf_secret = (st.secrets.get("PERF_CLIENT_SECRET", None) if hasattr(st, "secrets") else None) or os.getenv("PERF_CLIENT_SECRET") or os.getenv("OZON_PERF_CLIENT_SECRET") or os.getenv("OZON_CLIENT_SECRET")

    perf_id = str(perf_id or "").strip()
    perf_secret = str(perf_secret or "").strip()
    if not perf_id or not perf_secret:
        # нет ключей — просто вернём пусто (в UI будет 0)
        return {}

    BASE = "https://api-performance.ozon.ru"

    # 1) token
    try:
        r = requests.post(
            f"{BASE}/api/client/token",
            json={"client_id": perf_id, "client_secret": perf_secret, "grant_type": "client_credentials"},
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=60,
        )
    except Exception:
        return {}

    if r.status_code != 200:
        return {}

    token = (r.json() or {}).get("access_token") or ""
    if not token:
        return {}

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}

    # 2) generate json report
    try:
        gen = requests.post(
            f"{BASE}/api/client/statistics/products/generate/json",
            json={"from": _rfc3339_day(date_from, end=False), "to": _rfc3339_day(date_to, end=True)},
            headers=headers,
            timeout=60,
        )
    except Exception:
        return {}

    if gen.status_code != 200:
        return {}

    uuid = (gen.json() or {}).get("UUID") or (gen.json() or {}).get("uuid")
    if not uuid:
        return {}

    # 3) poll report json (ВАЖНО: у Ozon встречаются разные пути выдачи)
    report_headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    candidates = [
        (f"{BASE}/api/client/statistics/report/json", {"UUID": str(uuid)}),
        (f"{BASE}/api/client/statistics/report", {"UUID": str(uuid), "format": "json"}),
        (f"{BASE}/api/client/statistics/report", {"UUID": str(uuid)}),
    ]

    last_err = None
    data = None
    for _ in range(90):  # ~180 секунд при sleep 2
        for url, params in candidates:
            try:
                rep = requests.get(url, params=params, headers=report_headers, timeout=60)
            except Exception as e:
                last_err = f"request error: {e}"
                continue

            if rep.status_code == 200:
                try:
                    data = rep.json()
                except Exception:
                    # иногда content-type странный, но json есть
                    try:
                        data = json.loads(rep.text)
                    except Exception:
                        return {}
                break

            if rep.status_code in (404, 409, 425):
                # отчёт ещё готовится
                last_err = f"{rep.status_code} preparing"
                continue

            # 400/401/5xx — реальные ошибки
            last_err = f"{rep.status_code} {rep.text[:200]}"
            return {}

        if data is not None:
            break
        time.sleep(2)

    if data is None:
        return {}

    rows = (data.get("rows") or data.get("Rows") or data.get("result") or data.get("data") or [])
    if not isinstance(rows, list) or not rows:
        return {}

    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    # SKU
    if "SKU" in df.columns:
        sku_field = "SKU"
    elif "Sku" in df.columns:
        sku_field = "Sku"
    elif "sku" in df.columns:
        sku_field = "sku"
    else:
        return {}

    df[sku_field] = pd.to_numeric(df[sku_field], errors="coerce")
    df = df.dropna(subset=[sku_field]).copy()
    if df.empty:
        return {}
    df[sku_field] = df[sku_field].astype(int)

    # MoneySpentFromCPC (spend_click)
    if "MoneySpentFromCPC" in df.columns:
        col = "MoneySpentFromCPC"
    elif "moneySpentFromCPC" in df.columns:
        col = "moneySpentFromCPC"
    else:
        # если вдруг переименовали — лучше 0, чем падать
        return {}

    df["spend_click"] = df[col].apply(_parse_ru_money)

    agg = df.groupby(sku_field, as_index=True)["spend_click"].sum()
    out = {int(k): float(v) for k, v in agg.to_dict().items()}
    return out

# --- DEBUG helper: raw Performance products report ---
def _perf_products_report_debug(date_from: str, date_to: str) -> dict:
    """DEBUG: возвращает мета-информацию по отчёту Performance products (json), чтобы понять почему 0.
    Ничего не кэширует и не меняет расчёты.
    """
    info = {"ok": False, "stage": None, "status": None}
    perf_id = (st.secrets.get("PERF_CLIENT_ID", None) if hasattr(st, "secrets") else None) or os.getenv("PERF_CLIENT_ID") or os.getenv("OZON_PERF_CLIENT_ID") or os.getenv("OZON_CLIENT_ID")
    perf_secret = (st.secrets.get("PERF_CLIENT_SECRET", None) if hasattr(st, "secrets") else None) or os.getenv("PERF_CLIENT_SECRET") or os.getenv("OZON_PERF_CLIENT_SECRET") or os.getenv("OZON_CLIENT_SECRET")
    perf_id = str(perf_id or "").strip()
    perf_secret = str(perf_secret or "").strip()
    info["has_keys"] = bool(perf_id and perf_secret)
    if not perf_id or not perf_secret:
        info["stage"] = "keys"
        return info

    BASE = "https://api-performance.ozon.ru"

    # token
    info["stage"] = "token"
    r = requests.post(
        f"{BASE}/api/client/token",
        json={"client_id": perf_id, "client_secret": perf_secret, "grant_type": "client_credentials"},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=60,
    )
    info["status"] = r.status_code
    if r.status_code != 200:
        info["error"] = (r.text or "")[:500]
        return info
    token = (r.json() or {}).get("access_token") or ""
    info["token"] = bool(token)
    if not token:
        info["error"] = "no access_token"
        return info

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}

    # generate
    info["stage"] = "generate"
    gen = requests.post(
        f"{BASE}/api/client/statistics/products/generate/json",
        json={"from": _rfc3339_day(date_from, end=False), "to": _rfc3339_day(date_to, end=True)},
        headers=headers,
        timeout=60,
    )
    info["generate_status"] = gen.status_code
    if gen.status_code != 200:
        info["error"] = (gen.text or "")[:500]
        return info
    uuid = (gen.json() or {}).get("UUID") or (gen.json() or {}).get("uuid")
    info["uuid"] = uuid
    if not uuid:
        info["error"] = "no UUID in generate response"
        info["gen_json_keys"] = list((gen.json() or {}).keys())
        return info

    # poll
    info["stage"] = "poll"
    report_headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    wait_seconds = 0
    max_wait_seconds = 10 * 60  # 10 минут
    last = None

    while wait_seconds < max_wait_seconds:
        rep = requests.get(
            f"{BASE}/api/client/statistics/report/json",
            params={"UUID": str(uuid), "uuid": str(uuid)},
            headers=report_headers,
            timeout=60,
        )
        last = rep.status_code

        if rep.status_code == 200:
            try:
                data = rep.json()
            except Exception:
                info["error"] = f"report not json, head={(rep.text or '')[:200]!r}"
                return info
            rows = data.get("rows") or []
            info["rows"] = len(rows)
            if rows:
                info["row_keys"] = list(rows[0].keys())[:40]
                df = pd.DataFrame(rows)
                if "MoneySpentFromCPC" in df.columns:
                    info["sum_MoneySpentFromCPC"] = float(df["MoneySpentFromCPC"].apply(_parse_ru_money).sum())
                if "MoneySpent" in df.columns:
                    info["sum_MoneySpent"] = float(df["MoneySpent"].apply(_parse_ru_money).sum())
            info["ok"] = True
            info["stage"] = "done"
            return info

        # 404/409/425 = отчёт ещё готовится
        if rep.status_code in (404, 409, 425):
            time.sleep(3)
            wait_seconds += 3
            continue

        info["error"] = (rep.text or "")[:500]
        info["report_status"] = rep.status_code
        return info

    info["report_status"] = last
    info["error"] = "report timeout"
    return info

def allocate_acquiring_cost_by_posting(df_ops: pd.DataFrame) -> pd.DataFrame:
    """Эквайринг по SKU за период (как в ЛК).
    Ищем finance-операции с operation_type_name/type_name = 'Оплата эквайринга' (или содержит 'эквайринг').
    Считаем NET по amount (учитываем сторно/плюсовые строки): расход = max(-sum(amount), 0).
    Если строка без SKU — распределяем внутри posting_number по SKU пропорционально accruals_for_sale (если есть), иначе поровну.
    Возвращает df с колонкой acquiring_cost на строках с SKU.
    """
    if df_ops is None or df_ops.empty:
        return df_ops

    df = df_ops.copy()
    if "acquiring_cost" not in df.columns:
        df["acquiring_cost"] = 0.0

    tn_col = "operation_type_name" if "operation_type_name" in df.columns else ("type_name" if "type_name" in df.columns else None)
    if tn_col is None or "amount" not in df.columns:
        return df

    tn = df[tn_col].astype(str).str.lower()
    mask_acq = tn.str.contains("оплата эквайринга", na=False) | tn.str.contains("эквайринг", na=False) | tn.str.contains("acquiring", na=False)
    if not mask_acq.any():
        return df

    amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # Если нет posting_number — только прямые строки с SKU
    if "posting_number" not in df.columns:
        direct = mask_acq & df["sku"].notna()
        if direct.any():
            tmp = df.loc[direct, ["sku"]].copy()
            tmp["amount"] = amount[direct].values
            net = tmp.groupby("sku")["amount"].sum()
            # добавим на первую строку каждого SKU
            for sku, amt_sum in net.items():
                idx = df.index[df["sku"] == sku]
                if len(idx) > 0:
                    df.loc[idx[0], "acquiring_cost"] += float(max(-amt_sum, 0.0))
        return df

    # 1) прямые строки с SKU: NET по (posting_number, sku)
    direct = mask_acq & df["sku"].notna() & df["posting_number"].astype(str).ne("")
    if direct.any():
        tmp = df.loc[direct, ["posting_number", "sku"]].copy()
        tmp["amount"] = amount[direct].values
        net = tmp.groupby(["posting_number", "sku"], as_index=False)["amount"].sum()
        net["acq_cost"] = net["amount"].apply(lambda x: float(max(-x, 0.0)))

        key = df.loc[df["sku"].notna() & df["posting_number"].astype(str).ne(""), ["posting_number", "sku"]].copy()
        key["idx"] = key.index.values
        net2 = net.merge(key, on=["posting_number", "sku"], how="left").dropna(subset=["idx"])
        if not net2.empty:
            df.loc[net2["idx"].astype(int).values, "acquiring_cost"] += net2["acq_cost"].astype(float).values

    # 2) строки без SKU: NET по posting_number и распределение
    und = mask_acq & df["sku"].isna() & df["posting_number"].astype(str).ne("")
    if not und.any():
        return df

    acq_post = df.loc[und, ["posting_number"]].copy()
    acq_post["amount"] = amount[und].values
    acq_sum = acq_post.groupby("posting_number", as_index=False)["amount"].sum()
    acq_sum["acq_cost"] = acq_sum["amount"].apply(lambda x: float(max(-x, 0.0)))
    acq_sum = acq_sum[acq_sum["acq_cost"] > 0]
    if acq_sum.empty:
        return df

    base = df.loc[df["sku"].notna() & df["posting_number"].astype(str).ne(""), ["posting_number", "sku"]].copy()
    if base.empty:
        return df

    if "accruals_for_sale" in df.columns:
        w = pd.to_numeric(df.loc[base.index, "accruals_for_sale"], errors="coerce").fillna(0.0).abs()
        base["w"] = w.values
        base.loc[base["w"] <= 0, "w"] = 1.0
    else:
        base["w"] = 1.0

    base["w_sum"] = base.groupby("posting_number")["w"].transform("sum")
    base["share"] = base["w"] / base["w_sum"]

    alloc = base.merge(acq_sum[["posting_number", "acq_cost"]], on="posting_number", how="inner")
    if alloc.empty:
        return df
    alloc["acq_alloc"] = alloc["acq_cost"] * alloc["share"]

    key = df.loc[df["sku"].notna() & df["posting_number"].astype(str).ne(""), ["posting_number", "sku"]].copy()
    key["idx"] = key.index.values
    alloc2 = alloc.merge(key, on=["posting_number", "sku"], how="left").dropna(subset=["idx"])
    if alloc2.empty:
        return df

    df.loc[alloc2["idx"].astype(int).values, "acquiring_cost"] += alloc2["acq_alloc"].astype(float).values
    return df

def allocate_acquiring_amount_by_posting(df_ops: pd.DataFrame) -> pd.DataFrame:
    """Распределяет acquiring_amount (amount по операциям эквайринга) по SKU на уровне posting_number.

    ВАЖНО: распределяем именно amount (может быть + и -), чтобы возвраты/компенсации уменьшали эквайринг (как в ЛК).
    Возвращает df_ops с колонкой acquiring_amount_alloc (amount, распределенный на SKU-строки).
    """
    if df_ops is None or df_ops.empty:
        return df_ops
    df = df_ops.copy()

    if "acquiring_amount_alloc" not in df.columns:
        df["acquiring_amount_alloc"] = 0.0

    # определяем, какие строки являются эквайрингом
    tn_col = "operation_type_name" if "operation_type_name" in df.columns else ("type_name" if "type_name" in df.columns else None)
    if tn_col is None:
        return df
    tn = df[tn_col].astype(str).str.lower()
    mask_acq = tn.str.contains("эквайринг", na=False) | tn.str.contains("acquiring", na=False)
    if not mask_acq.any():
        return df

    # берём специализированную колонку, если есть
    if "acquiring_amount" in df.columns:
        amt = pd.to_numeric(df["acquiring_amount"], errors="coerce").fillna(0.0)
    else:
        amt = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0.0)

    # 1) прямые строки с SKU
    direct = mask_acq & df["sku"].notna()
    df.loc[direct, "acquiring_amount_alloc"] += amt[direct].astype(float)

    # 2) строки без SKU — распределяем внутри posting_number
    if "posting_number" not in df.columns:
        return df

    und = mask_acq & df["sku"].isna() & df["posting_number"].astype(str).ne("")
    if not und.any():
        return df

    base = df.loc[df["sku"].notna() & df["posting_number"].astype(str).ne(""), ["posting_number", "sku"]].copy()
    if base.empty:
        return df

    # веса: accruals_for_sale (если есть), иначе 1
    if "accruals_for_sale" in df.columns:
        w = pd.to_numeric(
            df.loc[df["sku"].notna() & df["posting_number"].astype(str).ne(""), "accruals_for_sale"],
            errors="coerce"
        ).fillna(0.0).abs()
        base["w"] = w.values
        base.loc[base["w"] <= 0, "w"] = 1.0
    else:
        base["w"] = 1.0

    base["w_sum"] = base.groupby("posting_number")["w"].transform("sum")
    base["share"] = base["w"] / base["w_sum"]

    acq_by_post = df.loc[und, ["posting_number"]].copy()
    acq_by_post["acq_amt"] = amt[und].values
    acq_sum = acq_by_post.groupby("posting_number", as_index=False)["acq_amt"].sum()

    alloc = base.merge(acq_sum, on="posting_number", how="inner")
    if alloc.empty:
        return df
    alloc["acq_alloc"] = alloc["acq_amt"] * alloc["share"]

    key = df.loc[df["sku"].notna() & df["posting_number"].astype(str).ne(""), ["posting_number", "sku"]].copy()
    key["idx"] = key.index.values
    alloc2 = alloc.merge(key, on=["posting_number", "sku"], how="left").dropna(subset=["idx"])
    if alloc2.empty:
        return df

    idx = alloc2["idx"].astype(int).values
    df.loc[idx, "acquiring_amount_alloc"] += alloc2["acq_alloc"].astype(float).values

    return df

from datetime import date, timedelta, datetime
from dotenv import load_dotenv
import sys
from pathlib import Path

# ================== AUTH ==================
APP_PASSWORD = os.getenv("APP_PASSWORD")

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.markdown(
        """
        <style>
        .auth-banner {
            max-width: 520px;
            margin: 60px auto 18px;
            padding: 16px 22px;
            border-radius: 12px;

            background: var(--secondary-background-color);
            color: var(--text-color);

            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            text-align: center;
            font-weight: 700;
            font-size: 20px;
        }

        /* СТИЛИЗУЕМ САМ st.form — ЭТО ВАЖНО */
        div[data-testid="stForm"] {
            max-width: 520px;
            margin: 0 auto 80px;
            padding: 26px 26px 18px;
            border-radius: 12px;

            background: var(--background-color);
            border: 1px solid rgba(49, 51, 63, 0.18);
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="auth-banner">Оцифровка по Ozon</div>',
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        st.markdown("## 🔐 Вход в приложение")

        pwd = st.text_input(
            "Пароль",
            type="password",
            placeholder="Введите пароль",
        )

        submitted = st.form_submit_button("Войти")

        if submitted:
            if pwd == APP_PASSWORD:
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("Неверный пароль")

    st.stop()


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from ozon_client import OzonSellerClient, last_closed_month, OzonAPIError

# ================== FRIENDLY API ERROR UI ==================

def _humanize_ozon_error(exc: Exception) -> tuple[str, str]:
    """Возвращает (заголовок, детали) для отображения в UI."""
    if isinstance(exc, OzonAPIError):
        sc = exc.status_code
        # типовые причины
        if sc in (401, 403):
            title = "Ozon API: нет доступа (401/403)"
            details = (
                f"Запрос: {exc.path}\n"
                "Проверь Client-Id / Api-Key (Streamlit secrets) и доступы ключа.\n\n"
                f"Ответ: {exc.body}"
            )
            return title, details
        if sc == 429:
            title = "Ozon API: лимит запросов (429)"
            details = (
                f"Запрос: {exc.path}\n"
                "Ozon вернул ограничение по частоте. Подожди 30–60 секунд и нажми «Повторить».\n\n"
                f"Ответ: {exc.body}"
            )
            return title, details
        if sc >= 500:
            title = "Ozon API: временная ошибка сервера (5xx)"
            details = f"Запрос: {exc.path}\n\nОтвет: {exc.body}"
            return title, details
        return f"Ozon API error ({sc})", f"Запрос: {exc.path}\n\nОтвет: {exc.body}"

    # прочие ошибки сети/таймауты
    if isinstance(exc, requests.exceptions.Timeout):
        return "Ozon API: таймаут", str(exc)
    if isinstance(exc, requests.exceptions.RequestException):
        return "Ozon API: ошибка сети", str(exc)
    return "Ошибка", str(exc)


def _block_with_retry(title: str, details: str, cache_clear_fn=None):
    """Показывает сообщение и останавливает приложение без красного падения."""
    st.error(title)
    with st.expander("Детали", expanded=False):
        st.code(details)

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Повторить", use_container_width=True):
            try:
                if cache_clear_fn is not None:
                    cache_clear_fn()
                else:
                    st.cache_data.clear()
            finally:
                st.rerun()
    with c2:
        st.caption("Если ошибка повторяется — проверь ключи Ozon API в secrets и лимиты API.")

    st.stop()


# ================== CONFIG ==================
st.set_page_config(
    layout="wide",
    page_title="Оцифровка Ozon",
    initial_sidebar_state="collapsed",
)
import sys
from pathlib import Path

def resource_path(rel: str) -> str:
    # при запуске в exe: файлы лежат в _MEIPASS
    if hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS) / rel)
    # при обычном запуске: рядом с app.py
    return str(Path(__file__).resolve().parent / rel)

def _get_setting(name: str, default: str = "") -> str:
    """Берём значение из:
    1) переменных окружения
    2) Streamlit secrets (для Streamlit Cloud)
    3) default
    """
    v = os.getenv(name)
    if v is not None and str(v).strip() != "":
        return str(v).strip()
    try:
        # st.secrets может отсутствовать локально/в exe
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets.get(name)).strip()
    except Exception:
        pass
    return default

# Локальная разработка/EXE: можно держать .env рядом, но в облаке его не будет.
try:
    load_dotenv(resource_path(".env"), override=False)
except Exception:
    pass


# --- Ozon Seller API ---
client_id = _get_setting("OZON_CLIENT_ID", "")
api_key = _get_setting("OZON_API_KEY", "")
if not client_id or not api_key:
    st.error("Нет OZON_CLIENT_ID / OZON_API_KEY (переменные окружения или Streamlit secrets).")
    st.stop()

client = OzonSellerClient(client_id, api_key)

# --- Supabase (опционально, для постоянного хранения COGS/OPEX в облаке) ---
SUPABASE_URL = _get_setting("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = _get_setting("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_SCHEMA = _get_setting("SUPABASE_SCHEMA", "public") or "public"
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)

def _sb_headers() -> dict:
    h = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    # если используешь не public схему — Supabase REST требует указать профиль
    if SUPABASE_SCHEMA and SUPABASE_SCHEMA != "public":
        h["Accept-Profile"] = SUPABASE_SCHEMA
        h["Content-Profile"] = SUPABASE_SCHEMA
    return h

def _sb_url(table: str) -> str:
    base = SUPABASE_URL.rstrip("/")
    return f"{base}/rest/v1/{table}"

def _sb_fetch(table: str, select: str = "*", limit: int = 10000) -> pd.DataFrame:
    # простая загрузка (для маленьких таблиц)
    params = {"select": select, "limit": str(limit)}
    r = requests.get(_sb_url(table), headers=_sb_headers(), params=params, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase fetch {table}: {r.status_code} {r.text}")
    data = r.json() if r.text else []
    return pd.DataFrame(data)

def _sb_replace_all(table: str, rows: list[dict], delete_filter: str) -> None:
    # 1) очищаем таблицу по фильтру
    r_del = requests.delete(_sb_url(table) + f"?{delete_filter}", headers=_sb_headers(), timeout=30)
    if r_del.status_code >= 300:
        raise RuntimeError(f"Supabase delete {table}: {r_del.status_code} {r_del.text}")
    # 2) вставляем новые строки
    if rows:
        r_ins = requests.post(
            _sb_url(table),
            headers={**_sb_headers(), "Prefer": "return=minimal"},
            data=json.dumps(rows, ensure_ascii=False),
            timeout=30,
        )
        if r_ins.status_code >= 300:
            raise RuntimeError(f"Supabase insert {table}: {r_ins.status_code} {r_ins.text}")


# ================== PERF API (ADS) ==================
class OzonPerfClient:
    """
    Ozon Performance API (ADS) — рабочий вариант под реальный ответ CSV.

    - campaigns: GET /api/client/campaign (JSON)
    - daily:     GET /api/client/statistics/daily (CSV, sep=';', decimal=',')

    ВАЖНО:
    - campaignIds нельзя слать строкой "1,2,3"
      Нужно campaignIds=1&campaignIds=2 (requests сделает это если передать список)
    """

    def __init__(self, client_id: str, client_secret: str, base_url: str = "https://api-performance.ozon.ru"):
        self.client_id = str(client_id).strip()
        self.client_secret = str(client_secret).strip()
        self.base_url = base_url.rstrip("/")
        self._token = None
        self._token_ts = 0.0
        self._token_ttl = 50 * 60  # 50 мин
        self._last_debug = {}

    # ----------------- token -----------------
    def _request_json(self, method: str, path: str, *, headers=None, params=None, json_body=None, timeout=30) -> dict:
        url = self.base_url + path
        h = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ozon-ads-dashboard/1.0",
        }
        if headers:
            h.update(headers)

        r = requests.request(method=method.upper(), url=url, headers=h, params=params, json=json_body, timeout=timeout)

        if r.status_code < 200 or r.status_code >= 300:
            raise RuntimeError(f"{r.status_code} {path}: {r.text}")

        try:
            return r.json()
        except Exception:
            # иногда JSON без content-type
            raise RuntimeError(f"Ожидал JSON, но пришло не JSON: {r.text[:1000]}")

    def _get_token_uncached(self) -> str:
        data = self._request_json(
            "POST",
            "/api/client/token",
            json_body={
                "client_id": int(self.client_id) if self.client_id.isdigit() else self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
            },
        )
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"Не получил access_token. Ответ: {data}")
        return token

    def get_token(self) -> str:
        now = time.time()
        if self._token and (now - self._token_ts) < self._token_ttl:
            return self._token
        token = self._get_token_uncached()
        self._token = token
        self._token_ts = now
        return token

    # ----------------- helpers -----------------
    @staticmethod
    def _safe_date10(s: str):
        try:
            s10 = str(s)[:10]
            return datetime.strptime(s10, "%Y-%m-%d").date()
        except Exception:
            return None

    @staticmethod
    def _iter_chunks(d1: date, d2: date, max_days: int = 62):
        cur = d1
        while cur <= d2:
            chunk_to = min(cur + timedelta(days=max_days - 1), d2)
            yield cur, chunk_to
            cur = chunk_to + timedelta(days=1)

    def last_debug(self) -> dict:
        return self._last_debug or {}
    @staticmethod
    def _humanize_error(msg: str) -> str:
        m = (msg or "").strip()

        # Самая частая: превышен лимит периода
        if "max statistics period" in m and "62" in m:
            return (
                "Performance API: выбран слишком длинный период.\n\n"
                "⚠️ Ozon ограничивает выгрузку статистики рекламы максимум **62 дня** за один запрос.\n"
                "Сократи диапазон дат (например, до 30–60 дней) — и показатели рекламы появятся.\n\n"
                "Подсказка: продажи/прибыль по операциям считаются отдельно и будут работать и на большом периоде."
            )

        # На будущее — если токен/доступ
        if "401" in m or "unauthorized" in m.lower():
            return (
                "Performance API: нет доступа (401).\n\n"
                "Проверь PERF_CLIENT_ID / PERF_CLIENT_SECRET в .env и права приложения."
            )

        return f"Performance API недоступен: {m}"

    # ----------------- campaigns -----------------
    def list_campaigns(self) -> list[dict]:
        token = self.get_token()
        headers = {"Authorization": f"Bearer {token}"}

        data = self._request_json("GET", "/api/client/campaign", headers=headers)

        if isinstance(data, list):
            return data

        for k in ("list", "result", "campaigns", "items", "data", "rows"):
            if isinstance(data, dict) and isinstance(data.get(k), list):
                return data[k]

        return []

    def campaign_ids_overlapping(self, date_from_str: str, date_to_str: str, limit: int = 300) -> list[int]:
        d_from = self._safe_date10(date_from_str)
        d_to = self._safe_date10(date_to_str)
        camps = self.list_campaigns()

        ids: list[int] = []
        for c in camps:
            cid = c.get("id")
            if cid is None:
                continue

            c_from = self._safe_date10(c.get("fromDate") or c.get("dateFrom") or c.get("date_from") or "")
            c_to = self._safe_date10(c.get("toDate") or c.get("dateTo") or c.get("date_to") or "")

            ok = True
            if d_from and d_to and (c_from or c_to):
                left = c_from or d_from
                right = c_to or d_to
                ok = not (right < d_from or left > d_to)

            if ok:
                try:
                    ids.append(int(cid))
                except Exception:
                    pass

            if len(ids) >= limit:
                break

        return ids

    # ----------------- daily CSV -----------------
    @staticmethod
    def _parse_csv_semicolon(text: str) -> pd.DataFrame:
        cleaned = (text or "").lstrip("\ufeff").strip()
        if not cleaned:
            return pd.DataFrame()

        df = pd.read_csv(
            io.StringIO(cleaned),
            sep=";",
            dtype=str,
            keep_default_na=False,
        )

        # Приводим числа: "1 234,56" -> 1234.56
        for col in df.columns:
            s = df[col].astype(str)
            if s.str.contains(r"\d", regex=True).any():
                s2 = (
                    s.str.replace("\u00a0", "", regex=False)
                     .str.replace(" ", "", regex=False)
                     .str.replace(",", ".", regex=False)
                )
                df[col] = pd.to_numeric(s2, errors="ignore")

        return df

    @staticmethod
    def _norm_col(s: str) -> str:
        return str(s).strip().lower().replace("\u00a0", " ").replace("  ", " ")

    @staticmethod
    def _pick_num(df: pd.DataFrame, candidates: list[str]) -> float:
        cols_l = {OzonPerfClient._norm_col(c): c for c in df.columns}
        for name in candidates:
            key = OzonPerfClient._norm_col(name)
            if key in cols_l:
                v = pd.to_numeric(df[cols_l[key]], errors="coerce").fillna(0).sum()
                try:
                    return float(v)
                except Exception:
                    return 0.0
        return 0.0

    @staticmethod
    def _pick_int(df: pd.DataFrame, candidates: list[str]) -> int:
        return int(round(OzonPerfClient._pick_num(df, candidates)))

    def fetch_statistics_daily(self, date_from_str: str, date_to_str: str, campaign_ids: list[int]) -> tuple[dict, dict]:
        token = self.get_token()
        headers = {"Authorization": f"Bearer {token}"}

        # params: если ids пустые — пробуем без campaignIds (у тебя так реально работало)
        params: dict = {"dateFrom": date_from_str, "dateTo": date_to_str}

        ids = [int(x) for x in (campaign_ids or []) if str(x).isdigit()]
        if ids:
            # КЛЮЧЕВОЕ: список -> campaignIds=1&campaignIds=2...
            params["campaignIds"] = ids[:200]

        url = self.base_url + "/api/client/statistics/daily"

        r = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "*/*",
                "User-Agent": "ozon-ads-dashboard/1.0",
            },
            params=params,
            timeout=20
        )

        meta = {
            "ok_variant": {"method": "GET", "path": "/api/client/statistics/daily", "params": params},
            "status_code": r.status_code,
            "content_type": r.headers.get("Content-Type", ""),
            "url": r.url,
        }

        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} /api/client/statistics/daily: {r.text}")

        df = self._parse_csv_semicolon(r.text)

        # --- МЕТРИКИ ИЗ ТВОЕГО CSV (русские колонки) ---
        spent = self._pick_num(df, ["Расход, ₽", "Расход", "Затраты", "Cost", "Spent"])
        revenue = self._pick_num(df, ["Заказы, ₽", "Заказы ₽", "Выручка", "Оборот", "Revenue"])
        orders = self._pick_int(df, ["Заказы, шт.", "Заказы, шт", "Заказы шт", "Orders", "Orders count"])
        clicks = self._pick_num(df, ["Клики", "Clicks"])
        shows = self._pick_num(df, ["Показы", "Impressions", "Shows"])

        drr = (spent / revenue * 100.0) if revenue else 0.0
        cpc = (spent / clicks) if clicks else 0.0
        ctr = (clicks / shows * 100.0) if shows else 0.0

        dbg = {
            **meta,
            "rows_count": int(len(df)),
            "columns": [str(c) for c in df.columns],
            "head": df.head(5).to_dict(orient="records") if not df.empty else [],
        }

        metrics = {"spent": spent, "revenue": revenue, "orders": orders, "clicks": float(clicks), "shows": float(shows), "drr": drr, "cpc": cpc, "ctr": ctr}
        return metrics, dbg

    def fetch_shop_summary(self, date_from_str: str, date_to_str: str) -> tuple[dict, str, dict]:
        base = {"spent": 0.0, "revenue": 0.0, "orders": 0, "drr": 0.0, "cpc": 0.0, "ctr": 0.0}

        try:
            d1 = self._safe_date10(date_from_str)
            d2 = self._safe_date10(date_to_str)
            if not d1 or not d2:
                return base, "Performance API: некорректные даты периода.", {"error": "bad dates"}

            # ids как раньше
            ids = self.campaign_ids_overlapping(date_from_str, date_to_str, limit=300)

            totals = {
                "spent": 0.0,
                "revenue": 0.0,
                "orders": 0,
                "clicks": 0.0,
                "shows": 0.0,
            }

            chunks_info = []
            errors = []

            for a, b in self._iter_chunks(d1, d2, max_days=62):
                a_s = a.strftime("%Y-%m-%d")
                b_s = b.strftime("%Y-%m-%d")

                try:
                    m, dbg = self.fetch_statistics_daily(a_s, b_s, ids)

                    # суммируем "сырые" итоги
                    totals["spent"] += float(m.get("spent", 0.0))
                    totals["revenue"] += float(m.get("revenue", 0.0))
                    totals["orders"] += int(m.get("orders", 0) or 0)

                    # clicks/shows берем из dbg? у нас они в metrics уже посчитаны только как cpc/ctr,
                    # поэтому лучше пересчитать внутри fetch_statistics_daily и вернуть clicks/shows тоже.
                    # Но чтобы не ломать, извлечём из df через dbg нельзя.
                    # => Решение: ДОБАВИ в metrics ниже clicks/shows. (см. мини-правку №2.1)
                    totals["clicks"] += float(m.get("clicks", 0.0))
                    totals["shows"] += float(m.get("shows", 0.0))

                    chunks_info.append({"from": a_s, "to": b_s, "rows": int(dbg.get("rows_count", 0))})
                except Exception as e:
                    errors.append({"from": a_s, "to": b_s, "error": str(e)})

            # если вообще ничего не смогли получить
            if totals["spent"] == 0 and totals["revenue"] == 0 and totals["orders"] == 0 and not chunks_info:
                note = self._humanize_error(errors[0]["error"]) if errors else "Performance API недоступен."
                dbg = {"error": "all chunks failed", "errors": errors, "chunks": chunks_info}
                self._last_debug = dbg
                return base, note, dbg

            # пересчёт метрик от итогов
            spent = totals["spent"]
            revenue = totals["revenue"]
            orders = totals["orders"]
            clicks = totals["clicks"]
            shows = totals["shows"]

            drr = (spent / revenue * 100.0) if revenue else 0.0
            cpc = (spent / clicks) if clicks else 0.0
            ctr = (clicks / shows * 100.0) if shows else 0.0

            metrics = {"spent": spent, "revenue": revenue, "orders": orders, "drr": drr, "cpc": cpc, "ctr": ctr}

            note = ""
            if (d2 - d1).days + 1 > 62:
                note = f"Performance: период больше 62 дней — посчитано чанками ({len(chunks_info)} запросов)."
            if errors:
                note = (note + "\n" if note else "") + f"⚠️ Не удалось получить {len(errors)} чанков — итог может быть неполным."

            dbg = {
                "campaign_ids_count": len(ids),
                "campaign_ids_sample": ids[:50],
                "chunks": chunks_info,
                "errors": errors,
                "dateFrom": date_from_str,
                "dateTo": date_to_str,
            }
            self._last_debug = dbg

            return metrics, note, dbg

        except Exception as e:
            msg = str(e)
            note = self._humanize_error(msg)
            dbg = {"error": msg}
            self._last_debug = dbg
            return base, note, dbg


perf_id = _get_setting("PERF_CLIENT_ID", "")
perf_secret = _get_setting("PERF_CLIENT_SECRET", "")
perf_client: OzonPerfClient | None = None
if perf_id and perf_secret:
    try:
        perf_client = OzonPerfClient(perf_id, perf_secret)
    except Exception:
        perf_client = None

st.title("Оцифровка Ozon")

# ================== GLOBAL STYLES ==================
st.markdown(
    """
<style>
div[data-testid="stDateInput"] small { display: none !important; }

.ts-tile{
  border-radius: 14px;
  padding: 14px 16px 12px 16px;
  border: 2px solid rgba(0,0,0,0.2);
  background: rgba(255,255,255,0.06);
  min-height: 108px;
}
.ts-title{ font-size: 13px; opacity: 0.9; margin-bottom: 6px; }
.ts-value{ font-size: 26px; font-weight: 800; line-height: 1.05; }
.ts-delta{ margin-top: 6px; font-size: 13px; font-weight: 700; opacity: 0.95; }

.ts-good{ background: rgba(34,197,94,0.18); border-color: rgba(34,197,94,0.60); }
.ts-good .ts-delta{ color: rgba(34,197,94,1); }

.ts-bad{ background: rgba(239,68,68,0.18); border-color: rgba(239,68,68,0.60); }
.ts-bad .ts-delta{ color: rgba(239,68,68,1); }

.ts-neutral{ background: rgba(148,163,184,0.12); border-color: rgba(148,163,184,0.40); }
.ts-neutral .ts-delta{ color: rgba(148,163,184,1); }

@media (max-width: 900px){
  .ts-value{ font-size: 22px; }
  .ts-tile{ min-height: 98px; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* Делает текст/placeholder в input визуально по центру (за счёт padding/line-height) */
div[data-testid="stTextInput"] input {
  padding-top: 0.55rem !important;
  padding-bottom: 0.55rem !important;
  line-height: 1.2 !important;
}

/* Чуть выравниваем selectbox, чтобы он выглядел как остальные поля */
div[data-testid="stSelectbox"] div[role="combobox"] {
  padding-top: 0.45rem !important;
  padding-bottom: 0.45rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ===== OPEX: выравниваем инпуты в строке "Добавить расход" ===== */

/* общий контейнер: выравниваем содержимое колонок по низу */
div[data-testid="stHorizontalBlock"]{
  align-items: flex-end;
}

/* делаем одинаковую высоту полей ввода (date/text/number) */
div[data-testid="stDateInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input{
  height: 44px !important;
  line-height: 44px !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}

/* st_tags (baseweb tag input) — чтобы был той же высоты */
div[data-baseweb="tag-input"]{
  min-height: 44px !important;
  align-items: center !important;
}
div[data-baseweb="tag-input"] > div{
  min-height: 44px !important;
  align-items: center !important;
}

/* кнопка добавления — в ту же высоту */
button[kind="secondary"], button[kind="primary"]{
  height: 44px !important;
}
</style>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
def money(x) -> str:
    try:
        return f"{float(x):,.0f} ₽".replace(",", " ")
    except Exception:
        return "0 ₽"

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def _to_int(x):
    try:
        return int(float(x))
    except Exception:
        return 0

def to_cost(x: float) -> float:
    x = float(x or 0.0)
    return abs(x)

DATA_DIR = "data"
COGS_PATH = os.path.join(DATA_DIR, "cogs.csv")
OPEX_PATH = os.path.join(DATA_DIR, "opex.csv")
OPEX_TYPES_PATH = os.path.join(DATA_DIR, "opex_types.json")

def load_opex_types() -> list[str]:
    ensure_data_dir()
    if os.path.exists(OPEX_TYPES_PATH):
        try:
            with open(OPEX_TYPES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                out = []
                for x in data:
                    s = str(x).strip()
                    if s:
                        out.append(s)
                return sorted(set(out), key=lambda s: s.lower())
        except Exception:
            pass
    return []

def save_opex_types(types: list[str]):
    ensure_data_dir()
    out = []
    for x in (types or []):
        s = str(x).strip()
        if s:
            out.append(s)
    out = sorted(set(out), key=lambda s: s.lower())
    with open(OPEX_TYPES_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

# ================== COGS ==================
def normalize_cogs_upload(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["article", "sku", "cogs"])

    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]

    sku_candidates = [c for c in df2.columns if c.lower() in ("sku", "item.sku", "озон sku", "ozon sku")]
    if not sku_candidates:
        sku_candidates = [c for c in df2.columns if "sku" in c.lower()]
    sku_col = sku_candidates[0] if sku_candidates else None

    cogs_candidates = [c for c in df2.columns if c.lower() in ("себестоимость", "себес", "cogs", "cost", "cost_price", "costprice")]
    if not cogs_candidates:
        cogs_candidates = [c for c in df2.columns if "себ" in c.lower() or "cost" in c.lower()]
    cogs_col = cogs_candidates[0] if cogs_candidates else None

    art_candidates = [c for c in df2.columns if "артикул" in c.lower() or "offer" in c.lower() or "article" in c.lower()]
    art_col = art_candidates[0] if art_candidates else None

    if sku_col is None or cogs_col is None:
        if df2.shape[1] >= 3:
            art_col = art_col or df2.columns[0]
            sku_col = sku_col or df2.columns[1]
            cogs_col = cogs_col or df2.columns[2]
        else:
            return pd.DataFrame(columns=["article", "sku", "cogs"])

    out = pd.DataFrame({
        "article": df2[art_col] if art_col in df2.columns else "",
        "sku": df2[sku_col],
        "cogs": df2[cogs_col],
    })

    out["sku"] = (
    out["sku"]
    .astype(str)
    .str.replace(r"[^\d]", "", regex=True)   # убираем запятые/пробелы/мусор
    )
    out["sku"] = pd.to_numeric(out["sku"], errors="coerce").astype("Int64")
    out["cogs"] = pd.to_numeric(out["cogs"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["sku"]).copy()
    out["sku"] = out["sku"].astype(int)
    out["article"] = out["article"].astype(str).fillna("")
    out = out[["article", "sku", "cogs"]].drop_duplicates(subset=["sku"], keep="last").sort_values("sku")
    return out

def load_cogs() -> pd.DataFrame:
    ensure_data_dir()
    # 1) Supabase приоритетнее
    if USE_SUPABASE:
        try:
            df = _sb_fetch("cogs", select="sku,article,cogs", limit=100000)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                if "article" not in df.columns:
                    df["article"] = ""
                df["sku"] = pd.to_numeric(df["sku"], errors="coerce").astype("Int64")
                df["cogs"] = pd.to_numeric(df["cogs"], errors="coerce").fillna(0.0)
                df = df.dropna(subset=["sku"]).copy()
                df["sku"] = df["sku"].astype(int)
                df["article"] = df["article"].fillna("").astype(str)
                df = df[["article","sku","cogs"]].drop_duplicates(subset=["sku"], keep="last").sort_values("sku")
                return df
        except Exception:
            pass
    if os.path.exists(COGS_PATH):
        try:
            df = pd.read_csv(COGS_PATH, encoding="utf-8-sig")
            df.columns = [str(c).strip() for c in df.columns]
            if "sku" not in df.columns or "cogs" not in df.columns:
                df = normalize_cogs_upload(df)
            else:
                if "article" not in df.columns:
                    df["article"] = ""
                df["sku"] = (
                    df["sku"]
                    .astype(str)
                    .str.replace(r"[^\d]", "", regex=True)
                )
                df["sku"] = pd.to_numeric(df["sku"], errors="coerce").astype("Int64")
                df["cogs"] = pd.to_numeric(df["cogs"], errors="coerce").fillna(0.0)
                df = df.dropna(subset=["sku"]).copy()
                df["sku"] = df["sku"].astype(int)
                df["article"] = df["article"].astype(str).fillna("")
                df = df[["article", "sku", "cogs"]].drop_duplicates(subset=["sku"], keep="last").sort_values("sku")
            return df
        except Exception:
            return pd.DataFrame(columns=["article", "sku", "cogs"])
    return pd.DataFrame(columns=["article", "sku", "cogs"])

def save_cogs(df: pd.DataFrame):
    ensure_data_dir()
    df2 = df.copy() if df is not None else pd.DataFrame(columns=["article","sku","cogs"])
    if "article" not in df2.columns:
        df2["article"] = ""
    if "cogs" not in df2.columns:
        df2["cogs"] = 0.0

    df2 = df2[["article", "sku", "cogs"]].copy()
    df2["sku"] = pd.to_numeric(df2["sku"], errors="coerce").astype("Int64")
    df2["cogs"] = pd.to_numeric(df2["cogs"], errors="coerce").fillna(0.0)
    df2 = df2.dropna(subset=["sku"]).copy()
    df2["sku"] = df2["sku"].astype(int)
    df2["article"] = df2["article"].astype(str).fillna("")
    df2 = df2.drop_duplicates(subset=["sku"], keep="last").sort_values("sku")

    # 1) Supabase
    if USE_SUPABASE:
        rows = df2[["sku", "article", "cogs"]].copy()
        payload = rows.to_dict(orient="records")
        _sb_replace_all("cogs", payload, "sku=gt.0")

    # 2) Локально (fallback)
    df2.to_csv(COGS_PATH, index=False, encoding="utf-8-sig")


# ================== OPEX (Operational expenses) ==================
def _opex_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "type", "amount"])

def load_opex() -> pd.DataFrame:
    """Загружает OPEX.

    В Streamlit Cloud файловая система может быть эфемерной, поэтому:
    1) если USE_SUPABASE=True — сначала читаем из Supabase таблицу `opex`;
    2) иначе / если не удалось — читаем локальный CSV (fallback).
    """
    ensure_data_dir()

    # 1) Supabase (приоритет)
    if USE_SUPABASE:
        try:
            df = _sb_fetch("opex", select="date,type,amount", limit=100000)
            if df is not None and not df.empty:
                df.columns = [str(c).strip().lower() for c in df.columns]
                # ожидаем: date, type, amount
                if "date" not in df.columns and "дата" in df.columns:
                    df = df.rename(columns={"дата": "date"})
                if "type" not in df.columns:
                    for c in df.columns:
                        if "тип" in c or "category" in c:
                            df = df.rename(columns={c: "type"})
                            break
                if "amount" not in df.columns:
                    for c in df.columns:
                        if "сумм" in c or "amount" in c:
                            df = df.rename(columns={c: "amount"})
                            break
                if {"date","type","amount"}.issubset(set(df.columns)):
                    df = df[["date","type","amount"]].copy()
                    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                    df["type"] = df["type"].fillna("").astype(str)
                    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
                    df = df.dropna(subset=["date"]).sort_values(["date","type"]).reset_index(drop=True)
                    return df
        except Exception:
            # упали на Supabase — идём в локальный fallback
            pass

    # 2) Локальный CSV (fallback)
    if os.path.exists(OPEX_PATH):
        try:
            df = pd.read_csv(OPEX_PATH, encoding="utf-8-sig")
            df.columns = [str(c).strip().lower() for c in df.columns]
            # ожидаем: date, type, amount
            if "date" not in df.columns:
                # поддержка русских названий
                if "дата" in df.columns:
                    df = df.rename(columns={"дата": "date"})
            if "type" not in df.columns:
                for c in df.columns:
                    if "тип" in c or "category" in c:
                        df = df.rename(columns={c: "type"})
                        break
            if "amount" not in df.columns:
                for c in df.columns:
                    if "сумм" in c or "amount" in c:
                        df = df.rename(columns={c: "amount"})
                        break
            if not {"date","type","amount"}.issubset(set(df.columns)):
                return _opex_empty()
            df = df[["date","type","amount"]].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            df["type"] = df["type"].fillna("").astype(str)
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
            df = df.dropna(subset=["date"]).sort_values(["date","type"]).reset_index(drop=True)
            return df
        except Exception:
            return _opex_empty()

    return _opex_empty()

def save_opex(df: pd.DataFrame):
    ensure_data_dir()

    df2 = df.copy() if df is not None else _opex_empty()
    if df2.empty:
        df2 = _opex_empty()

    df2 = df2[["date","type","amount"]].copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date
    df2["type"] = df2["type"].fillna("").astype(str)
    df2["amount"] = pd.to_numeric(df2["amount"], errors="coerce").fillna(0.0)
    df2 = df2.dropna(subset=["date"]).sort_values(["date","type"]).reset_index(drop=True)

    # 1) Supabase
    if USE_SUPABASE:
        rows = df2.copy()
        rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        payload = rows.to_dict(orient="records")
        _sb_replace_all("opex", payload, "date=gte.1900-01-01")

    # 2) Локально
    out = df2.copy()
    out["date"] = out["date"].apply(lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d))
    out.to_csv(OPEX_PATH, index=False, encoding="utf-8-sig")

def opex_sum_period(df_opex: pd.DataFrame, d_from: date, d_to: date) -> float:
    if df_opex is None or df_opex.empty:
        return 0.0
    mask = (df_opex["date"] >= d_from) & (df_opex["date"] <= d_to)
    return float(pd.to_numeric(df_opex.loc[mask, "amount"], errors="coerce").fillna(0.0).sum())


# --- Sidebar COGS ---
st.sidebar.header("Себестоимость")
ensure_data_dir()

uploaded = st.sidebar.file_uploader(
    "Загрузить файл себестоимости (Артикул / SKU / Себестоимость)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False
)

# --- ДИАГНОСТИКА SUPABASE (временно) ---
if USE_SUPABASE:
    st.sidebar.write("✅")
else:
    st.sidebar.write("SUPABASE OFF ❌ (нет SUPABASE_URL или SUPABASE_SERVICE_ROLE_KEY)")

cogs_df = load_cogs()
df_opex = load_opex()

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            tmp = pd.read_csv(uploaded, encoding="utf-8-sig")
        else:
            tmp = pd.read_excel(uploaded)

        cogs_df = normalize_cogs_upload(tmp)
        save_cogs(cogs_df)

        st.sidebar.success("Себестоимость загружена и сохранена")
    except Exception as e:
        st.sidebar.error(f"Не смог прочитать файл: {e}")

# ================== OPS -> DF ==================
def _service_bucket(name: str) -> str:
    n = (name or "").lower()

    # ✅ Оплата за заказ (CPA) — ВРЕМЕННО ИСКЛЮЧАЕМ из "Расходы Ozon"
    # (потом выведем отдельно)
    if ("оплата" in n and "заказ" in n) or ("pay" in n and "order" in n) or ("cpa" in n):
        return "ads_order"

    # Баллы за скидки
    if ("балл" in n and "скид" in n) or ("bonus" in n and "discount" in n):
        return "bonus"

    # Программы партнёров
    if ("партн" in n) or ("partner" in n):
        return "partner"

    return "other"


def extract_services_breakdown_from_ops(
    ops: list[dict],
    exclude_names: tuple[str, ...] = ("эквайринг", "acquiring"),
) -> pd.DataFrame:
    """
    DEBUG: вытаскивает все services из сырых операций finance (ops list).
    exclude_names — подстроки (lower), которые нужно исключить из services
    (например, эквайринг, чтобы не было двойного учета).
    """
    rows = []
    for op in (ops or []):
        op_id = op.get("operation_id")
        op_type_name = op.get("operation_type_name", "") or op.get("operation_type", "")
        op_date = op.get("operation_date", "")
        posting = op.get("posting") or {}
        posting_number = posting.get("posting_number", "")
        delivery_schema = posting.get("delivery_schema", "")

        services = op.get("services") or []
        for s in services:
            name = s.get("name") or s.get("service_name") or s.get("title") or ""
            name_l = str(name).lower()

            # ❌ исключаем эквайринг из services (он должен считаться по amount/operation_type_name)
            if exclude_names and any(x in name_l for x in exclude_names):
                continue

            price = _to_float(s.get("price", 0))
            rows.append({
                "operation_id": op_id,
                "operation_date": op_date,
                "type_name": op_type_name,
                "posting_number": posting_number,
                "delivery_schema": delivery_schema,
                "service_name": str(name),
                "price": float(price),
                "cost": float(max(-price, 0.0)),  # расходы как +число
                "is_acquiring": ("эквайринг" in name_l) or ("acquiring" in name_l),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        # удобно сразу нормализовать
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    return df

def ops_to_df(ops: list[dict]) -> pd.DataFrame:
    rows = []
    for op in ops:
        op_id = op.get("operation_id")
        op_group = op.get("type", "")
        op_code = op.get("operation_type", "")
        op_type_name = op.get("operation_type_name", "") or op_code
        op_date = op.get("operation_date", "")

        accruals_total = _to_float(op.get("accruals_for_sale", 0))
        commission_total = _to_float(op.get("sale_commission", 0))
        amount_total = _to_float(op.get("amount", 0))
        
        # Эквайринг как отдельная операция (из amount)
        is_acq_op = ("эквайринг" in str(op_type_name).lower()) or ("acquiring" in str(op_type_name).lower())
        acquiring_total = amount_total if is_acq_op else 0.0


        posting = op.get("posting") or {}
        posting_number = posting.get("posting_number", "")
        delivery_schema = posting.get("delivery_schema", "")

        items = op.get("items") or []
        services = op.get("services") or []        # --- услуги: общий + разрез на "баллы/партнерки/прочее"
        services_total = 0.0
        bonus_sum = 0.0
        partner_sum = 0.0
        
        acquiring_services_sum = 0.0
        ads_order_sum = 0.0

        for s in services:
            sname = (s.get("name") or s.get("service_name") or s.get("title") or "").lower()

            # ❌ эквайринг не должен попадать в services
            if "эквайринг" in sname or "acquiring" in sname:
                continue

            price = _to_float(s.get("price", 0))
            services_total += price
            sname = s.get("name") or s.get("service_name") or s.get("title") or ""
            sname_l = str(sname).lower()

            # Эквайринг берём из services (как в ЛК), но исключаем из services_sum
            if ("эквайринг" in sname_l) or ("acquiring" in sname_l):
                acquiring_services_sum += price
                continue

            b = _service_bucket(str(sname))
            if b == "bonus":
                bonus_sum += price
            elif b == "partner":
                partner_sum += price
            elif b == "ads_order":
                # ВАЖНО: временно исключаем из расходов Ozon (отдельно пока не выводим)
                ads_order_sum += price

        services_other = services_total - bonus_sum - partner_sum - ads_order_sum
        base = {
            "operation_id": op_id,
            "operation_date": op_date,
            "type": op_group,
            "operation_type": op_code,
            "type_name": op_type_name,
            "posting_number": posting_number,
            "delivery_schema": delivery_schema,
        }

        if not items:
            # items пусто — оставляем строку, потом распределим по SKU по posting_number
            rows.append({
                **base,
                "sku": None,
                "name": None,
                "qty": 0.0,
                "accruals_for_sale": accruals_total,
                "sale_commission": commission_total,
                "services_sum": services_other,
                "acquiring_service": acquiring_services_sum,
                "acquiring_amount": acquiring_total,
                "bonus_points": bonus_sum,
                "partner_programs": partner_sum,
                "amount": amount_total,
            })
            continue

        qtys = [max(_to_float(it.get("quantity", 1)), 0.0) for it in items]
        total_qty = sum(qtys) if qtys else 0.0
        if total_qty <= 0:
            total_qty = float(len(items))
            qtys = [1.0] * len(items)

        for it, q in zip(items, qtys):
            w = q / total_qty if total_qty else 0.0
            rows.append({
                **base,
                "sku": it.get("sku"),
                "name": it.get("name"),
                "qty": q,
                "accruals_for_sale": accruals_total * w,
                "sale_commission": commission_total * w,
                "services_sum": services_other * w,
                "acquiring_service": acquiring_services_sum * w,
                "acquiring_amount": acquiring_total * w,
                "bonus_points": bonus_sum * w,
                "partner_programs": partner_sum * w,
                "amount": amount_total * w,
            })

    df = pd.DataFrame(rows)
    for c in ["accruals_for_sale", "sale_commission", "services_sum", "acquiring_service", "bonus_points", "partner_programs", "amount", "qty"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def redistribute_ops_without_items(df_ops: pd.DataFrame) -> pd.DataFrame:
    """
    Распределяет строки, где sku=None (items пустые), по SKU внутри того же posting_number,
    используя веса по qty из строк этого же posting_number, где sku уже известен.
    Суммы сохраняются 1-в-1, просто перестают висеть в "sku=None".
    """
    if df_ops is None or df_ops.empty:
        return df_ops

    df = df_ops.copy()

    need = df["sku"].isna() & df["posting_number"].astype(str).str.strip().ne("")
    if not need.any():
        return df

    has = df["sku"].notna() & df["posting_number"].astype(str).str.strip().ne("")
    if not has.any():
        return df

    # веса по qty внутри posting_number и sku
    wbase = (
        df.loc[has, ["posting_number", "sku", "name", "qty"]]
        .copy()
    )
    wbase["sku"] = pd.to_numeric(wbase["sku"], errors="coerce")
    wbase = wbase.dropna(subset=["sku"])
    if wbase.empty:
        return df

    wbase["qty"] = pd.to_numeric(wbase["qty"], errors="coerce").fillna(0.0).clip(lower=0.0)
    w = (
        wbase.groupby(["posting_number", "sku"], as_index=False)
        .agg(w_qty=("qty", "sum"), name=("name", "first"))
    )
    # нормируем веса внутри posting_number
    w["w_sum"] = w.groupby("posting_number")["w_qty"].transform("sum")
    w = w[w["w_sum"] > 0].copy()
    if w.empty:
        return df
    w["weight"] = w["w_qty"] / w["w_sum"]

    # строки без items
    miss = df.loc[need].copy()
    keep = df.loc[~need].copy()

    out_rows = [keep]

    # разворачиваем miss по sku из w
    for _, r in miss.iterrows():
        pn = str(r.get("posting_number", "")).strip()
        ww = w[w["posting_number"] == pn]
        if ww.empty:
            # не нашли куда распределить — оставляем как было
            out_rows.append(pd.DataFrame([r]))
            continue

        for _, wr in ww.iterrows():
            k = float(wr["weight"])
            newr = r.copy()
            newr["sku"] = int(wr["sku"])
            if pd.isna(newr.get("name")) or str(newr.get("name")).strip() == "":
                newr["name"] = wr.get("name")

            # qty: если это orders/returns — распределяем qty как w_qty
            # (если qty в исходной строке = 0, это нормально — берем w_qty)
            if str(newr.get("type")) in ("orders", "returns"):
                newr["qty"] = float(wr["w_qty"])

            # суммы распределяем по весу
            for c in ["accruals_for_sale", "sale_commission", "services_sum", "acquiring_service", "bonus_points", "partner_programs", "amount"]:
                newr[c] = _to_float(newr.get(c, 0)) * k

            out_rows.append(pd.DataFrame([newr]))

    out = pd.concat(out_rows, ignore_index=True)

    # финальная чистка типов
    for c in ["accruals_for_sale", "sale_commission", "services_sum", "bonus_points", "partner_programs", "amount", "qty"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


# ================== MONTH-SAFE CHUNK LOADER ==================
def month_safe_chunks(d_from: date, d_to: date):
    cur = d_from
    while cur <= d_to:
        if cur.month == 12:
            next_month_start = date(cur.year + 1, 1, 1)
        else:
            next_month_start = date(cur.year, cur.month + 1, 1)
        month_end = next_month_start - timedelta(days=1)
        chunk_to = min(month_end, d_to)
        yield cur, chunk_to
        cur = chunk_to + timedelta(days=1)

@st.cache_data(ttl=600)
def load_ops_range(date_from_str: str, date_to_str: str) -> tuple[list[dict], str, str]:
    d1 = datetime.strptime(date_from_str, "%Y-%m-%d").date()
    d2 = datetime.strptime(date_to_str, "%Y-%m-%d").date()

    ops_all = []
    try:
        for a, b in month_safe_chunks(d1, d2):
            ops_part = client.fetch_finance_transactions(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"))
            ops_all.extend(ops_part)
        return ops_all, "", ""
    except Exception as e:
        title, details = _humanize_ozon_error(e)
        return [], title, details


def _merge_analytics_chunks(chunks: list[list[dict]]) -> pd.DataFrame:
    # Sum units per SKU across chunks
    if not chunks:
        return pd.DataFrame()
    rows = []
    for part in chunks:
        if part:
            rows.extend(part)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty or "sku" not in df.columns:
        return pd.DataFrame()
    for c in ["ordered_units", "delivered_units", "returned_units", "cancelled_units"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    return df.groupby("sku", as_index=False)[["ordered_units","delivered_units","returned_units","cancelled_units"]].sum()


@st.cache_data(ttl=3600)
def load_analytics_range(date_from_str: str, date_to_str: str) -> tuple[pd.DataFrame, str, str]:
    d1 = datetime.strptime(date_from_str, "%Y-%m-%d").date()
    d2 = datetime.strptime(date_to_str, "%Y-%m-%d").date()

    parts = []
    try:
        for a, b in month_safe_chunks(d1, d2):
            # Analytics API expects YYYY-MM-DD (no Z). We keep it consistent with UI.
            part = client.fetch_analytics_sku_units(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"))
            parts.append(part)
        df = _merge_analytics_chunks(parts)
        return df, "", ""
    except Exception as e:
        title, details = _humanize_ozon_error(e)
        return pd.DataFrame(), title, details
@st.cache_data(ttl=600)
def load_ads_summary(date_from_str: str, date_to_str: str) -> dict:
    base = {"spent": 0.0, "revenue": 0.0, "orders": 0, "drr": 0.0, "cpc": 0.0, "ctr": 0.0, "_note": "", "_debug": {}}

    if perf_client is None:
        base["_note"] = "Performance: PERF_CLIENT_ID / PERF_CLIENT_SECRET не заданы."
        return base

    metrics, note, dbg = perf_client.fetch_shop_summary(date_from_str, date_to_str)
    out = {**metrics, "_note": note, "_debug": dbg}
    return out




@st.cache_data(ttl=600)
def load_ads_spend_by_article(date_from_str: str, date_to_str: str) -> dict:
    """
    Распределение рекламных расходов по Артикулу за период.

    Источник:
      - GET /api/client/statistics/daily (CSV): расходы по кампаниям и дням
      - GET /api/client/campaign/{campaign_id}/objects (JSON): список SKU в кампании

    Маппинг SKU -> Артикул берём из COGS (load_cogs()).
    Нераспределённый остаток добиваем в __OTHER_ADS__ так, чтобы дневные суммы и итог совпали с кабинетом.

    Поведение "cloud-safe":
      - не падаем по HTTPError
      - ограничиваем ретраи и общее время на 1 кампанию
      - если кампанию не удалось разобрать (429/5xx/таймаут) — её расход попадёт в __OTHER_ADS__
    """

    out = {"by_article": {}, "total": 0.0, "other": 0.0, "note": "", "debug": {}}

    perf_id = _get_setting("PERF_CLIENT_ID", "").strip()
    perf_secret = _get_setting("PERF_CLIENT_SECRET", "").strip()
    if not perf_id or not perf_secret:
        out["note"] = "Performance: PERF_CLIENT_ID / PERF_CLIENT_SECRET не заданы — реклама по артикулам недоступна."
        return out

    BASE = "https://api-performance.ozon.ru/api/client"

    # --- настройки ограничений (важно для Streamlit Cloud) ---
    OBJ_MAX_RETRIES = 6          # максимум попыток на objects одной кампании
    OBJ_TIMEOUT_SEC = 20         # timeout одного запроса objects
    OBJ_TOTAL_BUDGET_SEC = 60    # максимум времени на одну кампанию (в сумме по ретраям)
    DAILY_TIMEOUT_SEC = 120      # daily обычно крупный
    SLEEP_BETWEEN_CALLS = 0.15   # чуть разгружаем API
    RETRY_429_BASE_SLEEP = 2.0   # базовая пауза при 429 (будет расти)

    def _get_token() -> str:
        r = requests.post(
            f"{BASE}/token",
            json={
                "client_id": int(perf_id) if perf_id.isdigit() else perf_id,
                "client_secret": perf_secret,
                "grant_type": "client_credentials",
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"Не получил access_token: {data}")
        return token

    def _headers(token: str) -> dict:
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "*/*",
            "User-Agent": "ozon-ads-dashboard/1.0",
        }

    def _parse_ru_money(s: str) -> float:
        s = (s or "").strip().replace("\ufeff", "").replace("\xa0", "").replace(" ", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return 0.0

    def _parse_daily(csv_text: str) -> list[dict]:
        import csv as _csv

        rows = []
        lines = (csv_text or "").splitlines()
        if not lines:
            return rows
        if lines[0].startswith("\ufeff"):
            lines[0] = lines[0].replace("\ufeff", "")

        reader = _csv.DictReader(lines, delimiter=";")
        for r in reader:
            cid = str((r.get("ID") or "")).strip()
            d = str((r.get("Дата") or "")).strip()[:10]
            spend = _parse_ru_money(str(r.get("Расход, ₽") or r.get("Расход") or "0"))
            if not cid or not d:
                continue
            if spend == 0:
                continue
            rows.append({"campaign_id": cid, "date": d, "spend": float(spend)})
        return rows

    def _get_campaign_objects_cloudsafe(token: str, campaign_id: str) -> tuple[list[str], str]:
        """
        Возвращает (skus, status):
          status:
            - "ok"
            - "rate_limited" (429)
            - "http_error:<code>"
            - "timeout"
            - "bad_json"
            - "unknown_error"
            - "budget_exceeded"
        """
        url = f"{BASE}/campaign/{campaign_id}/objects"
        t0 = time.time()
        last_status = "unknown_error"

        for attempt in range(1, OBJ_MAX_RETRIES + 1):
            # бюджет по времени на кампанию
            if (time.time() - t0) > OBJ_TOTAL_BUDGET_SEC:
                return [], "budget_exceeded"

            try:
                r = requests.get(url, headers=_headers(token), timeout=OBJ_TIMEOUT_SEC)

                if r.status_code == 429:
                    # backoff
                    last_status = "rate_limited"
                    sleep_s = RETRY_429_BASE_SLEEP * (attempt ** 1.3)
                    time.sleep(min(sleep_s, 15.0))
                    continue

                if r.status_code >= 300:
                    last_status = f"http_error:{r.status_code}"
                    # на 5xx можно попробовать ещё раз, на 4xx (кроме 429) обычно смысла нет
                    if 500 <= r.status_code <= 599:
                        time.sleep(1.0 + attempt * 0.5)
                        continue
                    return [], last_status

                try:
                    data = r.json()
                except Exception:
                    return [], "bad_json"

                skus = []
                if isinstance(data, dict) and isinstance(data.get("list"), list):
                    for item in data["list"]:
                        if isinstance(item, dict) and "id" in item:
                            skus.append(str(item["id"]).strip())

                skus = [s for s in skus if s]
                return skus, "ok"

            except requests.exceptions.Timeout:
                last_status = "timeout"
                time.sleep(0.8 + attempt * 0.4)
                continue
            except Exception:
                last_status = "unknown_error"
                time.sleep(0.8 + attempt * 0.4)
                continue

        return [], last_status

    # --- SKU -> ARTICLE из COGS ---
    sku2art = {}
    try:
        cogs_local = load_cogs()
        if cogs_local is not None and not cogs_local.empty:
            tmp = cogs_local[["sku", "article"]].copy()
            tmp["sku"] = pd.to_numeric(tmp["sku"], errors="coerce").astype("Int64")
            tmp = tmp.dropna(subset=["sku"]).copy()
            tmp["sku"] = tmp["sku"].astype(int).astype(str)
            tmp["article"] = tmp["article"].fillna("").astype(str).str.strip()
            sku2art = {r["sku"]: r["article"] for _, r in tmp.iterrows() if r["sku"] and r["article"]}
    except Exception:
        sku2art = {}

    # --- токен + daily ---
    try:
        token = _get_token()
    except Exception as e:
        out["note"] = f"Performance token error: {e}"
        return out

    try:
        r = requests.get(
            f"{BASE}/statistics/daily",
            headers=_headers(token),
            params={"dateFrom": date_from_str, "dateTo": date_to_str},
            timeout=DAILY_TIMEOUT_SEC,
        )
        if r.status_code >= 300:
            out["note"] = f"Performance daily error: {r.status_code}"
            out["debug"] = {"daily_status": r.status_code, "daily_text_head": (r.text or "")[:400]}
            return out
        daily_rows = _parse_daily(r.text)
    except Exception as e:
        out["note"] = f"Performance daily request failed: {e}"
        return out

    # --- TOTAL по дням (как в кабинете) ---
    total_by_day: dict[str, float] = {}
    for row in daily_rows:
        d = row["date"]
        total_by_day[d] = total_by_day.get(d, 0.0) + float(row["spend"])

    # --- распределение ---
    camp2skus: dict[str, list[str]] = {}
    camp2status: dict[str, str] = {}
    agg: dict[tuple[str, str], float] = {}   # (date, article) -> spend

    # статистика "почему ушло в OTHER"
    skipped_campaigns = 0
    failed_campaigns = 0
    empty_objects_campaigns = 0
    no_mapping_campaigns = 0

    for row in daily_rows:
        cid = row["campaign_id"]
        d = row["date"]
        spend = float(row["spend"])

        if cid not in camp2skus:
            skus, status = _get_campaign_objects_cloudsafe(token, cid)
            camp2skus[cid] = skus
            camp2status[cid] = status
            time.sleep(SLEEP_BETWEEN_CALLS)

        skus = camp2skus.get(cid) or []
        status = camp2status.get(cid, "unknown_error")

        if status != "ok":
            # не смогли получить objects -> пусть расход уходит в OTHER (через diff ниже)
            failed_campaigns += 1
            continue

        if not skus:
            empty_objects_campaigns += 1
            continue

        # артикулы по sku
        arts = []
        for sku in skus:
            art = sku2art.get(str(sku))
            if art:
                arts.append(art)

        if not arts:
            no_mapping_campaigns += 1
            continue

        part = spend / len(arts)
        for art in arts:
            key = (d, art)
            agg[key] = agg.get(key, 0.0) + part

    # --- сколько распределили по дням ---
    alloc_by_day: dict[str, float] = {}
    for (d, _a), v in agg.items():
        alloc_by_day[d] = alloc_by_day.get(d, 0.0) + float(v)

    # --- добиваем OTHER, чтобы дневные суммы совпали ---
    for d, total in total_by_day.items():
        alloc = alloc_by_day.get(d, 0.0)
        diff = float(total) - float(alloc)
        if abs(diff) >= 0.01:
            key = (d, "__OTHER_ADS__")
            agg[key] = agg.get(key, 0.0) + diff

    # --- агрегация по артикулу за период ---
    by_article: dict[str, float] = {}
    for (_d, art), v in agg.items():
        by_article[art] = by_article.get(art, 0.0) + float(v)

    total = float(sum(by_article.values()))
    other = float(by_article.get("__OTHER_ADS__", 0.0))

    out["by_article"] = by_article
    out["total"] = total
    out["other"] = other

    # note (коротко)
    notes = []
    if failed_campaigns:
        notes.append(f"⚠️ Кампаний с ошибками objects: {failed_campaigns} (их расход ушёл в __OTHER_ADS__).")
    if no_mapping_campaigns:
        notes.append(f"⚠️ Кампаний без сопоставления SKU→Артикул: {no_mapping_campaigns} (ушло в __OTHER_ADS__).")
    out["note"] = " ".join(notes)

    out["debug"] = {
        "period": f"{date_from_str}..{date_to_str}",
        "daily_rows_with_spend": int(len(daily_rows)),
        "campaigns_cached": int(len(camp2skus)),
        "sku2art_size": int(len(sku2art)),
        "failed_objects_campaigns": int(failed_campaigns),
        "empty_objects_campaigns": int(empty_objects_campaigns),
        "no_mapping_campaigns": int(no_mapping_campaigns),
        "other_total": round(other, 2),
        "total_allocated": round(total, 2),
        "status_sample": dict(list(camp2status.items())[:15]),
    }
    return out

def allocate_ads_by_article(sku_table: pd.DataFrame, ads_by_article: dict) -> pd.DataFrame:
    """Распределяем рекламу по SKU-строкам, но так, чтобы сумма по одному артикулу сохранялась.

    ads_by_article: dict {article: spend_total_for_period}
    """
    out = sku_table.copy()
    if out is None or out.empty:
        out["ads_total"] = 0.0
        return out

    if "article" not in out.columns:
        out["article"] = ""
    out["article"] = out["article"].fillna("").astype(str).str.strip()

    # база для долей внутри артикула: выручка SKU
    base = pd.to_numeric(out.get("accruals_net", 0.0), errors="coerce").fillna(0.0)
    out["_ads_base"] = base

    out["ads_total"] = 0.0

    # распределяем внутри каждого артикула
    for art, spend in (ads_by_article or {}).items():
        spend = float(spend or 0.0)
        if spend == 0:
            continue

        mask = out["article"].eq(str(art))
        if not mask.any():
            continue

        s = out.loc[mask, "_ads_base"]
        denom = float(s.sum())
        if denom > 0:
            out.loc[mask, "ads_total"] = spend * (s / denom)
        else:
            # если выручки нет — делим поровну
            n = int(mask.sum())
            out.loc[mask, "ads_total"] = spend / max(n, 1)

    # Остаток OTHER не распределяем по SKU (чтобы не искажать товары),
    # но он будет учтён в плитке "Расход на рекламу" (total включает OTHER).
    out = out.drop(columns=["_ads_base"], errors="ignore")
    return out


# ================== SOLD SKU TABLE ==================
def build_sold_sku_table(df_ops: pd.DataFrame, cogs_df_local: pd.DataFrame, analytics_units=None) -> pd.DataFrame:
    # Эквайринг: распределяем acquiring_amount по SKU на уровне posting_number (до фильтрации sku)
    df_ops = allocate_acquiring_amount_by_posting(df_ops)

    sku_df = df_ops[df_ops["sku"].notna()].copy()
    if sku_df.empty:
        return pd.DataFrame()

    sku_df["sku"] = pd.to_numeric(sku_df["sku"], errors="coerce").astype("Int64")
    sku_df = sku_df.dropna(subset=["sku"]).copy()
    sku_df["sku"] = sku_df["sku"].astype(int)

    # расходы (в API обычно минусом)
    sku_df["commission_cost"] = (-sku_df["sale_commission"]).clip(lower=0.0)
    sku_df["services_cost"] = (-sku_df["services_sum"]).clip(lower=0.0)
    # Эквайринг: храним распределенный amount (со знаком). Расход посчитаем НЕТТО на уровне SKU.
    sku_df["acquiring_amount"] = pd.to_numeric(sku_df.get("acquiring_amount_alloc", 0), errors="coerce").fillna(0.0)

    # “Баллы за скидки” и “Программы партнёров” (в ЛК идут ПЛЮСОМ к выручке)
    sku_df["bonus_amt"] = pd.to_numeric(sku_df.get("bonus_points", 0), errors="coerce").fillna(0.0)
    sku_df["partner_amt"] = pd.to_numeric(sku_df.get("partner_programs", 0), errors="coerce").fillna(0.0)


    # количества
    # Важно: в finance API поле `type` не всегда строго = orders/returns для всех строк,
    # которые ЛК относит к заказам/возвратам. Поэтому классифицируем по названию операции и знаку выручки.
    def _qty_kind(row) -> str:
        t = str(row.get("type", "")).lower()
        name = str(row.get("type_name", "")).lower()
        code = str(row.get("operation_type", "")).lower()
        s = f"{t} {name} {code}"

        # Возвраты/сторно
        if any(k in s for k in ["возврат", "return", "refund", "сторно", "отмена", "cancell"]):
            return "returns"

        # Продажи/реализация/доставка (выкуп)
        if any(k in s for k in ["продаж", "реализац", "выкуп", "sale", "delivered", "достав", "передан"]):
            return "orders"

        # Фоллбек по знаку выручки
        accr = _to_float(row.get("accruals_for_sale", 0))
        if accr < 0:
            return "returns"
        if accr > 0:
            return "orders"
        return ""

    sku_df["_qty_kind"] = sku_df.apply(_qty_kind, axis=1)
    sku_df["qty_orders"] = sku_df.apply(lambda r: r["qty"] if r["_qty_kind"] == "orders" else 0.0, axis=1)
    sku_df["qty_returns"] = sku_df.apply(lambda r: r["qty"] if r["_qty_kind"] == "returns" else 0.0, axis=1)


    g = (
        sku_df.groupby(["sku"], as_index=False)
        .agg(
            name=("name", "first"),
            qty_orders=("qty_orders", "sum"),
            qty_returns=("qty_returns", "sum"),
            gross_sales=("accruals_for_sale", "sum"),
            amount_net=("amount", "sum"),
            commission=("commission_cost", "sum"),
            logistics=("services_cost", "sum"),
            acquiring_amount=("acquiring_amount", "sum"),
            bonus_points=("bonus_amt", "sum"),
            partner_programs=("partner_amt", "sum"),
        )
    )

    # --- Количества (Orders / Buyout / Returns) должны совпадать с ЛК.
    # ВАЖНО: ЛК (юнит-экономика) считает количества по Analytics, а не по Finance.
    # Finance-операции по "Доставка покупателю" могут приходить в другом месяце, поэтому qty из Finance будет меньше.
    if analytics_units is not None and not analytics_units.empty and "sku" in analytics_units.columns:
        au = analytics_units.copy()
        au["sku"] = pd.to_numeric(au["sku"], errors="coerce").astype("Int64")
        au = au.dropna(subset=["sku"]).copy()
        au["sku"] = au["sku"].astype(int)

        # normalize columns
        colmap = {
            "ordered_units": "qty_orders_analytics",
            "delivered_units": "qty_delivered_analytics",
            "returned_units": "qty_returns_analytics",
            "cancelled_units": "qty_cancelled_analytics",
        }
        for src, dst in colmap.items():
            if src in au.columns:
                au[dst] = pd.to_numeric(au[src], errors="coerce").fillna(0.0)
            else:
                au[dst] = 0.0

        au = au[["sku"] + list(colmap.values())]
        g = g.merge(au, on="sku", how="left")

        # Keep finance-based qty for debug
        g["qty_orders_finance"] = g["qty_orders"]
        g["qty_returns_finance"] = g["qty_returns"]

        # Override with Analytics (if present)
        g["qty_orders"] = g["qty_orders_analytics"].fillna(0.0)
        g["qty_returns"] = g["qty_returns_analytics"].fillna(0.0)

        # Buyout: prefer delivered_units, else ordered - returned
        delivered = g.get("qty_delivered_analytics")
        if delivered is not None:
            g["qty_buyout"] = delivered.fillna(0.0)
        else:
            g["qty_buyout"] = g["qty_orders"] - g["qty_returns"]
    else:
        g["qty_buyout"] = g["qty_orders"] - g["qty_returns"]

    # Эквайринг (как в ЛК): NЕТТО по amount, затем переводим в расход (плюс)
    if "acquiring_amount" in g.columns:
        g["acquiring"] = (-pd.to_numeric(g["acquiring_amount"], errors="coerce").fillna(0.0)).clip(lower=0.0)
    else:
        g["acquiring"] = 0.0
    # ВАЖНО: “Выручка” как в ЛК = Выручка Ozon + Баллы за скидки + Программы партнёров
    g["accruals_net"] = g["gross_sales"] + g["bonus_points"] + g["partner_programs"]

    g["sale_costs"] = g["commission"] + g["logistics"] + g["acquiring"]

    # COGS
    if cogs_df_local is None or cogs_df_local.empty:
        g["article"] = ""
        g["cogs_unit"] = 0.0
    else:
        c2 = cogs_df_local.copy()
        c2["sku"] = pd.to_numeric(c2["sku"], errors="coerce").astype("Int64")
        c2 = c2.dropna(subset=["sku"]).copy()
        c2["sku"] = c2["sku"].astype(int)
        if "article" not in c2.columns:
            c2["article"] = ""
        if "cogs" not in c2.columns:
            c2["cogs"] = 0.0
        c2["cogs"] = pd.to_numeric(c2["cogs"], errors="coerce").fillna(0.0)
        c2 = c2[["sku", "article", "cogs"]].drop_duplicates(subset=["sku"], keep="last")
        g = g.merge(c2.rename(columns={"cogs": "cogs_unit"}), how="left", on="sku")
        g["article"] = g["article"].fillna("").astype(str)
        g["cogs_unit"] = pd.to_numeric(g["cogs_unit"], errors="coerce").fillna(0.0)

    # автозаполнение артикулов по названию (оставляем как было)
    try:
        known = g[(g["article"].astype(str).str.strip() != "") & g["name"].notna()].copy()
        if not known.empty:
            name_to_article = (
                known.assign(_a=known["article"].astype(str).str.strip())
                .groupby("name")["_a"]
                .agg(lambda s: s.value_counts().index[0])
            )
            mask_empty = g["article"].astype(str).str.strip() == ""
            g.loc[mask_empty, "article"] = (
                g.loc[mask_empty, "name"]
                .map(name_to_article)
                .apply(lambda a: f"Дубль ({a})" if isinstance(a, str) and a.strip() else "")
            )
    except Exception:
        pass
    g["article"] = g["article"].fillna("").astype(str)

    g["cogs_total"] = (g["qty_buyout"].clip(lower=0.0) * g["cogs_unit"]).fillna(0.0)

    g = g.sort_values("accruals_net", ascending=False)
    return g


def allocate_tax_by_share(sku_table: pd.DataFrame, total_tax: float) -> pd.DataFrame:
    out = sku_table.copy()
    total_sales = out["accruals_net"].sum()
    out["tax_total"] = (out["accruals_net"] / total_sales) * float(total_tax) if total_sales and total_tax else 0.0
    return out

def allocate_cost_by_share(sku_table: pd.DataFrame, total_cost: float, out_col: str) -> pd.DataFrame:
    out = sku_table.copy()
    total_sales = float(out["accruals_net"].sum()) if "accruals_net" in out.columns else 0.0
    if total_sales and total_cost:
        out[out_col] = (out["accruals_net"] / total_sales) * float(total_cost)
    else:
        out[out_col] = 0.0
    return out


def compute_profitability(sku_table: pd.DataFrame) -> pd.DataFrame:
    out = sku_table.copy()

    # гарантируем колонки
    for c in ["accruals_net", "sale_costs", "cogs_total", "tax_total", "qty_buyout", "cogs_unit"]:
        if c not in out.columns:
            out[c] = 0.0

    for c in ["ads_total", "opex_total"]:
        if c not in out.columns:
            out[c] = 0.0

    
    # прибыль:
    # 1) Прибыль (до налога и опер. расходов) = Выручка − Расходы Ozon − Реклама (клик) − Реклама (заказ) − Себестоимость всего
    # helper: безопасно получить числовую колонку (если колонки нет — вернём нули по индексам)
    def _col_num(df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    # ВАЖНО: в твоём интерфейсе сейчас:
    # - "Реклама (клик), ₽" берётся из out["ads_total"]
    # - "Реклама (заказ), ₽" берётся из out["ads_spend_click"]
    ads_click = _col_num(out, "ads_total")
    ads_order = _col_num(out, "ads_spend_click")

    out["profit_before_tax_opex"] = (
        pd.to_numeric(out.get("accruals_net", 0.0), errors="coerce").fillna(0.0)
        - pd.to_numeric(out.get("sale_costs", 0.0), errors="coerce").fillna(0.0)
        - ads_click
        - ads_order
        - pd.to_numeric(out.get("cogs_total", 0.0), errors="coerce").fillna(0.0)
    )

    out["profit_before_tax_opex_per_unit"] = out.apply(
        lambda r: (float(r["profit_before_tax_opex"]) / float(r.get("qty_buyout", 0) or 0))
        if float(r.get("qty_buyout", 0) or 0) > 0 else 0.0,
        axis=1,
    )

    # 2) Прибыль (после налога и опер. расходов) — как было раньше, но учитываем и рекламу (заказ)
    out["profit"] = (
        out["profit_before_tax_opex"]
        - pd.to_numeric(out.get("tax_total", 0.0), errors="coerce").fillna(0.0)
        - pd.to_numeric(out.get("opex_total", 0.0), errors="coerce").fillna(0.0)
    )

    out["profit_per_unit"] = out.apply(
        lambda r: (float(r["profit"]) / float(r.get("qty_buyout", 0) or 0))
        if float(r.get("qty_buyout", 0) or 0) > 0 else 0.0,
        axis=1,
    )

    out["margin_%"] = out.apply(
        lambda r: (float(r["profit"]) / float(r["accruals_net"]) * 100.0) if float(r.get("accruals_net", 0) or 0) else 0.0,
        axis=1,
    )

    # ROI по ТЗ: (Прибыль на 1 шт) / (Себестоимость 1 шт)
    out["roi_%"] = out.apply(
        lambda r: (float(r["profit_per_unit"]) / float(r.get("cogs_unit", 0) or 0) * 100.0) if float(r.get("cogs_unit", 0) or 0) > 0 else 0.0,
        axis=1,
    )

    return out

def export_soldsku_xlsx(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="SoldSKU")
    return bio.getvalue()

# ================== TILE LOGIC ==================
def _delta_pct(cur, prev):
    prev = float(prev)
    cur = float(cur)
    if prev == 0:
        return None
    return (cur - prev) / abs(prev) * 100.0

def _tile_class(delta, is_expense: bool, good_when_up: bool = False):
    if delta is None:
        return "ts-neutral", "—"
    arrow = "▲" if delta > 0 else "▼"
    txt = f"{arrow} {delta:+.1f}%"
    if is_expense:
        if good_when_up:
            return ("ts-good" if delta > 0 else "ts-bad"), txt
        return ("ts-good" if delta < 0 else "ts-bad"), txt
    return ("ts-good" if delta > 0 else "ts-bad"), txt

def render_tiles(tiles: list[dict], cols_per_row: int = 4, row_gap_px: int = 16):
    for i in range(0, len(tiles), cols_per_row):
        row = tiles[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for c, t in zip(cols, row):
            klass, delta_txt = _tile_class(t.get("delta"), t.get("is_expense", False), t.get("good_when_up", False))
            c.markdown(
                f"""
<div class="ts-tile {klass}">
  <div class="ts-title">{t.get("title","")}</div>
  <div class="ts-value">{t.get("value","")}</div>
  <div class="ts-delta">{delta_txt}</div>
</div>
""",
                unsafe_allow_html=True
            )
        if i + cols_per_row < len(tiles):
            st.markdown(f"<div style='height:{int(row_gap_px)}px'></div>", unsafe_allow_html=True)

# ================== KPI CORE ==================
STORAGE_FBO_TYPE_NAMES = {"Услуга размещения товаров на складе"}

def calc_kpi(df_ops_local: pd.DataFrame, sold_local: pd.DataFrame):
    total_amount = df_ops_local["amount"].sum() if not df_ops_local.empty else 0.0
    sales_net = float(sold_local["accruals_net"].sum()) if sold_local is not None and not sold_local.empty else 0.0

    sum_sale_commission = df_ops_local["sale_commission"].sum() if not df_ops_local.empty else 0.0
    commission_cost = max(0.0, -float(sum_sale_commission))

    over_local = df_ops_local[df_ops_local["sku"].isna()].copy() if not df_ops_local.empty else pd.DataFrame(columns=["type_name", "amount"])
    storage_fbo_raw = 0.0
    if not over_local.empty:
        storage_fbo_raw = over_local[over_local["type_name"].astype(str).isin(STORAGE_FBO_TYPE_NAMES)]["amount"].sum()
    storage_fbo = to_cost(storage_fbo_raw)

    if sold_local is None or sold_local.empty:
        qty_orders = 0
        qty_returns = 0
        buyout_pct = 0.0
        total_tax = 0.0
        total_cogs = 0.0
        sale_costs = 0.0
    else:
        qty_orders = _to_int(sold_local["qty_orders"].sum())
        qty_returns = _to_int(sold_local["qty_returns"].sum())
        buyout_pct = ((qty_orders - qty_returns) / qty_orders * 100.0) if qty_orders else 0.0

        total_tax = float(sales_net) * 0.06

        tmp = allocate_tax_by_share(sold_local, total_tax)
        tmp = compute_profitability(tmp)
        total_cogs = float(tmp["cogs_total"].sum())
        sale_costs = float(sold_local["sale_costs"].sum())

    net_profit = float(total_amount) - float(total_cogs) - float(total_tax)
    profit_pct = (net_profit / sales_net * 100.0) if sales_net else 0.0

    return {
        "sales_net": float(sales_net),
        "qty_orders": int(qty_orders),
        "qty_returns": int(qty_returns),
        "buyout_pct": float(buyout_pct),
        "storage_fbo": float(storage_fbo),
        "sale_costs": float(to_cost(sale_costs)),
        "cogs": float(to_cost(total_cogs)),
        "tax": float(to_cost(total_tax)),
        "commission_cost": float(to_cost(commission_cost)),
        "amount_total": float(total_amount),
        "net_profit": float(net_profit),
        "profit_pct": float(profit_pct),
    }

def month_name_ru(m: int) -> str:
    names = ["", "Январь","Февраль","Март","Апрель","Май","Июнь","Июль","Август","Сентябрь","Октябрь","Ноябрь","Декабрь"]
    return names[m] if 1 <= m <= 12 else str(m)

# ================== UI ==================
tab1, tab2, tab3, tab4 = st.tabs(["Общие показатели", "Сводка по месяцам", "ABС-анализ", "Опер. расходы"])

# ================== TAB 1 ==================
with tab1:
    st.subheader("Сводка магазина за выбранный период")

    today = date.today()
    yesterday = today - timedelta(days=1)

    presets = ["Последний день", "Последние 7 дней", "Последние 30 дней", "Произвольный"]
    preset = st.selectbox("Период", presets, index=2)

    def compute_range_from_preset(p: str):
        if p == "Последний день":
            return (yesterday, yesterday)
        if p == "Последние 7 дней":
            return (yesterday - timedelta(days=6), yesterday)
        if p == "Последние 30 дней":
            return (yesterday - timedelta(days=29), yesterday)
        return (yesterday - timedelta(days=29), yesterday)

    d_from, d_to = compute_range_from_preset(preset)

    if preset == "Произвольный":
        c1, c2 = st.columns(2)
        with c1:
            d_from = st.date_input("Дата с", value=d_from)
        with c2:
            d_to = st.date_input("Дата по", value=d_to)
    else:
        st.caption(f"Выбранный период: {d_from.strftime('%Y-%m-%d')} — {d_to.strftime('%Y-%m-%d')}")

    if d_from > d_to:
        st.warning("Дата начала больше даты конца — поправь период.")
        st.stop()

    days_len = (d_to - d_from).days + 1
    prev_to = d_from - timedelta(days=1)
    prev_from = prev_to - timedelta(days=days_len - 1)
    st.caption(f"Сравнение: предыдущий период {prev_from.strftime('%Y-%m-%d')} — {prev_to.strftime('%Y-%m-%d')}")

    ops_now, ops_err_title, ops_err_details = load_ops_range(d_from.strftime("%Y-%m-%d"), d_to.strftime("%Y-%m-%d"))
    if ops_err_title:
        _block_with_retry(ops_err_title, ops_err_details, cache_clear_fn=load_ops_range.clear)


    # ================== DEBUG: разбор логистики по одному артикулу (Polyarnaya-210) ==================
    # ================== DEBUG: РАЗБОР УСЛУГ (services) ИЗ СЫРЫХ ОПЕРАЦИЙ ==================
    df_ops = ops_to_df(ops_now)
    # Эквайринг: распределяем по SKU на полном df_ops (по posting_number)
    df_ops = allocate_acquiring_cost_by_posting(df_ops)
    df_ops = redistribute_ops_without_items(df_ops)  # ✅ ДОБАВИТЬ

    df_ops_prev = pd.DataFrame(columns=df_ops.columns)
    if prev_from <= prev_to:
        ops_prev, ops_prev_err_title, ops_prev_err_details = load_ops_range(prev_from.strftime("%Y-%m-%d"), prev_to.strftime("%Y-%m-%d"))
        if ops_prev_err_title:
            _block_with_retry(ops_prev_err_title, ops_prev_err_details, cache_clear_fn=load_ops_range.clear)
        df_ops_prev = ops_to_df(ops_prev)
        df_ops_prev = redistribute_ops_without_items(df_ops_prev)  # ✅ ДОБАВИТЬ

    analytics_df, ana_err_title, ana_err_details = load_analytics_range(
        d_from.strftime("%Y-%m-%d"),
        d_to.strftime("%Y-%m-%d"),
    )
    if ana_err_title:
        st.warning(f"Analytics (qty) не загрузились: {ana_err_title}")
        with st.expander("Details", expanded=False):
            st.write(ana_err_details)
    sold = build_sold_sku_table(df_ops, cogs_df, analytics_df)

    # --- DEBUG: разрез операций по одному SKU (помогает найти расхождения с ЛК)
    debug_sku = st.sidebar.text_input(
        "DEBUG SKU (опционально)",
        value="",
        help="Введите SKU, чтобы увидеть разрез операций и понять, какие типы не попали в заказы/возвраты.",
    )
    if str(debug_sku).strip().isdigit():
        _sku = int(str(debug_sku).strip())
        with st.expander(f"DEBUG: операции по SKU {_sku}", expanded=False):
            _ops = df_ops.copy()
            _ops["sku"] = pd.to_numeric(_ops.get("sku"), errors="coerce").astype("Int64")
            _ops = _ops[_ops["sku"] == _sku].copy()
            if _ops.empty:
                st.info("По этому SKU нет операций в выбранном периоде.")
            else:
                # Короткий свод по типам
                cols = [
                    "operation_date",
                    "type",
                    "operation_type",
                    "type_name",
                    "posting_number",
                    "qty",
                    "accruals_for_sale",
                    "sale_commission",
                    "services_sum",
                    "acquiring_amount_alloc" if "acquiring_amount_alloc" in _ops.columns else "acquiring_amount",
                    "bonus_points",
                    "partner_programs",
                    "amount",
                ]
                cols = [c for c in cols if c in _ops.columns]
                st.dataframe(_ops[cols].sort_values("operation_date"))

                gb_cols = [c for c in ["type", "operation_type", "type_name"] if c in _ops.columns]
                if gb_cols:
                    agg_map = {c: "sum" for c in ["qty", "accruals_for_sale", "sale_commission", "services_sum", "bonus_points", "partner_programs", "amount"] if c in _ops.columns}
                    gdebug = _ops.groupby(gb_cols, as_index=False).agg(agg_map)
                    st.write("Свод по типам операций:")
                    st.dataframe(gdebug.sort_values("accruals_for_sale", ascending=False))
    analytics_prev_df, _, _ = load_analytics_range(prev_from.strftime('%Y-%m-%d'), prev_to.strftime('%Y-%m-%d')) if (not df_ops_prev.empty and prev_from <= prev_to) else (pd.DataFrame(), '', '')
    sold_prev = build_sold_sku_table(df_ops_prev, cogs_df, analytics_prev_df) if not df_ops_prev.empty else pd.DataFrame()

    k = calc_kpi(df_ops, sold)
    k_prev = calc_kpi(df_ops_prev, sold_prev)

    ads_now_raw = load_ads_summary(d_from.strftime("%Y-%m-%d"), d_to.strftime("%Y-%m-%d"))
    ads_prev_raw = load_ads_summary(prev_from.strftime("%Y-%m-%d"), prev_to.strftime("%Y-%m-%d"))

    # Распределение рекламных расходов по артикулам (точнее, чем пропорция выручке)
    ads_alloc_now = load_ads_spend_by_article(d_from.strftime("%Y-%m-%d"), d_to.strftime("%Y-%m-%d"))
    ads_alloc_prev = load_ads_spend_by_article(prev_from.strftime("%Y-%m-%d"), prev_to.strftime("%Y-%m-%d"))

    # Берём расход из распределения (total включает __OTHER_ADS__ и совпадает с "как в кабинете")
    ads_now = {**ads_now_raw}
    ads_prev = {**ads_prev_raw}
    ads_now["spent"] = float(ads_alloc_now.get("total", 0.0) or 0.0)
    ads_prev["spent"] = float(ads_alloc_prev.get("total", 0.0) or 0.0)

    # ---- ROAS ----
    def calc_roas(ads: dict) -> float:
        spent = float(ads.get("spent", 0) or 0)
        revenue = float(ads.get("revenue", 0) or 0)
        return (revenue / spent) if spent > 0 else 0.0

    roas_now = calc_roas(ads_now)
    roas_prev = calc_roas(ads_prev)


    # ---- OPEX (ручные операционные расходы) ----
    opex_now = opex_sum_period(df_opex, d_from, d_to)
    opex_prev = opex_sum_period(df_opex, prev_from, prev_to)

    # note от Performance показываем, но НЕ завязываем на него логику переменных
    ads_tiles = []

    # note можно показывать отдельно
    if ads_now.get("_note"):
        st.info(ads_now["_note"])
    if ads_alloc_now.get("note"):
        st.info(ads_alloc_now["note"])

    # ads_tiles формируем ВСЕГДА
    ads_tiles = [
        {"title": "Расход на рекламу", "value": money(ads_now.get("spent", 0.0)),
         "delta": _delta_pct(_to_float(ads_now.get("spent", 0.0)), _to_float(ads_prev.get("spent", 0.0))),
         "is_expense": True},

        {"title": "Выручка с рекламы", "value": money(ads_now.get("revenue", 0.0)),
         "delta": _delta_pct(_to_float(ads_now.get("revenue", 0.0)), _to_float(ads_prev.get("revenue", 0.0))),
         "is_expense": False},

        {"title": "Заказы с рекламы", "value": f'{_to_int(ads_now.get("orders", 0))} шт',
         "delta": _delta_pct(_to_float(ads_now.get("orders", 0)), _to_float(ads_prev.get("orders", 0))),
         "is_expense": False},

        {"title": "DRR", "value": f'{_to_float(ads_now.get("drr", 0.0)):.1f}%',
         "delta": _delta_pct(_to_float(ads_now.get("drr", 0.0)), _to_float(ads_prev.get("drr", 0.0))),
         "is_expense": True},

        {"title": "ROAS", "value": f'x{roas_now:.2f}',
         "delta": _delta_pct(roas_now, roas_prev),
         "is_expense": False},

        {"title": "CPC", "value": f'{_to_float(ads_now.get("cpc", 0.0)):.1f} ₽',
         "delta": _delta_pct(_to_float(ads_now.get("cpc", 0.0)), _to_float(ads_prev.get("cpc", 0.0))),
         "is_expense": True},

        {"title": "CTR", "value": f'{_to_float(ads_now.get("ctr", 0.0)):.2f}%',
         "delta": _delta_pct(_to_float(ads_now.get("ctr", 0.0)), _to_float(ads_prev.get("ctr", 0.0))),
         "is_expense": False},
    ]

    sales_tile_value = (
        f'{money(k["sales_net"])} / {k["qty_orders"]} шт'
        if k["qty_orders"]
        else money(k["sales_net"])
    )

    # --- пересчёт KPI по новым формулам (учитываем рекламу + опер. расходы) ---
    ads_spent_now = float(ads_now.get("spent", 0.0) or 0.0)
    ads_spent_prev = float(ads_prev.get("spent", 0.0) or 0.0)

    net_profit_now = float(k["sales_net"]) - float(k["sale_costs"]) - ads_spent_now - float(k["cogs"]) - float(k["tax"]) - float(opex_now)
    net_profit_prev = float(k_prev["sales_net"]) - float(k_prev["sale_costs"]) - ads_spent_prev - float(k_prev["cogs"]) - float(k_prev["tax"]) - float(opex_prev)

    margin_now = (net_profit_now / float(k["sales_net"]) * 100.0) if float(k["sales_net"]) else 0.0
    margin_prev = (net_profit_prev / float(k_prev["sales_net"]) * 100.0) if float(k_prev["sales_net"]) else 0.0

    roi_now = (net_profit_now / float(k["cogs"]) * 100.0) if float(k["cogs"]) else 0.0
    roi_prev = (net_profit_prev / float(k_prev["cogs"]) * 100.0) if float(k_prev["cogs"]) else 0.0

    sales_tile_value = f'{money(k["sales_net"])} / {k["qty_orders"]} шт' if k["qty_orders"] else money(k["sales_net"])
    commission_delta = _delta_pct(k["commission_cost"], k_prev["commission_cost"])

    tiles = [
        {"title": "Продажи", "value": sales_tile_value, "delta": _delta_pct(k["sales_net"], k_prev["sales_net"]), "is_expense": False},
        {"title": "Чистая прибыль", "value": money(net_profit_now), "delta": _delta_pct(net_profit_now, net_profit_prev), "is_expense": False},
        {"title": "Маржинальность", "value": f"{margin_now:.1f}%", "delta": _delta_pct(margin_now, margin_prev), "is_expense": False},
        {"title": "ROI", "value": f"{roi_now:.1f}%", "delta": _delta_pct(roi_now, roi_prev), "is_expense": False},

        {"title": "% выкупа", "value": f'{k["buyout_pct"]:.1f}%', "delta": _delta_pct(k["buyout_pct"], k_prev["buyout_pct"]), "is_expense": False},
        {"title": "Возвраты, шт", "value": str(k["qty_returns"]), "delta": _delta_pct(k["qty_returns"], k_prev["qty_returns"]), "is_expense": True},
        {"title": "Опер. расходы", "value": money(opex_now), "delta": _delta_pct(opex_now, opex_prev), "is_expense": True},
        {"title": "Расходы на продажу", "value": money(k["sale_costs"]), "delta": _delta_pct(k["sale_costs"], k_prev["sale_costs"]), "is_expense": True},

        {"title": "Хранение (FBO)", "value": money(k["storage_fbo"]), "delta": _delta_pct(k["storage_fbo"], k_prev["storage_fbo"]), "is_expense": True},
        {"title": "Себестоимость продаж", "value": money(k["cogs"]), "delta": _delta_pct(k["cogs"], k_prev["cogs"]), "is_expense": True, "good_when_up": True},
        {"title": "Налоги/Комиссия", "value": f'{money(k["tax"])} / {money(k["commission_cost"])}', "delta": commission_delta, "is_expense": True},
    ]

    st.markdown("### Ключевые показатели")
    render_tiles(tiles, cols_per_row=4)

    st.markdown("### Рекламные показатели")
    render_tiles(ads_tiles, cols_per_row=4)

    st.divider()

    over = df_ops[df_ops["sku"].isna()].copy()
    with st.expander("Детали", expanded=False):
        st.markdown("**Данные по операциям**")
        if over.empty:
            st.info("Нет операций без SKU в выбранном периоде.")
        else:
            over_g = (
                over.groupby("type_name", as_index=False)
                .agg(amount=("amount", "sum"))
                .sort_values("amount")
            )
            over_g = over_g.rename(columns={"type_name": "Тип операции", "amount": "Значение"}).copy()
            # оставляем число числом — чтобы сортировка работала корректно
            over_g["Значение"] = pd.to_numeric(over_g["Значение"], errors="coerce").fillna(0.0)
            st.dataframe(
                over_g,
                use_container_width=True,
                hide_index=True,
                column_config={"Значение": st.column_config.NumberColumn(format="%.0f")},
            )

    st.markdown("## Список проданных SKU ")
    if sold is None or sold.empty:
        st.warning("За выбранный период нет SKU-операций (items[].sku).")
    else:
        total_tax = float(sold["accruals_net"].sum()) * 0.06

        # распределяем: налог, реклама, опер. расходы
        sold_view = allocate_tax_by_share(sold, total_tax)

        ads_spent_now = float(ads_now.get("spent", 0.0) or 0.0)
        sold_view = allocate_ads_by_article(sold_view, ads_alloc_now.get("by_article", {}))

        # spend_click из отчёта Performance (products json): MoneySpentFromCPC по SKU
        with st.spinner("Performance API: загружаю рекламу (расходы по SKU)…"):
            try:
                _perf_map_raw = load_perf_spend_click_by_sku(d_from.strftime("%Y-%m-%d"), d_to.strftime("%Y-%m-%d"))
            except Exception:
                _perf_map_raw = {}

        # нормализуем ключи SKU (int) и значения (float), чтобы маппинг не давал 0 из‑за типов
        _perf_map = {}
        try:
            for k, v in (_perf_map_raw or {}).items():
                ks = str(k).strip()
                if ks.endswith(".0"):
                    ks = ks[:-2]
                if ks:
                    _perf_map[int(ks)] = float(v or 0.0)
        except Exception:
            _perf_map = {}

        if not _perf_map:
            st.caption("Performance API: не удалось получить данные по рекламе — колонка будет 0.")

        sold_view["ads_spend_click"] = (
            pd.to_numeric(
                sold_view["sku"].astype(str).str.replace(r"[^\d]", "", regex=True),
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
            .map(_perf_map)
            .fillna(0.0)
        )

        # Опер. расходы распределяем пропорционально выручке SKU
        opex_period = opex_sum_period(df_opex, d_from, d_to)
        sold_view = allocate_cost_by_share(sold_view, opex_period, "opex_total")

        # прибыльные метрики по новым формулам
        sold_view = compute_profitability(sold_view)

        show = sold_view.copy()
        show = show.rename(columns={
            "article": "Артикул",
            "sku": "SKU",
            "name": "Название",
            "qty_orders": "Заказы, шт",
            "qty_returns": "Возвраты, шт",
            "qty_buyout": "Выкуп, шт",
            "accruals_net": "Выручка, ₽",
            "commission": "Комиссия, ₽",
            "logistics": "Услуги/логистика, ₽",
            "acquiring": "Эквайринг, ₽",
            "sale_costs": "Расходы Ozon, ₽",
            "ads_total": "Реклама (клик), ₽",
            "ads_spend_click": "Реклама (заказ), ₽",
            "cogs_unit": "Себестоимость 1 шт, ₽",
            "cogs_total": "Себестоимость всего, ₽",
            "tax_total": "Налог, ₽",
            "opex_total": "Опер. расходы, ₽",
            "profit_before_tax_opex": "Прибыль, ₽",
            "profit_before_tax_opex_per_unit": "Прибыль/шт, ₽",
            "profit": "Прибыль (чистая), ₽",
            "profit_per_unit": "Прибыль/шт (чистая), ₽",
            "margin_%": "Маржинальность, %",
            "roi_%": "ROI, %",
        })


        # === Доп. колонки для витрины (как на Ozon) ===
        show["% выкупа"] = show.apply(
            lambda r: (float(r.get("Выкуп, шт", 0)) / float(r.get("Заказы, шт", 0)) * 100.0)
            if float(r.get("Заказы, шт", 0) or 0) else 0.0,
            axis=1,
        )
        show["Средняя цена продажи, ₽"] = show.apply(
            lambda r: (float(r.get("Выручка, ₽", 0)) / float(r.get("Выкуп, шт", 0)))
            if float(r.get("Выкуп, шт", 0) or 0) else 0.0,
            axis=1,
        )
        show["ДРР, %"] = show.apply(
            lambda r: (float(r.get("Реклама (клик), ₽", 0)) / float(r.get("Выручка, ₽", 0)) * 100.0)
            if float(r.get("Выручка, ₽", 0) or 0) else 0.0,
            axis=1,
        )

        # порядок колонок
        cols = [
            "Артикул","SKU","Название",
            "Заказы, шт","Возвраты, шт","Выкуп, шт","% выкупа",
            "Выручка, ₽","Средняя цена продажи, ₽","ДРР, %",
            "Комиссия, ₽","Услуги/логистика, ₽","Эквайринг, ₽","Расходы Ozon, ₽","Реклама (клик), ₽","Реклама (заказ), ₽",
            "Себестоимость 1 шт, ₽","Себестоимость всего, ₽","Налог, ₽","Опер. расходы, ₽",
            "Прибыль, ₽","Прибыль/шт, ₽","Прибыль (чистая), ₽","Прибыль/шт (чистая), ₽",
            "Маржинальность, %","ROI, %"
        ]
        # 1) Убираем дубли в списке колонок (важно!)
        cols = list(dict.fromkeys(cols))
        
        # 2) Дозаполняем отсутствующие колонки нулями
        for c in cols:
            if c not in show.columns:
                show[c] = 0.0
        
        # 3) Выбираем только нужные колонки и ещё раз убираем дубли (на всякий)
        show = show[cols].copy()
        show = show.loc[:, ~show.columns.duplicated()].copy()

        # 4) SKU держим числом (int) — без вспомогательных SKU
        if "SKU" in show.columns:
            show["SKU"] = pd.to_numeric(show["SKU"], errors="coerce").fillna(0).astype(int)
            # показываем SKU без разделителей тысяч
            show["SKU"] = show["SKU"].astype(str)

        # 5) Числовые целые колонки
        int_cols = ["Заказы, шт", "Возвраты, шт", "Выкуп, шт"]
        for c in int_cols:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").fillna(0).astype(int)
        
        # 6) Денежные колонки (float)
        money_cols = [
            "Выручка, ₽", "Средняя цена продажи, ₽", "Комиссия, ₽", "Услуги/логистика, ₽",
            "Расходы Ozon, ₽", "Реклама (клик), ₽", "Реклама (заказ), ₽",
            "Себестоимость 1 шт, ₽", "Себестоимость всего, ₽",
            "Налог, ₽", "Опер. расходы, ₽",
            "Прибыль, ₽", "Прибыль/шт, ₽", "Прибыль (чистая), ₽", "Прибыль/шт (чистая), ₽",
        ]
        for c in money_cols:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").fillna(0.0)

        pct_cols = ["% выкупа","ДРР, %","Маржинальность, %","ROI, %"]
        for c in pct_cols:
            show[c] = pd.to_numeric(show[c], errors="coerce").fillna(0.0)



        # --- Настройка порядка/видимости колонок (простое решение без компонентов) ---
        # Важно: в Streamlit нет drag&drop перестановки колонок в st.dataframe,
        # поэтому делаем лёгкий UI: выбрать колонку и двигать вверх/вниз + скрывать.
        default_cols = list(show.columns)
        order_key = "soldsku_col_order"
        hide_key = "soldsku_col_hidden"

        # Сохранение настроек (порядок/скрытые) в URL query params,
        # чтобы переживало F5 и повторный заход по ссылке.
        qp_order_key = "soldsku_cols"
        qp_hide_key = "soldsku_hide"

        def _qp_get_one(key: str) -> str:
            """Безопасно читаем query param как строку (поддержка разных версий Streamlit)."""
            try:
                # Streamlit 1.30+: st.query_params
                qp = getattr(st, "query_params", None)
                if qp is not None:
                    v = qp.get(key)
                    if isinstance(v, list):
                        return str(v[0]) if v else ""
                    return str(v) if v is not None else ""
            except Exception:
                pass

            try:
                # Старые версии: experimental_get_query_params
                qp2 = st.experimental_get_query_params()
                v = qp2.get(key, [])
                return str(v[0]) if v else ""
            except Exception:
                return ""

        def _qp_set(**kwargs):
            """Безопасно пишем query params (старая/новая API)."""
            try:
                qp = getattr(st, "query_params", None)
                if qp is not None:
                    for k, v in kwargs.items():
                        qp[k] = v
                    return
            except Exception:
                pass
            try:
                st.experimental_set_query_params(**kwargs)
            except Exception:
                pass

        # 1) Пробуем подхватить сохранённый порядок/скрытые из URL (переживает F5)
        qp_cols = _qp_get_one(qp_order_key).strip()
        qp_hide = _qp_get_one(qp_hide_key).strip()
        if qp_cols and (order_key not in st.session_state):
            cols = [c for c in qp_cols.split(",") if c]
            st.session_state[order_key] = cols
        if qp_hide and (hide_key not in st.session_state):
            hidden = [c for c in qp_hide.split(",") if c]
            st.session_state[hide_key] = hidden

        if order_key not in st.session_state or not isinstance(st.session_state[order_key], list):
            st.session_state[order_key] = default_cols
        if hide_key not in st.session_state or not isinstance(st.session_state[hide_key], list):
            st.session_state[hide_key] = []

        # если появились/исчезли колонки — аккуратно синхронизируем
        cur = [c for c in st.session_state[order_key] if c in default_cols]
        for c in default_cols:
            if c not in cur:
                cur.append(c)
        st.session_state[order_key] = cur
        st.session_state[hide_key] = [c for c in st.session_state[hide_key] if c in default_cols]

        # Держим блок настроек колонок открытым после кликов (Streamlit делает rerun)
        if "soldsku_cols_expanded" not in st.session_state:
            st.session_state["soldsku_cols_expanded"] = True
        if "soldsku_last_col" not in st.session_state:
            st.session_state["soldsku_last_col"] = ""

        with st.expander("⚙️ Колонки таблицы", expanded=False):
            colA, colB, colC = st.columns([2.2, 1.2, 1.6])

            with colA:
                # запоминаем последний выбранный столбец, чтобы после rerun не сбивалось
                _last = st.session_state.get("soldsku_col_picked")
                _idx = 0
                try:
                    if _last in st.session_state[order_key]:
                        _idx = st.session_state[order_key].index(_last)
                except Exception:
                    _idx = 0

                picked = st.selectbox(
                    "Колонка",
                    options=st.session_state[order_key],
                    index=_idx if st.session_state[order_key] else 0,
                    key="soldsku_col_picked",
                )

            with colB:
                left = st.button("⬅️ Влево", use_container_width=True)
                right = st.button("➡️ Вправо", use_container_width=True)

            with colC:
                c1, c2 = st.columns([1.2, 1.0])
                with c1:
                    if st.button("💾 Сохранить", use_container_width=True):
                        # сохраняем порядок/скрытые в URL (переживает F5)
                        try:
                            order_str = ",".join(st.session_state[order_key])
                            hide_str = ",".join(st.session_state[hide_key])
                            _qp_set(**{qp_order_key: order_str, qp_hide_key: hide_str})
                            st.success("Сохранено")
                        except Exception:
                            st.warning("Не удалось сохранить в URL")
                with c2:
                    if st.button("↩️ Сбросить", use_container_width=True):
                        st.session_state[order_key] = default_cols
                        st.session_state[hide_key] = []
                        try:
                            _qp_set(**{qp_order_key: "", qp_hide_key: ""})
                        except Exception:
                            pass
                        st.rerun()

            if picked and (left or right):
                st.session_state["soldsku_cols_expanded"] = True
                st.session_state["soldsku_last_col"] = picked
                cols = st.session_state[order_key]
                i = cols.index(picked)
                j = i - 1 if left else i + 1
                if 0 <= j < len(cols):
                    cols[i], cols[j] = cols[j], cols[i]
                    st.session_state[order_key] = cols
                    # Автосохранение в URL, чтобы переживало перезагрузку страницы
                    try:
                        _qp_set(**{qp_order_key: ",".join(st.session_state[order_key]), qp_hide_key: ",".join(st.session_state[hide_key])})
                    except Exception:
                        pass
                    st.rerun()

            st.session_state[hide_key] = st.multiselect(
                "Скрыть колонки",
                options=st.session_state[order_key],
                default=st.session_state[hide_key],
                key="soldsku_col_hidden_ui",
            )

        visible_cols = [c for c in st.session_state[order_key] if c not in set(st.session_state[hide_key])]
        if visible_cols:
            show = show[visible_cols]

        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Заказы, шт": st.column_config.NumberColumn(format="%.0f"),
                "Возвраты, шт": st.column_config.NumberColumn(format="%.0f"),
                "Выкуп, шт": st.column_config.NumberColumn(format="%.0f"),
                **{c: st.column_config.NumberColumn(format="%.0f") for c in money_cols},
                "% выкупа": st.column_config.NumberColumn(format="%.0f%%"),
                "ДРР, %": st.column_config.NumberColumn(format="%.1f%%"),
                "Маржинальность, %": st.column_config.NumberColumn(format="%.1f%%"),
                "ROI, %": st.column_config.NumberColumn(format="%.1f%%"),
            }
        )

        st.download_button(
            "Скачать XLSX (таблица проданных SKU)",
            data=export_soldsku_xlsx(show),
            file_name=f"ozon_soldsku_{d_from.strftime('%Y-%m-%d')}_{d_to.strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )





# ================== TAB 4 (OPEX) ==================
with tab4:
    st.subheader("Операционные расходы")
    st.caption("Ручные операционные расходы (не из Ozon). Они учитываются в прибыли и распределяются по SKU пропорционально выручке за выбранный период.")

    opex = load_opex()
    types_saved = load_opex_types()

    st.markdown("### Добавить расход")

    c1, c2, c3, c4 = st.columns([1.2, 3.0, 1.2, 1.3])

    with c1:
        st.markdown("**Дата**")
        new_date = st.date_input(
            "Дата",
            value=date.today(),
            key="opex_new_date",
            label_visibility="collapsed",
        )

    with c2:
        st.markdown("**Тип**")
        types_saved = load_opex_types()
        options = (types_saved or []) + ["➕ Добавить новый тип"]

        sel = st.selectbox(
            "Тип",
            options=options,
            index=None,  # важно: ничего не выбрано по умолчанию
            placeholder="Выберите тип расхода…",
            key="opex_type_select",
            label_visibility="collapsed",
        )

        if sel == "➕ Добавить новый тип":
            new_type = st.text_input(
                "Новый тип",
                value="",
                placeholder="Например: Зарплата, Аренда…",
                key="opex_new_type_manual",
                label_visibility="collapsed",
            ).strip()
        else:
            new_type = (sel or "").strip()

    with c3:
        st.markdown("**Сумма, ₽**")
        new_amount = st.number_input(
            "Сумма, ₽",
            min_value=0.0,
            value=0.0,
            step=100.0,
            key="opex_new_amount",
            label_visibility="collapsed",
        )

        with c4:
            st.markdown("&nbsp;")  # создаём “строку” под заголовок, чтобы кнопка стала на один уровень
            add_exp = st.button("Добавить расход", key="opex_add_btn", use_container_width=True)

    if add_exp:
        t = (new_type or "").strip()
        if float(new_amount or 0) <= 0:
            st.warning("Сумма должна быть больше 0.")
        elif not t:
            st.warning("Укажи тип расхода.")
        else:
            types_saved = load_opex_types()
            if t not in types_saved:
                types_saved.append(t)
                save_opex_types(types_saved)

            row = pd.DataFrame([{"date": new_date, "type": t, "amount": float(new_amount)}])
            opex2 = pd.concat([opex, row], ignore_index=True)
            save_opex(opex2)
            st.success("Расход добавлен.")
            st.rerun()

    st.divider()

    # Управление шаблонами (удаление/переименование)
    with st.expander("Шаблоны типов расходов", expanded=False):
        types_saved = load_opex_types()
        if not types_saved:
            st.info("Шаблонов пока нет. Добавь расход с новым типом — он автоматически появится в шаблонах.")
        else:
            tpl_df = pd.DataFrame({"Тип": types_saved, "Удалить": [False] * len(types_saved)})
            tpl_edit = st.data_editor(
                tpl_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "Тип": st.column_config.TextColumn(),
                    "Удалить": st.column_config.CheckboxColumn(width="small"),
                },
                key="opex_tpl_editor",
            )

            csave, cdel = st.columns([1.2, 1.6])
            with csave:
                if st.button("Сохранить шаблоны", key="opex_tpl_apply_btn", use_container_width=True):
                    df = tpl_edit.copy()
                    df["Тип"] = df["Тип"].fillna("").astype(str).str.strip()
                    df = df[df["Тип"] != ""].copy()
                    df = df[df["Удалить"] != True].copy()
                    save_opex_types(df["Тип"].tolist())
                    st.success("Шаблоны обновлены.")
                    st.rerun()
            with cdel:
                if st.button("Удалить отмеченные", key="opex_tpl_del_btn", use_container_width=True):
                    df = tpl_edit.copy()
                    df["Тип"] = df["Тип"].fillna("").astype(str).str.strip()
                    df = df[df["Тип"] != ""].copy()
                    df = df[df["Удалить"] != True].copy()
                    save_opex_types(df["Тип"].tolist())
                    st.success("Удалено.")
                    st.rerun()

    st.markdown("### Список расходов (редактирование / удаление)")

    opex = load_opex()
    # --- фильтр (по умолчанию всё время) ---
    if "opex_filter_year" not in st.session_state:
        st.session_state["opex_filter_year"] = "Все время"
    if "opex_filter_month" not in st.session_state:
        st.session_state["opex_filter_month"] = "Весь год"

    years = sorted({d.year for d in opex["date"].dropna()}, reverse=True) if not opex.empty else []
    year_options = ["Все время"] + [str(y) for y in years]
    # если в session_state значение года не из списка — сбрасываем
    if st.session_state.get("opex_filter_year") not in year_options:
        st.session_state["opex_filter_year"] = "Все время"

    cfy, cfm, cfr = st.columns([1.0, 1.0, 1.0])
    with cfy:
        sel_year_lbl = st.selectbox("Год", options=year_options, key="opex_filter_year")
    with cfm:
        if sel_year_lbl != "Все время":
            yy = int(sel_year_lbl)
            months_av = sorted({d.month for d in opex["date"].dropna() if d.year == yy})
            month_options = ["Весь год"] + [month_name_ru(m) for m in months_av]
        else:
            month_options = ["Весь год"]
        # если в session_state значение месяца не из списка — сбрасываем
        if st.session_state.get("opex_filter_month") not in month_options:
            st.session_state["opex_filter_month"] = month_options[0]
        sel_month_lbl = st.selectbox("Месяц", options=month_options, key="opex_filter_month")
    with cfr:
        def _opex_reset_period():
            st.session_state["opex_filter_year"] = "Все время"
            st.session_state["opex_filter_month"] = "Весь год"
        st.button("Сбросить период", key="opex_filter_reset", on_click=_opex_reset_period)

    # применяем фильтр
    opex_show = opex.copy()
    if sel_year_lbl != "Все время":
        yy = int(sel_year_lbl)
        opex_show = opex_show[opex_show["date"].apply(lambda d: hasattr(d, "year") and d.year == yy)].copy()
        if sel_month_lbl != "Весь год":
            mm = {month_name_ru(i): i for i in range(1, 13)}.get(sel_month_lbl)
            if mm:
                opex_show = opex_show[opex_show["date"].apply(lambda d: hasattr(d, "month") and d.month == mm)].copy()

    opex_show = opex_show.sort_values(["date", "type"], ascending=[False, True]).reset_index(drop=True)

    total_all = float(pd.to_numeric(opex["amount"], errors="coerce").fillna(0.0).sum()) if not opex.empty else 0.0
    st.markdown(f"**Сумма операционных расходов за всё время:** {money(total_all)}")

    if opex.empty:
        st.info("Пока нет записей.")
    else:
        view = opex_show.copy()
        view["delete"] = False
        view = view[["delete", "date", "type", "amount"]].rename(columns={
            "delete": "Удалить",
            "date": "Дата",
            "type": "Тип",
            "amount": "Сумма, ₽",
        })

        edited = st.data_editor(
            view,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Удалить": st.column_config.CheckboxColumn(width="small"),
                "Дата": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Тип": st.column_config.TextColumn(),
                "Сумма, ₽": st.column_config.NumberColumn(format="%.0f"),
            },
            key="opex_editor",
        )

        csave, cexp = st.columns([1.6, 6.4])

        with csave:
            if st.button("Сохранить изменения", key="opex_save_btn", use_container_width=True):
                df = edited.copy()
                df = df[df["Удалить"] != True].copy()

                df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce").dt.date
                df["Тип"] = df["Тип"].fillna("").astype(str).str.strip()
                df["Сумма, ₽"] = pd.to_numeric(df["Сумма, ₽"], errors="coerce").fillna(0.0)

                df = df.dropna(subset=["Дата"]).copy()
                df = df[df["Тип"] != ""].copy()

                df2 = df.rename(columns={"Дата": "date", "Тип": "type", "Сумма, ₽": "amount"})[["date", "type", "amount"]]
                save_opex(df2)

                st.success("Сохранено.")
                st.rerun()

        with cexp:
            st.download_button(
                "Скачать XLSX",
                data=export_soldsku_xlsx(opex.rename(columns={"date": "Дата", "type": "Тип", "amount": "Сумма, ₽"})),
                file_name="ozon_opex.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ================== TAB 2 (MONTHS) ==================
with tab2:
    st.subheader("Помесячная сводка")

    import calendar as _cal

    def _fmt_int(x):
        try:
            return f"{int(round(float(x))):,}".replace(",", " ")
        except Exception:
            return "0"

    def fmt_money2(x):
        return f"{_fmt_int(x)} ₽"

    def fmt_pct2(x, digits=1):
        try:
            v = float(x) * 100.0 if abs(float(x)) <= 1.0 else float(x)
            return f"{v:.{digits}f}%"
        except Exception:
            return "0.0%"

    def month_label(ym: str) -> str:
        y, m = ym.split("-")
        m_i = int(m)
        ru_months = ["Январь","Февраль","Март","Апрель","Май","Июнь","Июль","Август","Сентябрь","Октябрь","Ноябрь","Декабрь"]
        return f"{ru_months[m_i-1]} {y}"

    def month_start_end(year: int, month: int):
        start = date(year, month, 1)
        last_day = _cal.monthrange(year, month)[1]
        end = date(year, month, last_day)
        return start, end

    st.caption("Ozon API по операциям запрашивает данные по очереди периодами в один месяц.")

    y_l, m_l = last_closed_month(date.today())
    last_closed = date(y_l, m_l, 1)
    default_from = (last_closed.replace(day=1) - timedelta(days=365)).replace(day=1)

    c1, c2 = st.columns(2)
    with c1:
        m_from_dt = st.date_input("Месяц с", default_from, key="m_from_dt")
    with c2:
        m_to_dt = st.date_input("Месяц по", last_closed, key="m_to_dt")

    m_from_dt = m_from_dt.replace(day=1)
    m_to_dt = m_to_dt.replace(day=1)

    if m_from_dt > m_to_dt:
        st.error("Месяц 'с' не может быть больше месяца 'по'.")
        st.stop()

    months = []
    cur = m_from_dt
    while cur <= m_to_dt:
        months.append(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            cur = cur.replace(month=cur.month + 1, day=1)

    def month_metrics(df_ops_m: pd.DataFrame) -> dict:
        sku_ops = df_ops_m[df_ops_m["sku"].notna()].copy()

        sales_accr = float(sku_ops[sku_ops["type"].eq("orders")]["accruals_for_sale"].sum())
        returns_accr = float(sku_ops[sku_ops["type"].eq("returns")]["accruals_for_sale"].sum())
        revenue_net = sales_accr + returns_accr

        orders_qty = int(round(float(sku_ops[sku_ops["type"].eq("orders")].get("qty", 0).sum()))) if "qty" in sku_ops.columns else 0
        returns_qty = int(round(float(sku_ops[sku_ops["type"].eq("returns")].get("qty", 0).sum()))) if "qty" in sku_ops.columns else 0
        bought_qty = max(orders_qty - returns_qty, 0)
        buyout_pct = (bought_qty / orders_qty * 100.0) if orders_qty else 0.0

        commission_sum = float((-sku_ops["sale_commission"]).clip(lower=0).sum())
        commission_pct = (commission_sum / revenue_net * 100.0) if revenue_net else 0.0

        logistic_sum = float((-sku_ops["services_sum"]).clip(lower=0).sum())
        logistic_avg = (logistic_sum / bought_qty) if bought_qty else 0.0
        logistic_pct = (logistic_sum / revenue_net * 100.0) if revenue_net else 0.0

        over = df_ops_m[df_ops_m["sku"].isna()].copy()
        storage_fbo = float((-over[over["type_name"].eq("Услуга размещения товаров на складе")]["amount"]).clip(lower=0).sum())
        storage_pct = (storage_fbo / revenue_net * 100.0) if revenue_net else 0.0

        reviews_cost = float((-over[over["type_name"].str.contains("Баллы за отзывы", case=False, na=False)]["amount"]).clip(lower=0).sum())
        reviews_pct = (reviews_cost / revenue_net * 100.0) if revenue_net else 0.0

        mask_known = (
            over["type_name"].eq("Услуга размещения товаров на складе")
            | over["type_name"].str.contains("Баллы за отзывы", case=False, na=False)
        )
        other_expenses = float((-over[~mask_known]["amount"]).clip(lower=0).sum())
        other_expenses_pct = (other_expenses / revenue_net * 100.0) if revenue_net else 0.0

        compensations = float((over[over["type_name"].str.contains("Компенсац", case=False, na=False)]["amount"]).sum())
        fines = float((-over[over["type_name"].str.contains("штраф", case=False, na=False)]["amount"]).clip(lower=0).sum())
        paid_accept = float((-over[over["type_name"].str.contains("Платн", case=False, na=False)]["amount"]).clip(lower=0).sum())
        adjustments = float((over[over["type_name"].str.contains("Коррект", case=False, na=False)]["amount"]).sum())

        total_ozon_exp = commission_sum + logistic_sum + storage_fbo + reviews_cost + other_expenses
        share_ozon = (total_ozon_exp / revenue_net * 100.0) if revenue_net else 0.0

        taxes = revenue_net * 0.06 if revenue_net else 0.0
        to_pay = float(sku_ops["amount"].sum())

        profit = revenue_net - total_ozon_exp - taxes
        profit_pct_price = (profit / revenue_net * 100.0) if revenue_net else 0.0

        days_in_month = 30
        if not df_ops_m.empty:
            s = str(df_ops_m.iloc[0].get("operation_date", ""))[:10]
            try:
                dt0 = datetime.strptime(s, "%Y-%m-%d").date()
                days_in_month = _cal.monthrange(dt0.year, dt0.month)[1]
            except Exception:
                pass

        weeks = max(round(days_in_month / 7.0, 2), 1.0)
        profit_week = profit / weeks if weeks else profit
        avg_price = (revenue_net / bought_qty) if bought_qty else 0.0

        return {
            "Кол-во заказов": orders_qty,
            "Выкуплено шт.": bought_qty,
            "% выкупа": buyout_pct,

            "Выручка с учетом возвратов": revenue_net,
            "Ср. цена": avg_price,

            "Общие расходы Ozon": total_ozon_exp,
            "Комиссия": commission_sum,
            "% комиссии от выручки": commission_pct,

            "Логистика": logistic_sum,
            "Ср. логистика": logistic_avg,
            "% Логистики": logistic_pct,

            "Хранение": storage_fbo,
            "% хранения": storage_pct,

            "Отзывы за баллы": reviews_cost,
            "% отзывов за баллы": reviews_pct,

            "Прочие расходы": other_expenses,
            "% прочих расходов": other_expenses_pct,

            "Компенсации": compensations,
            "Штрафы": fines,
            "Платная приемка": paid_accept,
            "Корректировки": adjustments,

            "Доля Ozon, %": share_ozon,
            "Налоги": taxes,
            "К перечислению": to_pay,

            "Прибыль": profit,
            "% прибыли в выручке": profit_pct_price,

            "Кол-во недель": weeks,
            "Прибыль средненедельная": profit_week,
        }

    @st.cache_data(ttl=3600)
    def load_ops_month(year: int, month: int):
        d1, d2 = month_start_end(year, month)
        try:
            return client.fetch_finance_transactions(d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d")), "", ""
        except Exception as e:
            title, details = _humanize_ozon_error(e)
            return [], title, details

    month_rows = []
    progress = st.progress(0, text="Считаю месяцы…")
    for i, ym in enumerate(months, start=1):
        y, mo = map(int, ym.split("-"))
        ops_m, ops_m_err_title, ops_m_err_details = load_ops_month(y, mo)
        if ops_m_err_title:
            _block_with_retry(ops_m_err_title, ops_m_err_details, cache_clear_fn=load_ops_month.clear)
        df_ops_m = ops_to_df(ops_m)
        met = month_metrics(df_ops_m)
        # Опер. расходы за месяц (из таба "Опер. расходы")
        d1_m, d2_m = month_start_end(y, mo)
        met["Операционные расходы"] = opex_sum_period(df_opex, d1_m, d2_m)
        met["YM"] = ym
        month_rows.append(met)
        progress.progress(i / len(months), text=f"Считаю месяцы… {i}/{len(months)}")
    progress.empty()

    df_month = pd.DataFrame(month_rows).sort_values(["YM"]).reset_index(drop=True)
    df_month["Месяц"] = df_month["YM"].apply(month_label)

    st.markdown("### Сравнить несколько месяцев")

    month_options = df_month["Месяц"].tolist()
    default_sel = month_options[-3:] if len(month_options) >= 3 else month_options
    key_ms = "months_compare_sel"
    if key_ms not in st.session_state:
        st.session_state[key_ms] = default_sel

    b1, b2, _ = st.columns([1.4, 1.1, 3.5])
    with b1:
        if st.button("Выбрать все месяцы", use_container_width=True, key="btn_months_all"):
            st.session_state[key_ms] = month_options[:]
    with b2:
        if st.button("Снять выбор", use_container_width=True, key="btn_months_clear"):
            st.session_state[key_ms] = []

    sel_months = st.multiselect("Выберите месяцы для сравнения", options=month_options, key=key_ms)

    df_view = df_month.copy()
    if sel_months:
        df_view = df_view[df_view["Месяц"].isin(sel_months)].copy()

    metric_cols = [c for c in df_view.columns if c not in ("YM", "Месяц")]
    pivot = (
        df_view.set_index("Месяц")[metric_cols]
        .T
        .reset_index()
        .rename(columns={"index": "Показатель"})
    )

    percent_metrics = {
        "% выкупа",
        "% комиссии от выручки",
        "% Логистики",
        "% хранения",
        "% отзывов за баллы",
        "% прочих расходов",
        "Доля Ozon, %",
        "% прибыли в выручке",
    }
    int_metrics = {"Кол-во заказов", "Выкуплено шт."}

    def format_row(row):
        name = row["Показатель"]
        for col in row.index:
            if col == "Показатель":
                continue
            v = row[col]
            if name in percent_metrics:
                row[col] = fmt_pct2(v, 1)
            elif name in int_metrics:
                row[col] = _fmt_int(v)
            elif name in {"Кол-во недель"}:
                try:
                    row[col] = f"{float(v):.1f}"
                except Exception:
                    row[col] = "0.0"
            else:
                row[col] = fmt_money2(v)
        return row

    pivot_pretty = pivot.apply(format_row, axis=1)
    st.markdown("### Помесячная таблица")
    st.dataframe(
    pivot_pretty,
    use_container_width=True,
    hide_index=True
    )

    def export_monthly_xlsx(df_rows: pd.DataFrame, df_pivot_pretty: pd.DataFrame) -> bytes:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as w:
            df_rows.to_excel(w, index=False, sheet_name="Monthly_rows_raw")
            df_pivot_pretty.to_excel(w, index=False, sheet_name="Monthly_pivot_pretty")
        return bio.getvalue()

    st.download_button(
        "Скачать XLSX (помесячная сводка)",
        data=export_monthly_xlsx(df_view, pivot_pretty),
        file_name="ozon_monthly_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )



# ================== TAB 3 (ABC) ==================
with tab3:
    st.subheader("ABC-анализ магазина")

    import calendar as _cal2
    import io

    def _fmt_int(x):
        try:
            return f"{int(round(float(x))):,}".replace(",", " ")
        except Exception:
            return "0"

    def _fmt_money(x):
        try:
            return f"{float(x):,.0f} ₽".replace(",", " ")
        except Exception:
            return "0 ₽"

    def month_start_end(year: int, month: int):
        start = date(year, month, 1)
        last_day = _cal2.monthrange(year, month)[1]
        return start, date(year, month, last_day)

    def closed_months_until_today():
        y_last, m_last = last_closed_month(date.today())
        out = []
        y = 2020
        m = 1
        while (y < y_last) or (y == y_last and m <= m_last):
            out.append((y, m))
            if m == 12:
                y += 1
                m = 1
            else:
                m += 1
        return out

    def closed_quarters_for_year(year: int, closed_set):
        q_map = {1: [1,2,3], 2: [4,5,6], 3: [7,8,9], 4: [10,11,12]}
        res = []
        for q, mm in q_map.items():
            if all((year, m) in closed_set for m in mm):
                res.append(q)
        return res

    def abc_class_from_metric(df: pd.DataFrame, col: str):
        s = df[col].fillna(0).astype(float)
        total = float(s.sum())
        if total <= 0:
            return pd.Series(["C"] * len(df), index=df.index)
        share = s / total
        cum = share.cumsum()
        return cum.apply(lambda x: "A" if x <= 0.8 else "B" if x <= 0.95 else "C")

    @st.cache_data(ttl=3600)
    def load_ops_month_abc(y: int, m: int):
        d1, d2 = month_start_end(y, m)
        try:
            return client.fetch_finance_transactions(d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d")), "", ""
        except Exception as e:
            title, details = _humanize_ozon_error(e)
            return [], title, details

    closed = closed_months_until_today()
    closed_set = set(closed)
    years = sorted({y for y, _ in closed}, reverse=True)
    if not years:
        st.info("Нет закрытых месяцев.")
        st.stop()

    colA, colB, colC = st.columns([1.1, 1.2, 2.2])
    with colA:
        mode = st.radio("Период", ["Месяцы", "Кварталы"], horizontal=True, key="abc_mode")
    with colB:
        sel_year = st.selectbox("Год", years, index=0, key="abc_year")
    with colC:
        only_profit = st.checkbox("Только прибыльные SKU", value=False, key="abc_only_profit")

    months_in_year = [(y, m) for (y, m) in closed if y == sel_year]
    month_options = [month_name_ru(m) for (y, m) in months_in_year]
    month_map = {month_name_ru(m): (y, m) for (y, m) in months_in_year}

    q_list = closed_quarters_for_year(sel_year, closed_set)
    q_options = [f"{q} кв." for q in q_list]
    q_to_months = {
        f"{q} кв.": [(sel_year, mm) for mm in ([1,2,3] if q==1 else [4,5,6] if q==2 else [7,8,9] if q==3 else [10,11,12])]
        for q in q_list
    }

    selected_months = []
    chosen_labels = []
    chosen_q = []

    if mode == "Месяцы":
        if not month_options:
            st.info("За выбранный год нет закрытых месяцев.")
            st.stop()

        key_ms = "abc_months_sel"
        if key_ms not in st.session_state:
            st.session_state[key_ms] = month_options[:]

        b1, b2, _ = st.columns([1.6, 1.2, 3.2])
        with b1:
            if st.button("Выбрать все закрытые месяцы", use_container_width=True, key="abc_btn_all_m"):
                st.session_state[key_ms] = month_options[:]
        with b2:
            if st.button("Снять выбор", use_container_width=True, key="abc_btn_clear_m"):
                st.session_state[key_ms] = []

        chosen_labels = st.multiselect("Месяцы (закрытые)", options=month_options, key=key_ms)
        selected_months = [month_map[x] for x in chosen_labels if x in month_map]
    else:
        if not q_options:
            st.info("За выбранный год нет закрытых кварталов.")
            st.stop()

        key_q = "abc_quarters_sel"
        if key_q not in st.session_state:
            st.session_state[key_q] = q_options[:]

        b1, b2, _ = st.columns([1.6, 1.2, 3.2])
        with b1:
            if st.button("Выбрать все закрытые кварталы", use_container_width=True, key="abc_btn_all_q"):
                st.session_state[key_q] = q_options[:]
        with b2:
            if st.button("Снять выбор", use_container_width=True, key="abc_btn_clear_q"):
                st.session_state[key_q] = []

        chosen_q = st.multiselect("Кварталы (закрытые)", options=q_options, key=key_q)
        months_flat = []
        for qlbl in chosen_q:
            months_flat.extend(q_to_months.get(qlbl, []))
        selected_months = sorted(set(months_flat), key=lambda x: (x[0], x[1]))

    if not selected_months:
        st.warning("Выбери хотя бы один месяц/квартал.")
        st.stop()

    dfs = []
    p = st.progress(0, text="Загружаю операции по месяцам…")
    for i, (yy, mm) in enumerate(selected_months, 1):
        _ops_m, _err_t, _err_d = load_ops_month_abc(yy, mm)
        if _err_t:
            _block_with_retry(_err_t, _err_d, cache_clear_fn=load_ops_month_abc.clear)
        dfs.append(ops_to_df(_ops_m))
        p.progress(i / len(selected_months), text=f"Загружаю операции… {i}/{len(selected_months)}")
    p.empty()

    # ================== ABC: считаем "почти идеальную" прибыль как в TAB1 ==================
    # Собираем все операции за выбранные месяцы
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Период ABC (нужен для рекламы и оpex)
    d_from_abc = None
    d_to_abc = None
    for (yy, mm) in selected_months:
        a, b = month_start_end(yy, mm)
        d_from_abc = a if d_from_abc is None else min(d_from_abc, a)
        d_to_abc = b if d_to_abc is None else max(d_to_abc, b)
    if d_from_abc is None or d_to_abc is None:
        st.info("Не удалось определить период ABC.")
        st.stop()

    # 1) Берём ту же SKU-таблицу, что и в TAB1 (там уже есть комиссии/логистика/COGS/артикулы)
    analytics_abc_df, _, _ = load_analytics_range(d_from_abc.strftime("%Y-%m-%d"), d_to_abc.strftime("%Y-%m-%d"))
    sold_abc = build_sold_sku_table(df, cogs_df, analytics_abc_df)
    if sold_abc is None or sold_abc.empty:
        st.info("Нет операций со SKU за выбранный период.")
        st.stop()

    # 2) Налог 6% распределяем пропорционально выручке SKU (как в TAB1)
    total_tax_abc = float(sold_abc["accruals_net"].sum()) * 0.06
    sold_abc = allocate_tax_by_share(sold_abc, total_tax_abc)

    # 3) Реклама: используем ТУ ЖЕ логику, что и в TAB1
    # (берём общий расход на рекламу из Performance summary и распределяем по выручке SKU)
    ads_alloc_abc = load_ads_spend_by_article(
        d_from_abc.strftime("%Y-%m-%d"),
        d_to_abc.strftime("%Y-%m-%d"),
    )
    sold_abc = allocate_ads_by_article(sold_abc, ads_alloc_abc.get("by_article", {}))

    # 4) Опер. расходы распределяем пропорционально выручке SKU (как в TAB1)
    opex_period_abc = opex_sum_period(df_opex, d_from_abc, d_to_abc)
    sold_abc = allocate_cost_by_share(sold_abc, opex_period_abc, "opex_total")

    # 5) Финальная прибыль (та же формула, что и в TAB1)
    sold_abc = compute_profitability(sold_abc)

    # Приводим к формату, который дальше использует ABC (qty/accruals/profit)
    g = sold_abc.rename(columns={
        "qty_buyout": "buyout_qty",
        "accruals_net": "accruals",
    }).copy()

    # страховка типов
    g["buyout_qty"] = pd.to_numeric(g.get("buyout_qty", 0), errors="coerce").fillna(0).astype(int)
    g["accruals"] = pd.to_numeric(g.get("accruals", 0.0), errors="coerce").fillna(0.0)
    g["profit"] = pd.to_numeric(g.get("profit", 0.0), errors="coerce").fillna(0.0)

    # в TAB3 ниже используется name, sku, article
    if "name" not in g.columns:
        g["name"] = ""
    if "article" not in g.columns:
        g["article"] = ""
    g["article"] = g["article"].fillna("").astype(str)

    if only_profit:
        g = g[g["profit"] > 0].copy()

    if g.empty:
        st.info("После фильтров не осталось SKU.")
        st.stop()

    g_buy = g.sort_values("buyout_qty", ascending=False).copy()
    g_buy["grp_buyout"] = abc_class_from_metric(g_buy, "buyout_qty")

    g_turn = g.sort_values("accruals", ascending=False).copy()
    g_turn["grp_turn"] = abc_class_from_metric(g_turn, "accruals")

    g_prof = g.sort_values("profit", ascending=False).copy()
    g_prof["grp_profit"] = abc_class_from_metric(g_prof, "profit")

    g_res = g.merge(g_buy[["sku","grp_buyout"]], on="sku", how="left")
    g_res = g_res.merge(g_turn[["sku","grp_turn"]], on="sku", how="left")
    g_res = g_res.merge(g_prof[["sku","grp_profit"]], on="sku", how="left")
    g_res["ИТОГО"] = g_res["grp_buyout"].fillna("C") + g_res["grp_turn"].fillna("C") + g_res["grp_profit"].fillna("C")

    view = g_res.rename(columns={
        "sku": "SKU",
        "article": "Артикул",
        "name": "Товар",
        "buyout_qty": "Выкуплено, шт",
        "accruals": "Оборот, ₽",
        "profit": "Прибыль, ₽",
        "grp_buyout": "Группа по выкупу",
        "grp_turn": "Группа по обороту",
        "grp_profit": "Группа прибыль"
    }).copy()

    # экспорт (сырые числа)
    export_df = view.copy()
    for col in ["Выкуплено, шт", "Оборот", "Прибыль"]:
        if col in export_df.columns:
            export_df[col] = pd.to_numeric(export_df[col], errors="coerce")
    if "SKU" in export_df.columns:
        export_df["SKU"] = pd.to_numeric(export_df["SKU"], errors="coerce").astype("Int64")

    # отображение (оставляем числа для корректной сортировки)
    # кол-во: int, деньги: float
    if "Выкуплено, шт" in view.columns:
        view["Выкуплено, шт"] = pd.to_numeric(view["Выкуплено, шт"], errors="coerce").fillna(0).astype(int)
    if "Оборот, ₽" in view.columns:
        view["Оборот, ₽"] = pd.to_numeric(view["Оборот, ₽"], errors="coerce").fillna(0.0)
    if "Прибыль, ₽" in view.columns:
        view["Прибыль, ₽"] = pd.to_numeric(view["Прибыль, ₽"], errors="coerce").fillna(0.0)

    # перед отображением (чтобы сортировка была корректной)
    view["SKU"] = pd.to_numeric(view["SKU"], errors="coerce").fillna(0).astype(int)

    st.dataframe(
        view[[
            "Артикул", "SKU", "Товар",
            "Выкуплено, шт", "Группа по выкупу",
            "Оборот, ₽", "Группа по обороту",
            "Прибыль, ₽", "Группа прибыль", "ИТОГО"
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "SKU": st.column_config.NumberColumn(format="%.0f"),  # 👈 ключевое: без разделителей
            "Выкуплено, шт": st.column_config.NumberColumn(format="%.0f"),
            "Оборот, ₽": st.column_config.NumberColumn(format="%.0f"),
            "Прибыль, ₽": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    # ===== Кнопка скачивания XLSX (снизу слева) =====
    def _period_label(mode, chosen_labels, chosen_q, sel_year):
        if mode == "Месяцы":
            return f"{sel_year}_" + "-".join(chosen_labels) if chosen_labels else str(sel_year)
        else:
            return f"{sel_year}_" + "-".join(chosen_q) if chosen_q else str(sel_year)

    period_label = _period_label(mode, chosen_labels, chosen_q, sel_year)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="ABC")
    buf.seek(0)

    btn_col, _ = st.columns([1, 6])
    with btn_col:
        st.download_button(
            label="Скачать XLSX",
            data=buf,
            file_name=f"ABC_{period_label}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
