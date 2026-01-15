from dotenv import load_dotenv
load_dotenv()
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import pandas as pd
import streamlit as st
import yfinance as yf
from openai import OpenAI
import xml.etree.ElementTree as ET
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from io import BytesIO


def fmt_ccy(x: float, symbol: str) -> str:
    try:
        return f"{symbol}{x:,.2f}"
    except Exception:
        return f"{symbol}{x}"

# ---------- Setup ----------
now_ch = datetime.now(ZoneInfo("Europe/Zurich"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide")
st.title("🧮 Cocoa Trade Assistant — Margin Calculator")
st.write("Calculate trade margin from costs (manual inputs).")
BASE_CCY = "GBP"
BASE_SYMBOL = "£"

# --- LAYOUT ---
spacer, COL_IN, COL_OUT = st.columns([0.12, 0.58, 0.30])  # left spacer, middle inputs, right results

# Aliasy na kolumny
class _Panel:
    def __init__(self, col): self.col = col
    def __getattr__(self, name): return getattr(self.col, name)

IN  = _Panel(COL_IN)   # wejścia (środek)
OUT = _Panel(COL_OUT)  # wyniki (prawa)

# --- Większe widgety ---
st.markdown("""
<style>
label, .stMarkdown p { font-size: 1.05rem; }
div[data-baseweb="select"] > div { min-height: 48px; }
input[type="number"], input[type="text"] { height: 48px; font-size: 1.05rem; }
button[kind="primary"], .stButton button { height: 44px; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

ICE_XTICK_URL = os.getenv("ICE_XTICK_URL", "")
ICE_USERNAME  = os.getenv("ICE_USERNAME", "")
ICE_PASSWORD  = os.getenv("ICE_PASSWORD", "")

def _ice_ok() -> bool:
    return bool(ICE_XTICK_URL and ICE_USERNAME and ICE_PASSWORD)

@st.cache_data(ttl=15, show_spinner=False)
def fetch_ice_last_close(symbol: str) -> float | None:
    """
    Fetch latest 5-min bar close for a given ICE symbol via xtick.
    Returns float close or None.
    """
    if not _ice_ok():
        return None

    r = requests.get(
        ICE_XTICK_URL,
        params={
            "username": ICE_USERNAME,
            "pwd": ICE_PASSWORD,
            "symbol": symbol,
            "period": "i5",
            "options.nbars": "1",
        },
        timeout=20,
    )
    r.raise_for_status()
    xml_text = (r.text or "").strip()
    if not xml_text:
        return None
    if "status=" in xml_text and "not entitled" in xml_text.lower():
        return None

    root = ET.fromstring(xml_text)
    bar = root.find(".//bar")
    if bar is None:
        return None
    close_txt = bar.findtext("close")
    if close_txt is None:
        return None

    val = pd.to_numeric(close_txt, errors="coerce")
    return None if pd.isna(val) else float(val)

# ---------- FX helpers ----------
def get_fx_rate(pair: str):
    ticker = yf.Ticker(pair)
    data = ticker.history(period="1d")
    if not data.empty:
        return round(float(data["Close"].iloc[-1]), 4)
    return None

def to_base(amount: float, ccy: str) -> float:
    """Convert amount from ccy to GBP."""
    c = (ccy or BASE_CCY).upper()
    if c == "GBP":
        return amount
    if c == "EUR":
        return amount * eur_gbp_rate
    if c == "USD":
        return amount * usd_gbp_rate
    return amount  # fallback: treat unknown as already GBP

# Live rates (with fallbacks)
eur_usd_rate = get_fx_rate("EURUSD=X") or 1.08   # EUR → USD
usd_eur_rate = get_fx_rate("USDEUR=X") or 0.93   # USD → EUR
gbp_eur_rate = get_fx_rate("GBPEUR=X") or 1.17   # GBP → EUR
eur_gbp_rate = get_fx_rate("EURGBP=X") or 0.85   # EUR → GBP
usd_gbp_rate = get_fx_rate("USDGBP=X") or 0.79   # USD → GBP
gbp_usd_rate = get_fx_rate("GBPUSD=X") or (1.0 / usd_gbp_rate)      # GBP → USD

def choose_trade_fx(buy_ccy: str, sell_ccy: str):
    """Pick FX rate + label for summary."""
    buy_ccy = (buy_ccy or "GBP").upper()
    sell_ccy = (sell_ccy or "GBP").upper()
    if "USD" in (buy_ccy, sell_ccy):
        return usd_gbp_rate, "USD→GBP"
    if "EUR" in (buy_ccy, sell_ccy):
        return eur_gbp_rate, "EUR→GBP"
    return 1.0, "GBP"

# ---------- Input widgets helpers ----------
def money_input_gbp(label: str, default: float = 0.0, default_ccy: str = "GBP"):
    c1, c2 = IN.columns([2, 1])
    amt = c1.number_input(
        f"{label} amount",
        min_value=0.0,
        value=float(default),
        step=10.0,
        format="%.2f",
        key=f"{label}_amt",
    )
    ccy = c2.selectbox(
        f"{label} currency",
        ["GBP", "EUR", "USD"],
        index=["GBP", "EUR", "USD"].index(default_ccy),
        key=f"{label}_ccy",
    )
    if ccy == "EUR":
        return amt * eur_gbp_rate
    if ccy == "USD":
        return amt * usd_gbp_rate
    return amt  # already GBP

def percent_cost_from_buy(label: str, default_pct: float = 0.0):
    """Input for % of buy price (middle column)."""
    return IN.number_input(
        f"{label} (% of buy price)",
        min_value=0.0,
        value=float(default_pct),
        step=0.1,
        format="%.2f",
        key=f"{label}_pct",
    )

# ---------- Trade parameters ----------
pol_options = [
    "POL", "ABIDJAN", "TIN CAN", "APAPA", "CALLAO", "CONAKRY", "DIEGO SUAREZ", "DOUALA",
    "FREETOWN", "KAMPALA", "KRIBI", "LEKKI", "LOME", "MATADI", "MOMBASA", "MONROVIA",
    "NOSY BE", "SAN PEDRO", "TAKORADI", "TEMA", "CARTAGENA", "GUAYAQUIL", "POSORJA",
    "PAITA", "CAUCEDO", "ANTWERP", "KINSHASA", "LAGOS", "PISCO"
]
destination_options = [
    "ANTWERP", "BARCELONA", "AMSTERDAM", "HAMBURG", "ISTANBUL", "ROTTERDAM", "VALENCIA",
    "BATAM", "PASIR GUDANG", "SURABAYA", "PTP", "PHILADELPHIA", "SZCZECIN", "WELLINGTON",
    "AMBARLI", "GENOA", "VADO LIGURE", "SINGAPORE", "TALLINN", "JAKARTA", "PORT KLANG",
    "NEW YORK", "MONTREAL", "PIRAEUS", "YOKOHAMA", "VALENCIA", "BATAM VIA SINGAPORE",
    "SHANGHAI", "KLAIPEDA", "LIVERPOOL"
]
carrier_options = [
    "ARKAS","ONE","CMA","HAPAG","MAERSK","MSC","OOCL","STS_MSC","STS_GRIMALDI","STS_OOCL","STS_HAPAG-LLOYD",
    "STS_ONE","STS_PIL","STS_MESSINA","STS_CMA CGM","STS_MAERSK","PIL"
]
warehouse_options = [
    "STEINWEG AMSTERDAM", "COMMODITY CENTRE AMST", "COTTERELL AMST", "STEINWEG ANTWERP",
    "COMMODITY CENTRE ANTW", "KTN ANTWERP", "CWT ANTWERP", "VOLLERS ANTWERP", "DURME NATIE ANTWERP",
    "COMMODITY CENTRE UK FOR ICE", "VOLLERS HAMBURG", "COTTERELL HAMBURG", "QUAST & CONS HAMBURG",
    "FERRERO", "VILLARS", "ION SA", "GRAND CANDY", "DDS", "NONE"
]

IN.title("📦 Trade Parameters")
IN.markdown("## 🧾 Manual Cost Inputs (per ton)")

volume = IN.number_input("Volume (tons)", min_value=1, value=1)
buy_price = IN.number_input("Buy Price", value=7500.0, step=10.0, format="%.2f")
buy_currency = IN.selectbox("Buy Price Currency", ["GBP", "EUR", "USD"], index=0)
# Set base currency symbol dynamically
currency_symbols = {"EUR": "€", "USD": "$", "GBP": "£"}
base_currency_symbol = currency_symbols.get(buy_currency, "£")


# --- ICE London contract selectors ---
COCOA_DELIVERY_MONTHS = [("Mar","H"), ("May","K"), ("Jul","N"), ("Sep","U"), ("Dec","Z")]

IN.markdown("### ICE London contract")
ice_month_name = IN.selectbox(
    "Delivery month",
    [n for n, _ in COCOA_DELIVERY_MONTHS],
    index=0,
    key="ice_london_month"
)
ice_year_full = IN.number_input(
    "Delivery year (YYYY)",
    min_value=2024, max_value=2035,
    value=datetime.now().year, step=1,
    key="ice_london_year"
)
ice_month_code = dict(COCOA_DELIVERY_MONTHS)[ice_month_name]
IN.caption(f"Selected: C {ice_year_full % 100:02d}{ice_month_code}-ICE")

# ---- Use ICE London from API ----
use_ice_london = IN.toggle(
    "Use ICE London futures (LIVE from ICE)",
    value=False,
    help="Pulls latest 5-min close from ICE xtick connection."
)


if use_ice_london:
    yy = ice_year_full % 100
    target_symbol = f"C {yy:02d}{ice_month_code}-ICE"  # your London cocoa format

    try:
        last_val = fetch_ice_last_close(target_symbol)
        if last_val is None:
            IN.warning(f"No ICE value for '{target_symbol}' (not entitled / no data). Using manual Buy Price.")
        else:
            buy_price = last_val
            buy_currency = "GBP"
            base_currency_symbol = "£"
            st.session_state["ice_benchmark_gbp"] = last_val
            IN.caption(f"{target_symbol}: £{last_val:,.2f}/t (ICE live)")
    except Exception as e:
        IN.error(f"ICE fetch failed: {e}. Using manual Buy Price.")


# Convert buy price to GBP for all subsequent calc
if buy_currency == "EUR":
    buy_price *= eur_gbp_rate
elif buy_currency == "USD":
    buy_price *= usd_gbp_rate

# 👉 Buying Diff is part of the base price, not a cost
buying_diff = IN.number_input("Buying Diff (GBP per ton)", value=0, step=1)
base_buy = buy_price
base_buy_incl = buy_price + buying_diff

port = IN.selectbox("Port of Loading (POL)", sorted(pol_options))
destination = IN.selectbox("Destination", sorted(destination_options))
container_size = IN.selectbox("Container Size", ["20", "40"], index=1)
carrier = IN.selectbox("Shipping Line (optional)", ["Auto (priciest)"] + sorted(carrier_options))
selected_carrier = None if carrier == "Auto (priciest)" else carrier

selected_warehouse = IN.selectbox("Warehouse", sorted(warehouse_options))
rent_months = IN.number_input(
    "Warehouse rent (months)",
    min_value=0, value=1, step=1,
    key="warehouse_rent_months",
    help="Multiply the 'WAREHOUSE RENT' line in the Excel by this many months."
)

payment_days = IN.number_input("Payment Terms (days)", min_value=0, value=30, step=1)
if payment_days > 0:
    annual_rate = IN.number_input("Annual Financing Rate (%)", min_value=0.0, value=8.5, step=0.5) / 100
else:
    annual_rate = 0.0

sell_currency = IN.selectbox("Sell Price Currency", ["GBP", "EUR", "USD"], index=0)

IN.caption(f"Rates pulled: {datetime.now(ZoneInfo('Europe/Zurich')).strftime('%Y-%m-%d %H:%M:%S')}")
IN.markdown("### 💱 FX Rates (Live)")
IN.markdown(f"- **EUR/USD**: {eur_usd_rate}")
IN.markdown(f"- **USD/EUR**: {usd_eur_rate}")
IN.markdown(f"- **GBP/EUR**: {gbp_eur_rate}")
IN.markdown(f"- **GBP/USD**: {gbp_usd_rate}")

# --- SELL PRICE (default GBP, user can change) ---
sell_currency_input = IN.selectbox("Sell Price Currency (entry)", ["GBP", "EUR", "USD"], index=0, key="sell_ccy")
sell_price_input = IN.number_input(
    f"Sell Price ({sell_currency_input} per ton)",
    min_value=0.0, value=8500.0, step=10.0, format="%.2f", key="sell_price_input"
)

# Normalize to GBP exactly once for calculations
sell_price = to_base(sell_price_input, sell_currency_input)   # -> GBP
sell_currency = "GBP"  # keep downstream logic in GBP

# UI hint
IN.caption(
    f"Sell normalized to GBP: £{sell_price:,.2f}/t "
    f"(entered {sell_currency_input} {sell_price_input:,.2f}/t)"
)

# FX context
trade_fx_rate, trade_fx_label = choose_trade_fx(buy_currency, sell_currency_input)
trade_fx_rate, trade_fx_label = choose_trade_fx(buy_currency, sell_currency)

# ---------- Manual costs ----------
lid_yes = IN.checkbox("LID applies?", value=False)
lid_gbp = 400.0 if lid_yes else 0.0

cert_premium_gbp = money_input_gbp("CERT PREMIUM")

docs_pct = IN.number_input("DOCS COSTS (% of base buy)", min_value=0.0, value=0.2, step=0.1, format="%.2f")
docs_costs_gbp = (docs_pct / 100.0) * base_buy
IN.caption(f"Docs Costs = {docs_pct:.2f}% of base → {BASE_SYMBOL}{docs_costs_gbp:.2f}/t")

quality_claim_gbp = money_input_gbp("QUALITY CLAIM", default=50.0, default_ccy="GBP")

sel_ccy = st.session_state.get("QUALITY CLAIM_ccy", "GBP")
raw_amt = st.session_state.get("QUALITY CLAIM_amt", 50.0)
IN.caption(f"QC entered: {currency_symbols.get(sel_ccy,'£')}{raw_amt:,.2f}/t → stored as £{quality_claim_gbp:,.2f}/t")

wl_pct = percent_cost_from_buy("WEIGHT LOSS", default_pct=0.5)
weight_loss_gbp = (wl_pct / 100.0) * base_buy

qc_dep_gbp = money_input_gbp("QUALITY CONTROLE DEP", default=2.0, default_ccy=buy_currency)
qc_arr_gbp = money_input_gbp("QUALITY CONTROLE ARR", default=2.0, default_ccy=buy_currency)
origin_agent_gbp = money_input_gbp("ORIGIN AGENT")
dest_agent_gbp   = money_input_gbp("DESTINATION AGENT")

# FREIGHT — manual override or auto
use_manual_freight = IN.checkbox("Enter FREIGHT manually (override route table)?", value=False)
freight_per_ton = None
if use_manual_freight:
    freight_per_ton = money_input_gbp("FREIGHT")

# --- EUDR / TRACEABILITY FEES (GBP per ton) ---
eudr_fee_gbp = money_input_gbp("EUDR / TRACEABILITY FEES", default=50.0, default_ccy=buy_currency)

# --- DRESSING (Excel or manual) ---
dressing_excel = "Dressing.xlsx"
dressing_gbp = 0.0

if os.path.exists(dressing_excel):
    ddf = pd.read_excel(dressing_excel)
    ddf.columns = [str(c).strip().title() for c in ddf.columns]
    required_cols = {"Dressing", "Value", "Currency"}
    if required_cols.issubset(set(ddf.columns)):
        ddf["Currency"] = ddf["Currency"].astype(str).str.upper()
        ddf["Dressing"] = ddf["Dressing"].astype(str)
        dressing_options = ddf["Dressing"].tolist()
        chosen_dressing = IN.selectbox("Dressing (from table)", dressing_options, index=0)
        sel_row = ddf.loc[ddf["Dressing"] == chosen_dressing].iloc[0]
        raw_value = float(sel_row["Value"])
        raw_ccy = str(sel_row["Currency"])
        dressing_gbp = to_base(raw_value, raw_ccy)
        IN.caption(f"Selected '{chosen_dressing}': {BASE_SYMBOL}{dressing_gbp:.2f} per ton (auto)")
    else:
        IN.warning("Dressing file missing required columns (Dressing, Value, Currency). Using manual input.")
        dressing_gbp = money_input_gbp("DRESSING")
else:
    IN.warning("Dressing cost file not found (dressing_costs.xlsx). Using manual input.")
    dressing_gbp = money_input_gbp("DRESSING")

freight_correction_gbp  = money_input_gbp("FREIGHT CORRECTION")

# --- MARINE INSURANCE (Excel % of base buy) ---
marine_insurance_file = "Marine_insurance.xlsx"
marine_insurance_gbp = 0.0

if os.path.exists(marine_insurance_file):
    midf = pd.read_excel(marine_insurance_file)
    midf.columns = [str(c).strip().title() for c in midf.columns]
    required_cols = {"Marine Insurance", "Value", "Type"}
    if required_cols.issubset(set(midf.columns)):
        midf["Marine Insurance"] = midf["Marine Insurance"].astype(str)
        midf["Type"] = midf["Type"].astype(str).str.title()
        midf["Value"] = pd.to_numeric(midf["Value"], errors="coerce")
        midf = midf[(midf["Type"] == "Percentage") & midf["Value"].notna()].copy()
        if not midf.empty:
            mi_options = midf["Marine Insurance"].tolist()
            chosen_mi = IN.selectbox("Marine Insurance (from table)", mi_options, index=0)
            sel = midf.loc[midf["Marine Insurance"] == chosen_mi].iloc[0]
            pct = float(sel["Value"])
            marine_insurance_gbp = (pct / 100.0) * base_buy
            IN.caption(f"Selected '{chosen_mi}': {pct:.2f}% of base buy → {BASE_SYMBOL}{marine_insurance_gbp:.2f}/t")
        else:
            IN.warning("Marine insurance table has no valid 'Percentage' rows. Using manual input.")
            marine_insurance_gbp = money_input_gbp("MARINE INSURANCE")
    else:
        IN.warning("Marine insurance file missing required columns (Marine Insurance, Value, Type). Using manual input.")
        marine_insurance_gbp = money_input_gbp("MARINE INSURANCE")
else:
    IN.warning("Marine insurance file not found (marine_insurance.xlsx). Using manual input.")
    marine_insurance_gbp = money_input_gbp("MARINE INSURANCE")

# --- STOCK INSURANCE (percentage × months) ---
stock_ins_pct = IN.number_input(
    "Stock insurance (% of base buy, per month)",
    min_value=0.0, value=0.10, step=0.10, format="%.2f", key="stock_ins_pct"
)
stock_ins_months = IN.number_input(
    "Stock insurance months",
    min_value=0, value=int(rent_months), step=1, key="stock_ins_months"
)
stock_insurance_gbp = (stock_ins_pct / 100.0) * base_buy * stock_ins_months
IN.caption(
    f"Stock insurance = {stock_ins_pct:.2f}% × base £{base_buy:,.2f} × {stock_ins_months} mo "
    f"= £{stock_insurance_gbp:.2f}/t"
)

# ---------- Freight route table (optional) ----------
freight_costs = {}
warehouse_costs = None

excel_path = "logistics_freight_trade_calc.xlsx"
freight_costs = {}

if os.path.exists(excel_path):
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = [str(c).strip().upper() for c in df_excel.columns]
    if "CONTAINER" in df_excel.columns:
        df_excel["CONTAINER"] = df_excel["CONTAINER"].astype(str).str.strip()
        df_excel = df_excel[df_excel["CONTAINER"] == container_size].copy()
    if "CURRENCY" in df_excel.columns:
        df_excel["CURRENCY"] = df_excel["CURRENCY"].astype(str).str.upper()
    df_excel["ALL_IN"] = pd.to_numeric(df_excel.get("ALL_IN"), errors="coerce")
    df_excel.loc[df_excel["CURRENCY"] == "EUR", "ALL_IN"] *= eur_gbp_rate
    df_excel.loc[df_excel["CURRENCY"] == "USD", "ALL_IN"] *= usd_gbp_rate
    df_excel = df_excel.dropna(subset=["POL", "POD", "SHIPPING LINE", "ALL_IN"])
    df_excel["POL"] = df_excel["POL"].astype(str).str.strip().str.upper()
    df_excel["POD"] = df_excel["POD"].astype(str).str.strip().str.upper()
    df_excel["SHIPPING LINE"] = df_excel["SHIPPING LINE"].astype(str).str.strip().str.upper()

    for _, row in df_excel.iterrows():
        key = (container_size, row["POL"], row["POD"])
        carrier_name = row["SHIPPING LINE"]
        cost_gbp = float(row["ALL_IN"])
        freight_costs.setdefault(key, {})[carrier_name] = cost_gbp
else:
    IN.warning("Freight file not found: logistics_freight_trade_calc.xlsx")

def get_freight_per_ton(
    container: str,
    port_from: str,
    port_to: str,
    total_volume: float,
    selected_carrier: str | None = None,
    auto_mode: str = "priciest",
):
    """Return freight in GBP per ton."""
    key = (container, port_from, port_to)
    if key not in freight_costs:
        IN.error(f"No freight data for {container}' {port_from} → {port_to}")
        return None
    costs = freight_costs[key]
    if selected_carrier and selected_carrier in costs:
        per_container = costs[selected_carrier]
    else:
        chooser = max if auto_mode == "priciest" else min
        per_container = chooser(costs.values())
    tons_per_container = 20.0 if container == "20" else 40.0
    per_ton = per_container / tons_per_container
    return round(per_ton, 2)

# Auto-freight (if not manual)
if not use_manual_freight:
    auto_freight = get_freight_per_ton(
        container=container_size,
        port_from=port,
        port_to=destination,
        total_volume=volume,
        selected_carrier=selected_carrier,
        auto_mode="priciest"
    )
    if auto_freight is not None:
        freight_per_ton = auto_freight

# ---------- Warehouse costs (GBP/t; rent multiplied by months) ----------
warehouse_total_per_ton = 0.0
warehouse_excel_path = "warehouse_costs.xlsx"

use_manual_wh = IN.checkbox(
    "Enter WAREHOUSE costs manually (override Excel)?",
    value=False,
    help="If ON, edit the table below. Excel stays unchanged. 'WAREHOUSE RENT' is still multiplied by the selected months."
)

RENT_ALIASES = {"WAREHOUSE RENT", "RENT", "STORAGE RENT"}

def _load_excel_series_raw(warehouse_name: str) -> pd.Series | None:
    """Raw Excel series (rent is per-month)."""
    if not os.path.exists(warehouse_excel_path):
        IN.warning("Warehouse cost file not found: warehouse_costs.xlsx")
        return None
    df = pd.read_excel(warehouse_excel_path, index_col=0)
    df.columns = [str(c).strip() for c in df.columns]
    df.index = df.index.map(lambda x: str(x).strip().upper())
    if warehouse_name not in df.columns:
        IN.warning(f"No cost data found for selected warehouse: {warehouse_name}")
        return None
    return pd.to_numeric(df[warehouse_name].dropna(), errors="coerce").fillna(0.0).astype(float)

def _apply_rent_months(s: pd.Series) -> pd.Series:
    """Multiply rent-like row by rent_months; other rows unchanged."""
    s = s.copy()
    rk = next((k for k in RENT_ALIASES if k in s.index), None)
    if rk:
        base_val = float(s.loc[rk])
        s.loc[rk] = base_val * rent_months
        IN.caption(
            f"Warehouse rent: {base_currency_symbol}{base_val:.2f}/t × {rent_months} mo = "
            f"{base_currency_symbol}{base_val*rent_months:.2f}/t"
        )
    else:
        IN.caption("No 'WAREHOUSE RENT' row found; no monthly multiplier applied.")
    return s

excel_series_raw = _load_excel_series_raw(selected_warehouse)

# Session cache for manual editor
if "wh_manual_cache" not in st.session_state:
    st.session_state.wh_manual_cache = {}
if st.session_state.get("wh_manual_last_wh") != selected_warehouse:
    st.session_state.wh_manual_last_wh = selected_warehouse
    init_df = pd.DataFrame({"Cost Item": [], "GBP/ton (per unit)": []})
    if excel_series_raw is not None:
        init_df = pd.DataFrame({
            "Cost Item": excel_series_raw.index.tolist(),
            "GBP/ton (per unit)": [float(v) for v in excel_series_raw.values],
        })
    st.session_state.wh_manual_cache[selected_warehouse] = init_df

with OUT.expander("📦 Warehouse Cost Breakdown", expanded=True):
    OUT.write(f"🏷️ Selected Warehouse: **{selected_warehouse}**")

    if not use_manual_wh:
        if excel_series_raw is not None:
            warehouse_costs = _apply_rent_months(excel_series_raw)
            OUT.dataframe(
                warehouse_costs.round(2).to_frame(name=selected_warehouse),
                use_container_width=True
            )
        else:
            warehouse_costs = None
            OUT.info("No detailed cost breakdown available.")
    else:
        edited_df = OUT.data_editor(
            st.session_state.wh_manual_cache[selected_warehouse],
            num_rows="dynamic",
            use_container_width=True,
            key=f"wh_editor_{selected_warehouse}",
            column_config={
                "Cost Item": st.column_config.TextColumn(required=True),
                "GBP/ton (per unit)": st.column_config.NumberColumn(required=True, step=1.0, format="%.2f"),
            }
        )

        edited_df["Cost Item"] = edited_df["Cost Item"].astype(str).str.strip()
        edited_df["GBP/ton (per unit)"] = pd.to_numeric(edited_df["GBP/ton (per unit)"], errors="coerce").fillna(0.0)
        st.session_state.wh_manual_cache[selected_warehouse] = edited_df

        if edited_df.empty:
            warehouse_costs = pd.Series(dtype=float)
        else:
            manual_series = pd.Series(
                edited_df["GBP/ton (per unit)"].values,
                index=edited_df["Cost Item"].str.upper().values,
                dtype=float
            )
            warehouse_costs = _apply_rent_months(manual_series)

    if warehouse_costs is None:
        warehouse_total_per_ton = 0.0
    else:
        s = warehouse_costs.squeeze() if hasattr(warehouse_costs, "squeeze") else warehouse_costs
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        warehouse_total_per_ton = float(s.sum())

    badge = " (manual override)" if use_manual_wh else " (from Excel)"
    OUT.markdown(f"**📦 Warehouse cost per ton{badge}: {base_currency_symbol}{warehouse_total_per_ton:,.2f}**")

# ---------- TRANSPORT (inland) ----------
transport_per_ton_gbp = 0.0
transport_excel = "Transport.xlsx"
use_transport = IN.checkbox("Add TRANSPORT (inland)?", value=False, key="use_transport")

if use_transport:
    if os.path.exists(transport_excel):
        tdf = pd.read_excel(transport_excel)
        tdf.columns = [str(c).strip().upper() for c in tdf.columns]
        for col in ("POL", "POD", "SERVICE PROVIDER"):
            if col in tdf.columns:
                tdf[col] = tdf[col].astype(str).str.strip().str.upper()
        if "RATE" in tdf.columns:
            tdf["RATE"] = pd.to_numeric(tdf["RATE"], errors="coerce")  # EUR per truck

        pol_opts = sorted(tdf["POL"].dropna().unique())
        sel_pol = IN.selectbox("Transport POL", pol_opts, key="t_pol")

        pod_opts = sorted(tdf.loc[tdf["POL"] == sel_pol, "POD"].dropna().unique())
        sel_pod = IN.selectbox("Transport POD", pod_opts, key="t_pod")

        route_df = tdf[(tdf["POL"] == sel_pol) & (tdf["POD"] == sel_pod)].dropna(subset=["RATE"])
        if route_df.empty:
            IN.warning("No transport rate for that POL/POD.")
        else:
            route_df = route_df.sort_values("RATE")
            provider_labels = [
                f'{r["SERVICE PROVIDER"]} — €{r["RATE"]:,.2f}/truck'
                for _, r in route_df.iterrows()
            ]
            sel_idx = IN.selectbox("Service Provider", list(range(len(provider_labels))),
                                   format_func=lambda i: provider_labels[i], key="t_provider_idx")
            selected_row = route_df.iloc[int(sel_idx)]
            rate_eur = float(selected_row["RATE"])
            transport_per_ton_gbp = (rate_eur * eur_gbp_rate) / 24.0
            IN.caption(
                f"Transport: €{rate_eur:,.2f}/truck → £{(rate_eur*eur_gbp_rate):,.2f}/truck → "
                f"£{transport_per_ton_gbp:,.2f}/t (÷24)"
            )
    else:
        IN.warning(f"Transport file not found: {transport_excel}")

# ---------- Manual cost table (EXCLUDES Buying Diff) ----------
manual_rows = [
    {"Cost Item": "LID",                    "GBP/ton": round(lid_gbp, 2)},
    {"Cost Item": "CERT PREMIUM",           "GBP/ton": round(cert_premium_gbp, 2)},
    {"Cost Item": "DOCS COSTS",             "GBP/ton": round(docs_costs_gbp, 2)},
    {"Cost Item": "QUALITY CLAIM",          "GBP/ton": round(quality_claim_gbp, 2)},
    {"Cost Item": "WEIGHT LOSS",            "GBP/ton": round(weight_loss_gbp, 2)},
    {"Cost Item": "QUALITY CONTROLE DEP",   "GBP/ton": round(qc_dep_gbp, 2)},
    {"Cost Item": "QUALITY CONTROLE ARR",   "GBP/ton": round(qc_arr_gbp, 2)},
    {"Cost Item": "ORIGIN AGENT",           "GBP/ton": round(origin_agent_gbp, 2)},
    {"Cost Item": "DESTINATION AGENT",      "GBP/ton": round(dest_agent_gbp, 2)},
    {"Cost Item": "DRESSING",               "GBP/ton": round(dressing_gbp, 2)},
    {"Cost Item": "FREIGHT CORRECTION",     "GBP/ton": round(freight_correction_gbp, 2)},
    {"Cost Item": "MARINE INSURANCE",       "GBP/ton": round(marine_insurance_gbp, 2)},
    {"Cost Item": "EUDR / TRACEABILITY FEES", "GBP/ton": round(eudr_fee_gbp, 2)},
    {"Cost Item": "STOCK INSURANCE",        "GBP/ton": round(stock_insurance_gbp, 2)},
    {"Cost Item": "TRANSPORT (inland)",     "GBP/ton": round(transport_per_ton_gbp, 2)},
]
manual_df = pd.DataFrame(manual_rows)
manual_subtotal = float(manual_df["GBP/ton"].sum())

with OUT.expander("📊 Manual Cost Breakdown (per ton)"):
    OUT.dataframe(manual_df)
    OUT.write(f"🧮 Manual costs subtotal: **{base_currency_symbol}{manual_subtotal:.2f}**")

# Show the base buy clearly
OUT.markdown(f"**Base buy (incl. Buying Diff): {base_currency_symbol}{base_buy_incl:.2f}/t**")
OUT.caption(f"(Buy Price {base_currency_symbol}{buy_price:.2f} + Buying Diff {base_currency_symbol}{buying_diff:.2f})")

# ---------- Base landed cost & financing ----------
pre_finance_cost = (
    base_buy_incl
    + manual_subtotal
    + warehouse_total_per_ton
    + (freight_per_ton or 0.0)
)

if payment_days > 0:
    financing_per_ton = (annual_rate / 365) * payment_days * pre_finance_cost
else:
    financing_per_ton = 0.0

cost_per_ton = pre_finance_cost + financing_per_ton

OUT.write(f"💳 Financing cost per ton: {base_currency_symbol}{financing_per_ton:.2f}")
OUT.write(f"📦 Buy price per ton ({buy_currency}): {base_currency_symbol}{buy_price:.2f}")
OUT.write(f"➕ Buying Diff added to revenue ({buy_currency}): {base_currency_symbol}{buying_diff:.2f}")
OUT.write(f"➡️ Base buy per ton ({buy_currency}): **{base_currency_symbol}{base_buy:.2f}**")
OUT.write(f"🚢 Freight per ton ({buy_currency}): {base_currency_symbol}{(freight_per_ton or 0.0):.2f}")
OUT.write(f"🏭 Warehouse cost per ton ({buy_currency}): {base_currency_symbol}{warehouse_total_per_ton:.2f}")
OUT.write(f"💼 Total landed cost per ton ({buy_currency}): **{base_currency_symbol}{cost_per_ton:.2f}**")

# ---------- Market helper ----------
def get_cocoa_price():
    cocoa = yf.Ticker("CC=F")
    data = cocoa.history(period="1d")
    if not data.empty:
        return round(float(data["Close"].iloc[-1]), 2)
    return None

# ---------- AI commentary ----------
def generate_ai_comment(buy_price, sell_price, freight_cost, cocoa_price, fx_rate, fx_label, margin, mode):
    prompt = f"""
You are a commodity market analyst. Based on the following trade parameters:

- Calculation mode: {mode}
- Purchase price: {buy_price} GBP/ton
- Selling price: {sell_price} GBP/ton
- Freight cost: {freight_cost} GBP/ton
- Cocoa market price: {cocoa_price} GBP/ton
- FX rate ({fx_label}): {fx_rate}
- Calculated margin: {margin:.2f}%

Please provide:
1. An assessment of whether the margin is attractive in the current cocoa market.
2. A short list of potential risks (e.g., FX volatility, supply/demand shifts, freight rate changes).
3. 1 or 2 concise and practical recommendations for the trader based on these figures.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating AI comment: {str(e)}"

# ---------- Results + AI blocks ----------
margin_per_ton = (sell_price or 0.0) - cost_per_ton
total_margin = margin_per_ton * volume

# === LANDED DIFF vs benchmark (ICE/CC=F) ===
benchmark_price_gbp = st.session_state.get("ice_benchmark_gbp")

# Fallback: CC=F (USD/t) → GBP/t
if benchmark_price_gbp is None:
    cc_usd = get_cocoa_price()   # funkcja już jest niżej w pliku – jeśli masz ją niżej, przenieś definicję wyżej lub zostaw ten blok za definicją funkcji
    if cc_usd is not None:
        benchmark_price_gbp = round(cc_usd * usd_gbp_rate, 2)

# policz i pokaż na zielono
if benchmark_price_gbp is not None:
    landed_diff = cost_per_ton - benchmark_price_gbp
    OUT.success(f"Landed diff: {base_currency_symbol}{landed_diff:,.2f}/t")
    # (opcjonalnie, pokaż jaki benchmark został użyty)
    OUT.caption(f"Benchmark used: {base_currency_symbol}{benchmark_price_gbp:,.2f}/t")

# Twoje istniejące zielone boxy:
OUT.success(f"Margin per ton: **{base_currency_symbol}{margin_per_ton:.2f}**")
OUT.success(f"Total margin: **{base_currency_symbol}{total_margin:.2f}**")

margin_percent = ((margin_per_ton / (sell_price or 1.0)) * 100.0) if sell_price else 0.0

with OUT.expander("🧠 AI Analysis"):
    ai_comment = generate_ai_comment(
        buy_price=round((base_buy_incl if 'base_buy_incl' in globals() else (base_buy + buying_diff)), 2),
        sell_price=round(sell_price or 0.0, 2),
        freight_cost=round(freight_per_ton or 0.0, 2),
        cocoa_price=get_cocoa_price(),
        fx_rate=round(trade_fx_rate, 4),
        fx_label=trade_fx_label,
        margin=margin_percent,
        mode="Margin Calculation"
    )
    OUT.markdown(ai_comment)

def build_pdf_report(
    *,
    base_symbol: str,
    buy_price: float,
    buying_diff: float,
    base_buy: float,
    manual_df,  # pandas DataFrame
    manual_subtotal: float,
    freight_per_ton: float,
    warehouse_total_per_ton: float,
    financing_per_ton: float,
    cost_per_ton: float,
    mode_label: str,
    sell_price: float | None,
    target_margin: float | None,
    margin_per_ton: float | None,
    total_margin: float | None,
    required_sell_price: float | None,
    volume: float,
) -> BytesIO:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=12)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceAfter=6)
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("Cocoa Trade Summary", h1))
    story.append(Paragraph(f"Mode: {mode_label}", body))
    story.append(Spacer(1, 6))

    # Cost Summary
    story.append(Paragraph("Cost Summary (per ton)", h2))
    cost_rows = [
        ["Buy price", fmt_ccy(buy_price, base_symbol)],
        ["Buying Diff (added to revenue)", fmt_ccy(buying_diff, base_symbol)],
        ["Base buy", fmt_ccy(base_buy, base_symbol)],
        ["Manual costs subtotal", fmt_ccy(manual_subtotal, base_symbol)],
        ["Freight", fmt_ccy(freight_per_ton or 0.0, base_symbol)],
        ["Warehouse", fmt_ccy(warehouse_total_per_ton, base_symbol)],
        ["Financing", fmt_ccy(financing_per_ton, base_symbol)],
        ["Total landed cost", fmt_ccy(cost_per_ton, base_symbol)],
    ]
    t_cost = Table(cost_rows, colWidths=[200, 200])
    t_cost.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (1,0), (1,-1), "RIGHT"),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
    ]))
    story.append(t_cost)
    story.append(Spacer(1, 10))

    # Manual cost breakdown
    if manual_df is not None and not manual_df.empty:
        story.append(Paragraph("Manual Cost Breakdown (per ton)", h2))
        value_col = manual_df.columns[-1]
        table_data = [["Cost Item", value_col]]
        for _, r in manual_df.iterrows():
            table_data.append([str(r["Cost Item"]), fmt_ccy(float(r[value_col]) if pd.notna(r[value_col]) else 0.0, base_symbol)])
        t_manual = Table(table_data, colWidths=[260, 140])
        t_manual.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("ALIGN", (1,1), (1,-1), "RIGHT"),
        ]))
        story.append(t_manual)
        story.append(Spacer(1, 10))

    # Margin Summary
    story.append(Paragraph("Margin Summary", h2))
    if mode_label == "Margin Calculation":
        ms = [
            ["Sell price", fmt_ccy(sell_price or 0.0, base_symbol)],
            ["Total landed cost", fmt_ccy(cost_per_ton, base_symbol)],
            ["Margin per ton", fmt_ccy(margin_per_ton or 0.0, base_symbol)],
            ["Volume (t)", f"{volume:,.0f}"],
            ["Total margin", fmt_ccy(total_margin or 0.0, base_symbol)],
        ]
    else:
        ms = [
            ["Target margin per ton", fmt_ccy(target_margin or 0.0, base_symbol)],
            ["Total landed cost", fmt_ccy(cost_per_ton, base_symbol)],
            ["Required sell price per ton", fmt_ccy(required_sell_price or 0.0, base_symbol)],
            ["Volume (t)", f"{volume:,.0f}"],
        ]
    t_margin = Table(ms, colWidths=[260, 140])
    t_margin.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (1,0), (1,-1), "RIGHT"),
    ]))
    story.append(t_margin)

    doc.build(story)
    buf.seek(0)
    return buf

# ------- PDF Export -------
default_name = f"cocoa_trade_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
pdf_name = OUT.text_input("📄 PDF file name", value=default_name)

if OUT.button("Generate PDF"):
    mode_label = "Margin Calculation"
    m_pt = (sell_price or 0.0) - cost_per_ton
    t_m  = m_pt * volume

    pdf_buf = build_pdf_report(
        base_symbol=base_currency_symbol if 'base_currency_symbol' in globals() else "£",
        buy_price=buy_price,
        buying_diff=buying_diff,
        base_buy=base_buy_incl,
        manual_df=manual_df,
        manual_subtotal=manual_subtotal,
        freight_per_ton=freight_per_ton or 0.0,
        warehouse_total_per_ton=warehouse_total_per_ton,
        financing_per_ton=financing_per_ton if 'financing_per_ton' in globals() else 0.0,
        cost_per_ton=cost_per_ton,
        mode_label=mode_label,
        sell_price=(sell_price or 0.0),
        target_margin=None,
        margin_per_ton=m_pt,
        total_margin=t_m,
        required_sell_price=None,
        volume=volume,
    )
    # tu możesz zapisać/udostępnić pdf_buf jeśli chcesz
