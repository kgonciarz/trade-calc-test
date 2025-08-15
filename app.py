import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf
from openai import OpenAI

# ---------- Setup ----------
now_ch = datetime.now(ZoneInfo("Europe/Zurich"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide")
st.title("🧮 Cocoa Trade Assistant — Forward & Reverse Margin Calculator")
st.write("Calculate trade margin from costs (manual inputs).")
BASE_CCY = "GBP"
BASE_SYMBOL = "£"
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
    """
    Pick a single FX rate + label to summarize the main currency risk
    for the AI commentary. If both are EUR, return 1.0.
    """
    buy_ccy = (buy_ccy or "EUR").upper()
    sell_ccy = (sell_ccy or "EUR").upper()
    if "USD" in (buy_ccy, sell_ccy):
        return usd_eur_rate, "USD→EUR"
    if "GBP" in (buy_ccy, sell_ccy):
        return gbp_eur_rate, "GBP→EUR"
    return 1.0, "EUR"

# ---------- Input widgets helpers ----------
def money_input_gbp(label: str, default: float = 0.0, default_ccy: str = "GBP"):
    c1, c2 = st.sidebar.columns([2, 1])
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
    """
    Sidebar input for % of buy price.
    Returns the number (percentage), not the computed EUR/t.
    """
    return st.sidebar.number_input(
        f"{label} (% of buy price)",
        min_value=0.0,
        value=float(default_pct),
        step=0.1,
        format="%.2f",
        key=f"{label}_pct",
    )

# ---------- Sidebar: Trade parameters ----------
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

st.sidebar.title("📦 Trade Parameters")
st.sidebar.markdown("## 🧾 Manual Cost Inputs (per ton)")

volume = st.sidebar.number_input("Volume (tons)", min_value=1, value=25)
buy_price = st.sidebar.number_input("Buy Price", value=7500.0, step=10.0, format="%.2f")
buy_currency = st.sidebar.selectbox("Buy Price Currency", ["GBP", "EUR", "USD"], index=0)
# Set base currency symbol dynamically
currency_symbols = {"EUR": "€", "USD": "$", "GBP": "£"}
base_currency_symbol = currency_symbols.get(buy_currency, "€")


# Convert buy price to EUR for all subsequent calc
if buy_currency == "EUR":
    buy_price *= eur_gbp_rate
elif buy_currency == "USD":
    buy_price *= usd_gbp_rate

# 👉 Buying Diff is part of the base price, not a cost
buying_diff = st.sidebar.number_input(
    "Buying Diff (€ per ton)",
    min_value=0.0,
    value=0.0,
    step=10.0,
    format="%.2f",
)
base_buy = buy_price

port = st.sidebar.selectbox("Port of Loading (POL)", sorted(pol_options))
destination = st.sidebar.selectbox("Destination", sorted(destination_options))
carrier = st.sidebar.selectbox("Shipping Line (optional)", ["Auto (cheapest)"] + sorted(carrier_options))
selected_carrier = None if carrier == "Auto (cheapest)" else carrier

selected_warehouse = st.sidebar.selectbox("Warehouse", sorted(warehouse_options))

payment_days = st.sidebar.number_input("Payment Terms (days)", min_value=0, value=30, step=1)
if payment_days > 0:
    annual_rate = st.sidebar.number_input("Annual Financing Rate (%)", min_value=0.0, value=8.5, step=0.5) / 100
else:
    annual_rate = 0.0

sell_currency = st.sidebar.selectbox("Sell Price Currency", ["GBP", "EUR", "USD"], index=0)

st.sidebar.caption(f"Rates pulled: {datetime.now(ZoneInfo('Europe/Zurich')).strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.markdown("### 💱 FX Rates (Live)")
st.sidebar.markdown(f"- **EUR/USD**: {eur_usd_rate}")
st.sidebar.markdown(f"- **USD/EUR**: {usd_eur_rate}")
st.sidebar.markdown(f"- **GBP/EUR**: {gbp_eur_rate}")
st.sidebar.markdown(f"- **GBP/USD**: {gbp_usd_rate}")

calc_type = st.sidebar.selectbox("Calculation Type", ["Sell Price Calculation", "Margin Calculation"])
is_reverse = (calc_type == "Margin Calculation")

if is_reverse:
    sell_price = None
    target_margin = st.sidebar.number_input("Target Margin (€ per ton)", min_value=0.0, value=200.0, step=10.0)
else:
    target_margin = None
    sell_price = st.sidebar.number_input("Sell Price", min_value=0.0, value=8500.0, step=10.0)
    # convert sell price to EUR if needed
    if sell_currency == "USD":
        sell_price *= usd_eur_rate
    elif sell_currency == "GBP":
        sell_price *= gbp_eur_rate

# For AI commentary
trade_fx_rate, trade_fx_label = choose_trade_fx(buy_currency, sell_currency)

# ---------- Manual costs ----------
# LID switch → 400 GBP/t if YES else 0
lid_yes = st.sidebar.checkbox("LID applies?", value=False)
lid_gbp = (400.0 * eur_gbp_rate) if lid_yes else 0.0

cert_premium_gbp   = money_input_gbp("CERT PREMIUM")
docs_costs_gbp     = money_input_gbp("DOCS COSTS")

qc_type = st.sidebar.selectbox(
    f"QUALITY CLAIM type", 
    [f"{base_currency_symbol}/t", "% of buy"], 
    index=0
)

if qc_type == f"{base_currency_symbol}/t":
    quality_claim_gbp = money_input_gbp("QUALITY CLAIM")
else:
    qc_pct = percent_cost_from_buy("QUALITY CLAIM")
    quality_claim_gbp = (qc_pct / 100.0) * base_buy  # use base including Buying Diff

wl_pct          = percent_cost_from_buy("WEIGHT LOSS")
weight_loss_gbp = (wl_pct / 100.0) * base_buy      # use base including Buying Diff

qc_dep_gbp       = money_input_gbp("QUALITY CONTROLE DEP")
qc_arr_gbp       = money_input_gbp("QUALITY CONTROLE ARR")
origin_agent_gbp = money_input_gbp("ORIGIN AGENT")
dest_agent_gbp   = money_input_gbp("DESTINATION AGENT")

# FREIGHT — manual override or auto from table
use_manual_freight = st.sidebar.checkbox("Enter FREIGHT manually (override route table)?", value=False)
freight_per_ton = None
if use_manual_freight:
    freight_per_ton = money_input_gbp("FREIGHT")


dressing_gbp            = money_input_gbp("DRESSING")
freight_correction_gbp  = money_input_gbp("FREIGHT CORRECTION")
marine_insurance_gbp    = money_input_gbp("MARINE INSURANCE")
stock_insurance_gbp     = money_input_gbp("STOCK INSURANCE")

# ---------- Freight route table (optional) ----------
freight_costs = {}
warehouse_costs = None  # ensure defined for later display

excel_path = "logistics_freight_trade_calc.xlsx"
freight_costs = {}

if os.path.exists(excel_path):
    df_excel = pd.read_excel(excel_path)

    # Normalize columns
    df_excel.columns = [str(c).strip().upper() for c in df_excel.columns]
    if "CONTAINER" in df_excel.columns:
        df_excel = df_excel[df_excel["CONTAINER"].astype(str).str.contains("20", na=False)].copy()

    # Ensure data types
    if "CURRENCY" in df_excel.columns:
        df_excel["CURRENCY"] = df_excel["CURRENCY"].astype(str).str.upper()
    df_excel["ALL_IN"] = pd.to_numeric(df_excel.get("ALL_IN"), errors="coerce")

    # --- Convert ALL_IN to GBP ---
    # Expecting file mostly in EUR; handle USD/GBP just in case.
    # eur_gbp_rate: EUR → GBP, usd_gbp_rate: USD → GBP
    df_excel.loc[df_excel["CURRENCY"] == "EUR", "ALL_IN"] *= eur_gbp_rate
    df_excel.loc[df_excel["CURRENCY"] == "USD", "ALL_IN"] *= usd_gbp_rate
    # If already GBP, leave as-is

    # Validate required columns
    for c in ["POL", "POD", "SHIPPING LINE"]:
        if c not in df_excel.columns:
            st.error(f"Freight file missing column: {c}")

    # Clean names
    df_excel = df_excel.dropna(subset=["POL", "POD", "SHIPPING LINE", "ALL_IN"])
    df_excel["POL"] = df_excel["POL"].astype(str).str.strip().str.upper()
    df_excel["POD"] = df_excel["POD"].astype(str).str.strip().str.upper()
    df_excel["SHIPPING LINE"] = df_excel["SHIPPING LINE"].astype(str).str.strip().str.upper()

    # Store route costs (per container, in GBP). You'll divide by 25 later.
    for _, row in df_excel.iterrows():
        route = (row["POL"], row["POD"])
        carrier_name = row["SHIPPING LINE"]
        cost_gbp = float(row["ALL_IN"])
        freight_costs.setdefault(route, {})[carrier_name] = cost_gbp
else:
    st.warning("Freight file not found: logistics_freight_trade_calc.xlsx")


def get_freight_per_ton(port_from, port_to, selected_carrier=None):
    route = (port_from, port_to)
    if route in freight_costs:
        costs = freight_costs[route]
        if selected_carrier and selected_carrier in costs:
            return round(costs[selected_carrier] / 25, 2)  # 25 t per 20' container
        return round(min(costs.values()) / 25, 2)
    else:
        st.error(f"No freight data available for route: {port_from} → {port_to}")
        return None

# Only fetch auto-freight if user did NOT override
if not use_manual_freight:
    auto_freight = get_freight_per_ton(port, destination, selected_carrier)
    if auto_freight is not None:
        freight_per_ton = auto_freight

# ---------- Warehouse costs (optional file) ----------
warehouse_total_per_ton = 0.0
warehouse_excel_path = "warehouse_costs.xlsx"
if os.path.exists(warehouse_excel_path):
    warehouse_df = pd.read_excel(warehouse_excel_path, index_col=0)
    if selected_warehouse in warehouse_df.columns:
        # assuming values in GBP/t → convert to EUR/t
        warehouse_costs = warehouse_df[selected_warehouse].dropna()
        warehouse_total_per_ton = float(warehouse_costs.sum())
    else:
        st.warning(f"No cost data found for selected warehouse: {selected_warehouse}")
else:
    st.warning("Warehouse cost file not found: warehouse_costs.xlsx")

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
    {"Cost Item": "FREIGHT",                "GBP/ton": round((freight_per_ton or 0.0), 2)},
    {"Cost Item": "DRESSING",               "GBP/ton": round(dressing_gbp, 2)},
    {"Cost Item": "FREIGHT CORRECTION",     "GBP/ton": round(freight_correction_gbp, 2)},
    {"Cost Item": "MARINE INSURANCE",       "GBP/ton": round(marine_insurance_gbp, 2)},
    {"Cost Item": "STOCK INSURANCE",        "GBP/ton": round(stock_insurance_gbp, 2)},
]
manual_df = pd.DataFrame(manual_rows)
manual_subtotal = float(manual_df["GBP/ton"].sum())

with st.expander("📊 Manual Cost Breakdown (per ton)"):
    st.dataframe(manual_df)
    st.write(f"🧮 Manual costs subtotal: **{base_currency_symbol}{manual_subtotal:.2f}**")

# Show the base buy clearly
st.markdown(f"**Base buy (incl. Buying Diff): {base_currency_symbol}{base_buy:.2f}/t**")
st.caption(f"(Buy Price {base_currency_symbol}{buy_price:.2f} + Buying Diff {base_currency_symbol}{buying_diff:.2f})")


# ---------- Containers estimate ----------
containers_needed = round(volume / 25)
st.markdown(f"🧱 Estimated containers: **{containers_needed} × 20'**")

# ---------- Base landed cost & financing (apply ONCE) ----------
base_cost_per_ton = base_buy + manual_subtotal + warehouse_total_per_ton

if payment_days > 0:
    financing_per_ton = (annual_rate / 365) * payment_days * base_cost_per_ton
    cost_per_ton = base_cost_per_ton + financing_per_ton
    st.write(f"💳 Financing cost per ton: {base_currency_symbol}{financing_per_ton:.2f}")
    st.caption(f"Based on {payment_days} days @ {round(annual_rate * 100, 1)}% annual interest")
else:
    cost_per_ton = base_cost_per_ton

st.write(f"💳 Financing cost per ton: {base_currency_symbol}{financing_per_ton:.2f}")
st.write(f"📦 Buy price per ton ({buy_currency}): {base_currency_symbol}{buy_price:.2f}")
st.write(f"➕ Buying Diff added to revenue ({buy_currency}): {base_currency_symbol}{buying_diff:.2f}")
st.write(f"➡️ Base buy per ton ({buy_currency}): **{base_currency_symbol}{base_buy:.2f}**")
st.write(f"🚢 Freight per ton ({buy_currency}): {base_currency_symbol}{(freight_per_ton or 0.0):.2f}")
st.write(f"🏭 Warehouse cost per ton ({buy_currency}): {base_currency_symbol}{warehouse_total_per_ton:.2f}")
st.write(f"💼 Total landed cost per ton ({buy_currency}): **{base_currency_symbol}{cost_per_ton:.2f}**")


# ---------- Warehouse breakdown ----------
with st.expander("📦 Warehouse Cost Breakdown"):
    st.write(f"🏷️ Selected Warehouse: **{selected_warehouse}**")
    if warehouse_costs is not None:
        styled = (
            warehouse_costs.to_frame(name=selected_warehouse).round(2)
            .style.format(precision=2)
            .set_properties(**{'text-align': 'left'})
            .set_table_styles([{'selector': 'th',
                                'props': [('background-color', '#f0f0f0'),
                                          ('text-align', 'left')]}])
        )
        st.write(styled)
    else:
        st.write("No detailed cost breakdown available.")
    st.write(f"📦 Warehouse cost per ton: **{base_currency_symbol}{warehouse_total_per_ton:.2f}**")

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
cocoa_market_price = get_cocoa_price() or 3500

if is_reverse:
    # Given target margin → required sell
    required_sell_price = cost_per_ton + target_margin - buying_diff
    total_revenue = (required_sell_price + buying_diff) * volume

    st.success(f"Required sell price per ton: **€{required_sell_price:.2f}**")
    st.success(f"Total revenue to meet margin target: **€{total_revenue:.2f}**")

    margin_percent = (target_margin / required_sell_price) * 100 if required_sell_price else 0

    with st.expander("🧠 AI Analysis"):
        st.write("Generating AI commentary based on trade parameters...")
        ai_comment = generate_ai_comment(
            buy_price=round(base_buy, 2),          # ✅ show base including Buying Diff
            sell_price=round(required_sell_price, 2),
            freight_cost=round(freight_per_ton or 0.0, 2),
            cocoa_price=cocoa_market_price,
            fx_rate=round(trade_fx_rate, 4),
            fx_label=trade_fx_label,
            margin=margin_percent,
            mode="Target Margin Mode"
        )
        st.markdown(ai_comment)

else:
    # Given sell price → margin
    margin_per_ton = ((sell_price or 0.0) + buying_diff) - cost_per_ton
    total_margin = margin_per_ton * volume
    st.success(f"Margin per ton: **{base_currency_symbol}{margin_per_ton:.2f}**")
    st.success(f"Total margin: **{base_currency_symbol}{total_margin:.2f}**")

    margin_percent = (margin_per_ton / (sell_price + buying_diff)) * 100 if sell_price else 0.0


    with st.expander("🧠 AI Analysis"):
        st.write("Generating AI commentary based on trade parameters...")
        ai_comment = generate_ai_comment(
            buy_price=round(base_buy, 2),          # ✅ show base including Buying Diff
            sell_price=round(sell_price or 0.0, 2),
            freight_cost=round(freight_per_ton or 0.0, 2),
            cocoa_price=cocoa_market_price,
            fx_rate=round(trade_fx_rate, 4),
            fx_label=trade_fx_label,
            margin=margin_percent,
            mode="Margin Calculation Mode"
        )
        st.markdown(ai_comment)
