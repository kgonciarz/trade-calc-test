import os
import re
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from openai import OpenAI

# --- Setup
now_ch = datetime.now(ZoneInfo("Europe/Zurich"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide")
st.title("🧮 Cocoa Trade Assistant — Forward & Reverse Margin Calculator")
st.write("Calculate trade margin from costs (manual inputs)")

# --- FX
def get_fx_rate(pair: str):
    t = yf.Ticker(pair)
    data = t.history(period="1d")
    return round(float(data["Close"][-1]), 4) if not data.empty else None

eur_usd_rate = get_fx_rate("EURUSD=X") or 1.08
usd_eur_rate = get_fx_rate("USDEUR=X") or 0.93
gbp_eur_rate = get_fx_rate("GBPEUR=X") or 1.17

def choose_trade_fx(buy_ccy: str, sell_ccy: str):
    buy_ccy = (buy_ccy or "EUR").upper()
    sell_ccy = (sell_ccy or "EUR").upper()
    if "USD" in (buy_ccy, sell_ccy):
        return usd_eur_rate, "USD→EUR"
    if "GBP" in (buy_ccy, sell_ccy):
        return gbp_eur_rate, "GBP→EUR"
    return 1.0, "EUR"

# Money input that converts to EUR on the fly
def money_input_eur(label: str, default: float = 0.0, default_ccy: str = "EUR"):
    col1, col2 = st.sidebar.columns([2, 1])
    amount = col1.number_input(f"{label} amount", min_value=0.0, value=float(default), step=10.0, format="%.2f", key=f"{label}_val")
    ccy = col2.selectbox(f"{label} currency", ["EUR", "GBP", "USD"], index=["EUR","GBP","USD"].index(default_ccy), key=f"{label}_ccy")
    if ccy == "GBP":
        return amount * gbp_eur_rate
    if ccy == "USD":
        return amount * usd_eur_rate
    return amount

def percent_input(label: str, default_pct: float = 0.0):
    return st.sidebar.number_input(f"{label} (% of buy price)", min_value=0.0, value=float(default_pct), step=0.1, format="%.2f", key=f"{label}_pct")

# --- Sidebar: Trade parameters
pol_options = ["POL","ABIDJAN","TIN CAN","APAPA","CALLAO","CONAKRY","DIEGO SUAREZ","DOUALA",
               "FREETOWN","KAMPALA","KRIBI","LEKKI","LOME","MATADI","MOMBASA","MONROVIA",
               "NOSY BE","SAN PEDRO","TAKORADI","TEMA","CARTAGENA","GUAYAQUIL","POSORJA",
               "PAITA","CAUCEDO","ANTWERP","KINSHASA","LAGOS","PISCO"]

destination_options = ["ANTWERP","BARCELONA","AMSTERDAM","HAMBURG","ISTANBUL","ROTTERDAM","VALENCIA",
                       "BATAM","PASIR GUDANG","SURABAYA","PTP","PHILADELPHIA","SZCZECIN","WELLINGTON",
                       "AMBARLI","GENOA","VADO LIGURE","SINGAPORE","TALLINN","JAKARTA","PORT KLANG",
                       "NEW YORK","MONTREAL","PIRAEUS","YOKOHAMA","VALENCIA","BATAM VIA SINGAPORE",
                       "SHANGHAI","KLAIPEDA","LIVERPOOL"]

carrier_options = ["ARKAS","ONE","CMA","HAPAG","MAERSK","MSC","OOCL","STS_MSC","STS_GRIMALDI","STS_OOCL",
                   "STS_HAPAG-LLOYD","STS_ONE","STS_PIL","STS_MESSINA","STS_CMA CGM","STS_MAERSK","PIL"]

st.sidebar.title("📦 Trade Parameters")
volume = st.sidebar.number_input("Volume (tons)", min_value=1, value=25)
buy_currency = st.sidebar.selectbox("Buy Price Currency", ["EUR","USD","GBP"], index=0)
buy_price = st.sidebar.number_input("Buy Price", value=7500.0, step=10.0, format="%.2f")
sell_currency = st.sidebar.selectbox("Sell Price Currency", ["EUR","USD","GBP"], index=0)
calc_type = st.sidebar.selectbox("Calculation Type", ["Sell Price Calculation","Margin Calculation"])

is_reverse = (calc_type == "Margin Calculation")
sell_price = None
target_margin = None
if is_reverse:
    target_margin = st.sidebar.number_input("Target Margin (€ per ton)", min_value=0.0, value=200.0, step=10.0)
else:
    sell_price = st.sidebar.number_input("Sell Price", min_value=0.0, value=8500.0, step=10.0, format="%.2f")

port = st.sidebar.selectbox("Port of Loading (POL)", sorted(pol_options))
destination = st.sidebar.selectbox("Destination", sorted(destination_options))
carrier = st.sidebar.selectbox("Shipping Line (optional)", ["Auto (cheapest)"] + sorted(carrier_options))
selected_carrier = None if carrier == "Auto (cheapest)" else carrier

warehouse_options = ["STEINWEG AMSTERDAM","COMMODITY CENTRE AMST","COTTERELL AMST","STEINWEG ANTWERP",
                     "COMMODITY CENTRE ANTW","KTN ANTWERP","CWT ANTWERP","VOLLERS ANTWERP",
                     "DURME NATIE ANTWERP","COMMODITY CENTRE UK FOR ICE","VOLLERS HAMBURG","COTTERELL HAMBURG",
                     "QUAST & CONS HAMBURG","FERRERO","VILLARS","ION SA","GRAND CANDY","DDS","NONE"]
selected_warehouse = st.sidebar.selectbox("Warehouse", sorted(warehouse_options))

payment_days = st.sidebar.number_input("Payment Terms (days)", min_value=0, value=0)
annual_rate = st.sidebar.number_input("Annual Financing Rate (%)", min_value=0.0, value=10.0, step=0.5) / 100 if payment_days > 0 else 0.0

st.sidebar.caption(f"Rates pulled: {datetime.now(ZoneInfo('Europe/Zurich')).strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.markdown("### 💱 FX Rates (Live)")
st.sidebar.markdown(f"- **EUR/USD**: {eur_usd_rate}")
st.sidebar.markdown(f"- **USD/EUR**: {usd_eur_rate}")
st.sidebar.markdown(f"- **GBP/EUR**: {gbp_eur_rate}")

# Convert buy/sell to EUR
if buy_currency == "USD":
    buy_price *= usd_eur_rate
elif buy_currency == "GBP":
    buy_price *= gbp_eur_rate

if sell_price is not None:
    if sell_currency == "USD":
        sell_price *= usd_eur_rate
    elif sell_currency == "GBP":
        sell_price *= gbp_eur_rate

trade_fx_rate, trade_fx_label = choose_trade_fx(buy_currency, sell_currency)

# --- Manual cost inputs (all per ton, converted to EUR)
st.sidebar.markdown("## 🧾 Manual Cost Inputs (per ton)")

buying_diff = st.sidebar.number_input("Buying Diff (€ per ton)", min_value=0.0, value=0.0, step=10.0, format="%.2f")

# LID
lid_yes = st.sidebar.checkbox("LID applies? (400 GBP/t)", value=False)
lid_eur = (400.0 * gbp_eur_rate) if lid_yes else 0.0

cert_premium_eur   = money_input_eur("CERT PREMIUM", default=0.0)
docs_costs_eur     = money_input_eur("DOCS COSTS", default=0.0)

qc_type = st.sidebar.selectbox("QUALITY CLAIM type", ["€/t", "% of buy"], index=0)
if qc_type == "€/t":
    quality_claim_eur = money_input_eur("QUALITY CLAIM", default=0.0)
else:
    qc_pct = percent_input("QUALITY CLAIM", default_pct=0.0)
    quality_claim_eur = (qc_pct/100.0) * buy_price

wl_pct = percent_input("WEIGHT LOSS", default_pct=0.0)
weight_loss_eur = (wl_pct/100.0) * buy_price

qc_dep_eur       = money_input_eur("QUALITY CONTROLE DEP", default=0.0)
qc_arr_eur       = money_input_eur("QUALITY CONTROLE ARR", default=0.0)
origin_agent_eur = money_input_eur("ORIGIN AGENT", default=0.0)
dest_agent_eur   = money_input_eur("DESTINATION AGENT", default=0.0)
dressing_eur     = money_input_eur("DRESSING", default=0.0)
freight_corr_eur = money_input_eur("FREIGHT CORRECTION", default=0.0)
marine_ins_eur   = money_input_eur("MARINE INSURANCE", default=0.0)
stock_ins_eur    = money_input_eur("STOCK INSURANCE", default=0.0)

# --- Freight: from Excel or manual override
use_manual_freight = st.sidebar.checkbox("Enter FREIGHT manually (override route table)?", value=False)
freight_per_ton = None

# Build route cost map (for auto freight)
excel_path = "logistics_freight_trade_calc.xlsx"
freight_costs = {}
if os.path.exists(excel_path):
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = [str(c).strip().upper() for c in df_excel.columns]
    df_excel = df_excel[df_excel["CONTAINER"].astype(str).str.contains("20", na=False)]
    # Convert USD to EUR for ALL_IN
    df_excel.loc[df_excel["CURRENCY"] == "USD", "ALL_IN"] = (
        df_excel.loc[df_excel["CURRENCY"] == "USD", "ALL_IN"].astype(float) * usd_eur_rate
    )
    df_excel = df_excel[["POL", "POD", "SHIPPING LINE", "ALL_IN"]].dropna()
    for c in ["POL","POD","SHIPPING LINE"]:
        df_excel[c] = df_excel[c].astype(str).strip().upper()
    for _, row in df_excel.iterrows():
        route = (row["POL"], row["POD"])
        carrier_name = row["SHIPPING LINE"]
        cost = float(row["ALL_IN"])
        freight_costs.setdefault(route, {})[carrier_name] = cost

def get_freight_per_ton(port_from, port_to, selected_carrier=None):
    route = (port_from, port_to)
    if route not in freight_costs:
        st.error(f"No freight data available for route: {port_from} → {port_to}")
        return None
    costs = freight_costs[route]
    if selected_carrier and selected_carrier in costs:
        return round(costs[selected_carrier] / 25.0, 2)  # assume 25t per 20'
    return round(min(costs.values()) / 25.0, 2)

if use_manual_freight:
    freight_per_ton = money_input_eur("FREIGHT", default=0.0)
else:
    freight_per_ton = get_freight_per_ton(port.upper(), destination.upper(), selected_carrier)

if freight_per_ton is None:
    st.error("❌ Cannot continue: Freight cost is missing for the selected route.")
    st.stop()

# --- Warehouse costs (GBP → EUR)
warehouse_costs = None
warehouse_total_per_ton = 0.0
warehouse_excel_path = "warehouse_costs.xlsx"
if os.path.exists(warehouse_excel_path):
    wdf = pd.read_excel(warehouse_excel_path, index_col=0)
    if selected_warehouse in wdf.columns:
        warehouse_costs = (wdf[selected_warehouse].dropna() * gbp_eur_rate)
        warehouse_total_per_ton = float(warehouse_costs.sum())
    else:
        st.warning(f"No cost data found for selected warehouse: {selected_warehouse}")
else:
    st.warning("Warehouse cost file not found: warehouse_costs.xlsx")

# --- Manual cost breakdown table (includes FREIGHT row)
manual_rows = [
    {"Cost Item": "BUYING DIFF (manual)",  "EUR/ton": round(buying_diff, 2)},
    {"Cost Item": "LID",                   "EUR/ton": round(lid_eur, 2)},
    {"Cost Item": "CERT PREMIUM",          "EUR/ton": round(cert_premium_eur, 2)},
    {"Cost Item": "DOCS COSTS",            "EUR/ton": round(docs_costs_eur, 2)},
    {"Cost Item": "QUALITY CLAIM",         "EUR/ton": round(quality_claim_eur, 2)},
    {"Cost Item": "WEIGHT LOSS",           "EUR/ton": round(weight_loss_eur, 2)},
    {"Cost Item": "QUALITY CONTROLE DEP",  "EUR/ton": round(qc_dep_eur, 2)},
    {"Cost Item": "QUALITY CONTROLE ARR",  "EUR/ton": round(qc_arr_eur, 2)},
    {"Cost Item": "ORIGIN AGENT",          "EUR/ton": round(origin_agent_eur, 2)},
    {"Cost Item": "DESTINATION AGENT",     "EUR/ton": round(dest_agent_eur, 2)},
    {"Cost Item": "FREIGHT",               "EUR/ton": round(freight_per_ton, 2)},
    {"Cost Item": "DRESSING",              "EUR/ton": round(dressing_eur, 2)},
    {"Cost Item": "FREIGHT CORRECTION",    "EUR/ton": round(freight_corr_eur, 2)},
    {"Cost Item": "MARINE INSURANCE",      "EUR/ton": round(marine_ins_eur, 2)},
    {"Cost Item": "STOCK INSURANCE",       "EUR/ton": round(stock_ins_eur, 2)},
]
manual_df = pd.DataFrame(manual_rows)
manual_subtotal = float(manual_df["EUR/ton"].sum())

with st.expander("📊 Manual Cost Breakdown (per ton)"):
    st.dataframe(manual_df)
    st.write(f"🧮 Manual costs subtotal: **€{round(manual_subtotal, 2)}**")

# --- Total landed cost (EUR/t)
base_cost_per_ton = buy_price + warehouse_total_per_ton + manual_subtotal
if payment_days > 0:
    financing_per_ton = (annual_rate / 365.0) * payment_days * base_cost_per_ton
    cost_per_ton = base_cost_per_ton + financing_per_ton
    st.write(f"💳 Financing cost per ton: €{round(financing_per_ton, 2)}")
else:
    cost_per_ton = base_cost_per_ton

st.write(f"📦 Buy price per ton (EUR): **€{round(buy_price, 2)}**")
st.write(f"🚢 Freight per ton (EUR): **€{round(freight_per_ton, 2)}**")
st.write(f"🏬 Warehouse cost per ton (EUR): **€{round(warehouse_total_per_ton, 2)}**")
st.write(f"💼 Total landed cost per ton (EUR): **€{round(cost_per_ton, 2)}**")

# --- Cocoa price (for AI context)
def get_cocoa_price():
    cocoa = yf.Ticker("CC=F")
    data = cocoa.history(period="1d")
    return round(float(data["Close"][-1]), 2) if not data.empty else None

cocoa_market_price = get_cocoa_price() or 3500

# --- AI commentary
def generate_ai_comment(buy_price, sell_price, freight_cost, cocoa_price, fx_rate, fx_label, margin, mode):
    prompt = f"""
You are a commodity market analyst. Based on the following trade parameters:

- Calculation mode: {mode}
- Purchase price: {buy_price} EUR/ton
- Selling price: {sell_price} EUR/ton
- Freight cost: {freight_cost} EUR/ton
- Cocoa market price: {cocoa_price} EUR/ton
- FX rate ({fx_label}): {fx_rate}
- Calculated margin: {margin:.2f}%

Please provide:
1) Whether the margin is attractive in the current cocoa market,
2) Key risks (FX, supply/demand, freight),
3) 1–2 actionable recommendations.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error generating AI comment: {e}"

# Reverse vs forward
if is_reverse:
    required_sell_price = cost_per_ton + target_margin
    total_revenue = required_sell_price * volume
    st.success(f"Required sell price per ton: **€{round(required_sell_price, 2)}**")
    st.success(f"Total revenue to meet margin target: **€{round(total_revenue, 2)}**")
    margin_percent = (target_margin / required_sell_price * 100.0) if required_sell_price else 0.0

    with st.expander("🧠 AI Analysis"):
        ai_comment = generate_ai_comment(
            buy_price=round(buy_price, 2),
            sell_price=round(required_sell_price, 2),
            freight_cost=round(freight_per_ton, 2),
            cocoa_price=cocoa_market_price,
            fx_rate=round(trade_fx_rate, 4),
            fx_label=trade_fx_label,
            margin=margin_percent,
            mode="Target Margin Mode"
        )
        st.markdown(ai_comment)
else:
    margin_per_ton = (sell_price or 0) - cost_per_ton if sell_price is not None else 0.0
    total_margin = margin_per_ton * volume
    st.success(f"Margin per ton: **€{round(margin_per_ton, 2)}**")
    st.success(f"Total margin: **€{round(total_margin, 2)}**")
    margin_percent = (margin_per_ton / sell_price * 100.0) if sell_price else 0.0

    if sell_price is not None:
        with st.expander("🧠 AI Analysis"):
            ai_comment = generate_ai_comment(
                buy_price=round(buy_price, 2),
                sell_price=round(sell_price, 2),
                freight_cost=round(freight_per_ton, 2),
                cocoa_price=cocoa_market_price,
                fx_rate=round(trade_fx_rate, 4),
                fx_label=trade_fx_label,
                margin=margin_percent,
                mode="Margin Calculation Mode"
            )
            st.markdown(ai_comment)
