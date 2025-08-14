import streamlit as st
import os
from openai import OpenAI
import re
import pandas as pd
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime
from datetime import timezone
from zoneinfo import ZoneInfo

now_ch = datetime.now(ZoneInfo("Europe/Zurich"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide")
st.title("🧮 Cocoa Trade Assistant — Forward & Reverse Margin Calculator")
st.write("Calculate trade margin from costs")


def get_fx_rate(pair):
    import yfinance as yf
    ticker = yf.Ticker(pair)
    data = ticker.history(period="1d")
    if not data.empty:
        return round(data["Close"][-1], 4)
    return None

# Get FX rates
eur_usd_rate = get_fx_rate("EURUSD=X") or 1.08
usd_eur_rate = get_fx_rate("USDEUR=X") or 0.93
gbp_eur_rate = get_fx_rate("GBPEUR=X") or 1.17

fx_rate = usd_eur_rate

def convert_gbp_to_eur(amount_gbp):
    return amount_gbp * gbp_eur_rate

def percent_cost_from_buy(label, buy_price_eur, default_pct=0.0):
    pct = st.sidebar.number_input(f"{label} (% of buy price)", min_value=0.0, value=default_pct, step=0.1, format="%.2f")
    return (pct / 100) * buy_price_eur

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
    # default (EUR only or unknown)
    return 1.0, "EUR"
def money_input_eur(label, default=0.0, default_ccy="EUR"):
    col1, col2 = st.sidebar.columns([2, 1])
    amount = col1.number_input(f"{label} amount", min_value=0.0, value=default, step=10.0, format="%.2f")
    ccy = col2.selectbox(f"{label} currency", ["EUR", "GBP", "USD"], index=["EUR", "GBP", "USD"].index(default_ccy))
    
    if ccy == "GBP":
        return amount * gbp_eur_rate
    elif ccy == "USD":
        return amount * usd_eur_rate
    return amount


#setting up sidebar with trade parameters
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
    "STS_ONE","STS_PIL","STS_MESSINA","STS_CMA CGM","STS_MAERSK","PIL"]
st.sidebar.markdown("## 🧾 Manual Cost Inputs (per ton)")

st.sidebar.title("📦 Trade Parameters")
volume = st.sidebar.number_input("Volume (tons)", min_value=1, value=25)
buy_price = st.sidebar.number_input("Buy Price (€)", value=7500.0, step=10.0, format="%.2f")
buying_diff = st.sidebar.number_input(
    "Buying Diff (€ per ton)",
    min_value=0.0,
    value=0.0,
    step=10.0,
    format="%.2f"
)

port = st.sidebar.selectbox("Port of Loading (POL)", sorted(pol_options))
destination = st.sidebar.selectbox("Destination", sorted(destination_options))
carrier = st.sidebar.selectbox(
    "Shipping Line (optional)", 
    ["Auto (cheapest)"] + sorted(carrier_options)
)

# LID switch → 400 GBP/t if YES else 0
lid_yes = st.sidebar.checkbox("LID applies?", value=False)
lid_eur = (400.0 * gbp_eur_rate) if lid_yes else 0.0

# CERT PREMIUM (per ton)
cert_premium_eur = money_input_eur("CERT PREMIUM", default=0.0, default_ccy="EUR")

# DOCS COSTS (per ton)
docs_costs_eur = money_input_eur("DOCS COSTS", default=0.0, default_ccy="EUR")

# QUALITY CLAIM — often % of buy; give option for €/t or %
qc_type = st.sidebar.selectbox("QUALITY CLAIM type", ["€/t", "% of buy"], index=0)
if qc_type == "€/t":
    quality_claim_eur = money_input_eur("QUALITY CLAIM", default=0.0, default_ccy="EUR")
else:
    qc_pct = percent_cost_from_buy("QUALITY CLAIM")
    quality_claim_eur = (qc_pct/100.0) * buy_price

# WEIGHT LOSS — usually % of buy
wl_pct = percent_cost_from_buy("WEIGHT LOSS")
weight_loss_eur = (wl_pct/100.0) * buy_price

# QUALITY CONTROLE DEP/ARR (per ton)
qc_dep_eur = money_input_eur("QUALITY CONTROLE DEP", default=0.0, default_ccy="EUR")
qc_arr_eur = money_input_eur("QUALITY CONTROLE ARR", default=0.0, default_ccy="EUR")

# ORIGIN AGENT / DESTINATION AGENT (per ton)
origin_agent_eur = money_input_eur("ORIGIN AGENT", default=0.0, default_ccy="EUR")
dest_agent_eur   = money_input_eur("DESTINATION AGENT", default=0.0, default_ccy="EUR")

# FREIGHT — you can keep your auto-freight OR allow manual override
use_manual_freight = st.sidebar.checkbox("Enter FREIGHT manually (override route table)?", value=False)
if use_manual_freight:
    freight_per_ton = money_input_eur("FREIGHT", default=0.0, default_ccy="EUR")
# otherwise keep the freight_per_ton you already compute from your Excel routes

# DRESSING (per ton)
dressing_eur = money_input_eur("DRESSING", default=0.0, default_ccy="EUR")

# FREIGHT CORRECTION (per ton)
freight_correction_eur = money_input_eur("FREIGHT CORRECTION", default=0.0, default_ccy="EUR")

# MARINE INSURANCE — keep simple €/t (you can later switch to % of cargo value)
marine_insurance_eur = money_input_eur("MARINE INSURANCE", default=0.0, default_ccy="EUR")

# STOCK INSURANCE — €/t
stock_insurance_eur = money_input_eur("STOCK INSURANCE", default=0.0, default_ccy="EUR")





selected_carrier = carrier if carrier != "—" else None
warehouse_options = [
    "STEINWEG AMSTERDAM",
    "COMMODITY CENTRE AMST",
    "COTTERELL AMST",
    "STEINWEG ANTWERP",
    "COMMODITY CENTRE ANTW",
    "KTN ANTWERP",
    "CWT ANTWERP",
    "VOLLERS ANTWERP",
    "DURME NATIE ANTWERP",
    "COMMODITY CENTRE UK FOR ICE",
    "VOLLERS HAMBURG",
    "COTTERELL HAMBURG",
    "QUAST & CONS HAMBURG",
    "FERRERO",
    "VILLARS",
    "ION SA",
    "GRAND CANDY",
    "DDS",
    "NONE"
]

selected_warehouse = st.sidebar.selectbox("Warehouse", sorted(warehouse_options))
payment_days = st.sidebar.number_input("Payment Terms (days)", min_value=0, value=0)
if payment_days > 0:
    annual_rate = st.sidebar.number_input("Annual Financing Rate (%)", min_value=0.0, value=10.0, step=0.5) / 100
else:
    annual_rate = 0.0
buy_currency = st.sidebar.selectbox("Buy Price Currency", ["EUR", "USD", "GBP"], index=0)
sell_currency = st.sidebar.selectbox("Sell Price Currency", ["EUR", "USD", "GBP"], index=0)
st.sidebar.caption(
    f"Rates pulled: {datetime.now(ZoneInfo('Europe/Zurich')).strftime('%Y-%m-%d %H:%M:%S')}"
)
st.sidebar.markdown("### 💱 FX Rates (Live)")
st.sidebar.markdown(f"- **EUR/USD**: {eur_usd_rate}")
st.sidebar.markdown(f"- **USD/EUR**: {usd_eur_rate}")
st.sidebar.markdown(f"- **GBP/EUR**: {gbp_eur_rate}")

calc_type = st.sidebar.selectbox(
    "Calculation Type",
    ["Sell Price Calculation", "Margin Calculation"]
)

is_reverse = calc_type == "Margin Calculation"

if is_reverse:
    target_margin = st.sidebar.number_input("Target Margin (€ per ton)", min_value=0.0, value=200.0, step=10.0)
    sell_price = None
else:
    sell_price = st.sidebar.number_input("Sell Price (€ per ton)", min_value=0.0, value=8500.0, step=10.0)
    target_margin = None


# Convert buy price to EUR
if buy_currency == "USD":
    buy_price *= usd_eur_rate
elif buy_currency == "GBP":
    buy_price *= gbp_eur_rate

# Convert sell price to EUR (only if not None)
if sell_price is not None:
    if sell_currency == "USD":
        sell_price *= usd_eur_rate
    elif sell_currency == "GBP":
        sell_price *= gbp_eur_rate

trade_fx_rate, trade_fx_label = choose_trade_fx(buy_currency, sell_currency)

trade_data = {
    "volume": volume,
    "buy_term": buy_term,
    "buy_price": buy_price,
    "port": port,
    "destination": destination,
    "carrier": selected_carrier,
    "payment_days": payment_days,
    "is_reverse": is_reverse,
    "target_margin": target_margin,
    "sell_price": sell_price
}

def money_input_eur(label: str, default: float = 0.0, default_ccy: str = "EUR"):
    ccy = st.sidebar.selectbox(f"{label} currency", ["EUR", "GBP"], index=(0 if default_ccy=="EUR" else 1), key=f"{label}_ccy")
    val = st.sidebar.number_input(f"{label} ({ccy}/t)", min_value=0.0, value=float(default), step=10.0, format="%.2f", key=f"{label}_val")
    # convert to EUR/t if needed
    if ccy == "GBP":
        return val * gbp_eur_rate
    return val

def percent_cost_from_buy(label: str):
    pct = st.sidebar.number_input(f"{label} (% of buy price)", min_value=0.0, value=0.0, step=0.1, format="%.2f", key=f"{label}_pct")
    return pct


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

    

# importing the Excel file with freight costs and creating a dictionary of costs
import os
excel_path = "logistics_freight_trade_calc.xlsx"
freight_costs = {}


if os.path.exists(excel_path):
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = [str(col).strip().upper() for col in df_excel.columns]
    df_excel = df_excel[df_excel["CONTAINER"].astype(str).str.contains("20", na=False)]
    df_excel.loc[df_excel["CURRENCY"] == "USD", "ALL_IN"] = (
    df_excel.loc[df_excel["CURRENCY"] == "USD", "ALL_IN"].astype(float) * fx_rate
    )
    df_excel = df_excel[["POL", "POD", "SHIPPING LINE", "ALL_IN"]]
    df_excel = df_excel.dropna(subset=["POL", "POD", "SHIPPING LINE", "ALL_IN"])
    for col in ["POL", "POD", "SHIPPING LINE"]:
        df_excel[col] = df_excel[col].astype(str).str.upper()
    for _, row in df_excel.iterrows():
        route = (row["POL"].strip(), row["POD"].strip())
        carrier = row["SHIPPING LINE"].strip()
        try:
            cost = float(row["ALL_IN"])
        except ValueError:
            continue

        if route not in freight_costs:
            freight_costs[route] = {}
        freight_costs[route][carrier] = cost



# Returns the freight cost per ton for a given route and optional shipping line.
# If a specific carrier is provided and exists for the route, its cost is used.
# Otherwise, the lowest available cost for the route is used.
# Assumes container weight of 25 tons.
# Prints an error message and returns None if the route is not found.

def get_cocoa_price():
    cocoa = yf.Ticker("CC=F")
    data = cocoa.history(period="1d")
    if not data.empty:
        return round(data["Close"][-1], 2)
    return None

def get_freight_per_ton(port_from, port_to, selected_carrier=None):
    route = (port_from, port_to)
    if route in freight_costs:
        costs = freight_costs[route]
        if selected_carrier and selected_carrier in costs:
            return round(costs[selected_carrier] / 25, 2)
        min_value = min(costs.values())
        return round(min_value / 25, 2)
    else:
        st.error(f"No freight data available for route: {port_from} → {port_to}")
        return None

#preparing calculation of freight cost per ton
# Step 1: Container estimate
containers_needed = round(trade_data["volume"] / 25)
st.markdown(f"🧱 Estimated containers: **{containers_needed} x 20'**")

# Step 2: Freight cost
freight_per_ton = get_freight_per_ton(trade_data["port"], trade_data["destination"], selected_carrier)

if freight_per_ton is None:
    st.error("❌ Cannot continue: Freight cost is missing for the selected route.")
    st.stop()  # 🔒 zatrzymaj aplikację, żeby uniknąć dalszych błędów


if freight_per_ton is not None:
    # 📥 Ładowanie kosztów magazynowych z Excela
    warehouse_total_per_ton = 0.0  # fallback

    warehouse_excel_path = "warehouse_costs.xlsx"
    if os.path.exists(warehouse_excel_path):
        warehouse_df = pd.read_excel(warehouse_excel_path, index_col=0)

        if selected_warehouse in warehouse_df.columns:
            warehouse_costs = warehouse_df[selected_warehouse].dropna()
            warehouse_costs = warehouse_costs * gbp_eur_rate
            warehouse_total_per_ton = warehouse_costs.sum()
        else:
            st.warning(f"No cost data found for selected warehouse: {selected_warehouse}")
    else:
        st.warning("Warehouse cost file not found: warehouse_costs.xlsx")


manual_rows = [
    {"Cost Item": "LID",                     "EUR/ton": round(lid_eur, 2)},
    {"Cost Item": "CERT PREMIUM",           "EUR/ton": round(cert_premium_eur, 2)},
    {"Cost Item": "DOCS COSTS",             "EUR/ton": round(docs_costs_eur, 2)},
    {"Cost Item": "QUALITY CLAIM",          "EUR/ton": round(quality_claim_eur, 2)},
    {"Cost Item": "WEIGHT LOSS",            "EUR/ton": round(weight_loss_eur, 2)},
    {"Cost Item": "QUALITY CONTROLE DEP",   "EUR/ton": round(qc_dep_eur, 2)},
    {"Cost Item": "QUALITY CONTROLE ARR",   "EUR/ton": round(qc_arr_eur, 2)},
    {"Cost Item": "ORIGIN AGENT",           "EUR/ton": round(origin_agent_eur, 2)},
    {"Cost Item": "DESTINATION AGENT",      "EUR/ton": round(dest_agent_eur, 2)},
    {"Cost Item": "FREIGHT",                "EUR/ton": round(freight_per_ton or 0.0, 2)},
    {"Cost Item": "DRESSING",               "EUR/ton": round(dressing_eur, 2)},
    {"Cost Item": "FREIGHT CORRECTION",     "EUR/ton": round(freight_correction_eur, 2)},
    {"Cost Item": "MARINE INSURANCE",       "EUR/ton": round(marine_insurance_eur, 2)},
    {"Cost Item": "STOCK INSURANCE",        "EUR/ton": round(stock_insurance_eur, 2)},
]

manual_df = pd.DataFrame(manual_rows)
manual_subtotal = manual_df["EUR/ton"].sum()

with st.expander("📊 Manual Cost Breakdown (per ton)"):
    st.dataframe(manual_df)
    st.write(f"🧮 Manual costs subtotal: **€{round(manual_subtotal, 2)}**")


# 👉 Append the manual Buying Diff as part of the Incoterm breakdown
if buying_diff and buying_diff > 0:
    cost_breakdown_df = pd.concat(
        [
            cost_breakdown_df,
            pd.DataFrame([{"Cost Item": "BUYING DIFF (manual)", "EUR/ton": round(buying_diff, 2)}])
        ],
        ignore_index=True
    )

with st.expander("📊 Incoterm-Based Cost Breakdown"):
    st.dataframe(cost_breakdown_df)
    # Show subtotal for incoterm-based costs
    incoterm_total_cost = cost_breakdown_df["EUR/ton"].sum()
    st.write(f"🚢 Incoterm-based cost per ton: **€{round(incoterm_total_cost, 2)}**")


# 🔄 Dodanie kosztu magazynowego do całkowitego kosztu
# Warehouse total you already compute (in EUR/t): warehouse_total_per_ton
# Financing is applied AFTER assembling the base landed cost (buy + all costs)
base_cost_per_ton = buy_price + warehouse_total_per_ton + manual_subtotal

# Financing (if any)
if trade_data["payment_days"] > 0:
    financing_per_ton = (annual_rate / 365) * trade_data["payment_days"] * base_cost_per_ton
    cost_per_ton = base_cost_per_ton + financing_per_ton
    st.write(f"💳 Financing cost per ton: €{round(financing_per_ton, 2)}")
else:
    cost_per_ton = base_cost_per_ton

st.write(f"💼 Total landed cost per ton: **€{round(cost_per_ton, 2)}**")

# 💬 Pokazanie kosztów magazynu
with st.expander("📦 Warehouse Cost Breakdown"):
    st.write(f"🏷️ Selected Warehouse: **{selected_warehouse}**")

    if warehouse_costs is not None:
        # Konwertuj na DataFrame i zaokrąglij wartości do 2 miejsc
        df_display = warehouse_costs.to_frame(name=selected_warehouse).round(2)

        # Stylizacja: wyśrodkowanie liczb + jasne tło nagłówka
        styled_table = (
            df_display
            .style
            .format(precision=2)  # jeszcze raz zabezpieczenie zaokrąglenia
            .set_properties(**{'text-align': 'left'})
            .set_table_styles([{
                'selector': 'th',
                'props': [('background-color', '#f0f0f0'), ('text-align', 'left')]
            }])
        )

        # Styl działa tylko z write, nie z dataframe
        st.write(styled_table)
    else:
        st.write("No detailed cost breakdown available.")

    st.write(f"📦 Warehouse cost per ton: **€{round(warehouse_total_per_ton, 2)}**")


        # Financing cost calculation
if trade_data["payment_days"] > 0:
        financing_per_ton = (annual_rate / 365) * trade_data["payment_days"] * cost_per_ton
        cost_per_ton += financing_per_ton
        st.write(f"💳 Financing cost per ton: €{round(financing_per_ton, 2)}")
        st.write(f"💼 Updated total landed cost per ton (with financing): **€{round(cost_per_ton, 2)}**")
        st.caption(f"Based on {trade_data['payment_days']} days @ {round(annual_rate * 100, 1)}% annual interest")
else:
        st.write(f"💼 Total landed cost per ton: **€{round(cost_per_ton, 2)}**")


st.write(f"📦 Buy price per ton: €{trade_data['buy_price']}")
st.write(f"🚢 Freight per ton: €{freight_per_ton}")
st.write(f"💼 Total landed cost per ton: **€{round(cost_per_ton, 2)}**")

if trade_data["is_reverse"]:
    # Target margin given → calculate required sell price
    required_sell_price = cost_per_ton + trade_data["target_margin"]
    total_revenue = required_sell_price * trade_data["volume"]

    st.success(f"Required sell price per ton: **€{round(required_sell_price, 2)}**")
    st.success(f"Total revenue to meet margin target: **€{round(total_revenue, 2)}**")

    # AI analysis block for reverse mode
    margin_percent = (trade_data["target_margin"] / required_sell_price) * 100 if required_sell_price else 0
    cocoa_market_price = get_cocoa_price() or 3500  # fallback to 3500 if None

    with st.expander("🧠 AI Analysis"):
        st.write("Generating AI commentary based on trade parameters...")
        ai_comment = generate_ai_comment(
        buy_price=round(trade_data["buy_price"], 2),
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
    # Sell price given → calculate actual margin
        margin_per_ton = trade_data["sell_price"] - cost_per_ton
        total_margin = margin_per_ton * trade_data["volume"]

        st.success(f"Margin per ton: **€{round(margin_per_ton, 2)}**")
        st.success(f"Total margin: **€{round(total_margin, 2)}**")

        # AI analysis block
        margin_percent = (margin_per_ton / trade_data["sell_price"]) * 100 if trade_data["sell_price"] else 0
        cocoa_market_price = get_cocoa_price() or 3500  # fallback to 3500 if None


if freight_per_ton is not None and not trade_data["is_reverse"] and trade_data["sell_price"] is not None:
    with st.expander("🧠 AI Analysis"):
        st.write("Generating AI commentary based on trade parameters...")
        ai_comment = generate_ai_comment(
        buy_price=round(trade_data["buy_price"], 2),
        sell_price=round(trade_data["sell_price"], 2),
        freight_cost=round(freight_per_ton, 2),
        cocoa_price=cocoa_market_price,
        fx_rate=round(trade_fx_rate, 4),
        fx_label=trade_fx_label,
        margin=margin_percent,
        mode="Margin Calculation Mode"
    )
        st.markdown(ai_comment)

