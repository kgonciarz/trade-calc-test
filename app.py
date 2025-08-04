import streamlit as st
import openai
import re
import pandas as pd
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

st.set_page_config(layout="wide")
st.title("🧮 Cocoa Trade Assistant — Forward & Reverse Margin Calculator")
st.write("Calculate trade margin from costs")

#fx rates
usd_to_eur = 0.93

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


st.sidebar.title("📦 Trade Parameters")
volume = st.sidebar.number_input("Volume (tons)", min_value=1, value=25)
buy_term = st.sidebar.selectbox("Buy Term", ["EXW", "FCA", "FOB"], index=0)
buy_price = st.sidebar.number_input("Buy Price (€)", value=1200.0, step=10.0, format="%.2f")
port = st.sidebar.selectbox("Port of Loading (POL)", sorted(pol_options))
destination = st.sidebar.selectbox("Destination", sorted(destination_options))
carrier = st.sidebar.selectbox(
    "Shipping Line (optional)", 
    ["Auto (cheapest)"] + sorted(carrier_options)
)
selected_carrier = carrier if carrier != "—" else None
payment_days = st.sidebar.number_input("Payment Terms (days)", min_value=0, value=90)
if payment_days > 0:
    annual_rate = st.sidebar.number_input("Annual Financing Rate (%)", min_value=0.0, value=10.0, step=0.5) / 100
else:
    annual_rate = 0.0
buy_currency = st.sidebar.selectbox("Buy Price Currency", ["EUR", "USD"], index=0)
sell_currency = st.sidebar.selectbox("Sell Price Currency", ["EUR", "USD"], index=0)
calc_type = st.sidebar.selectbox(
    "Calculation Type",
    ["Sell Price Calculation", "Margin Calculation"]
)

is_reverse = calc_type == "Margin Calculation"

if is_reverse:
    target_margin = st.sidebar.number_input("Target Margin (€ per ton)", min_value=0.0, value=200.0, step=10.0)
    sell_price = None
else:
    sell_price = st.sidebar.number_input("Sell Price (€ per ton)", min_value=0.0, value=1400.0, step=10.0)
    target_margin = None


if buy_currency == "USD":
    buy_price *= usd_to_eur

if sell_currency == "USD":
    sell_price *= usd_to_eur
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

def generate_ai_comment(buy_price, sell_price, freight_cost, cocoa_price, fx_rate, margin):
    prompt = f"""
You are a commodity market analyst. Based on the following data:

- Purchase price: {buy_price} EUR/ton
- Selling price: {sell_price} EUR/ton
- Freight cost: {freight_cost} EUR/ton
- Cocoa market price: {cocoa_price} EUR/ton
- FX rate (USD/EUR): {fx_rate}
- Calculated margin: {margin:.2f}%

Please provide:
1. Assessment of whether the margin is attractive in the current cocoa market.
2. Identification of potential risks (e.g., FX volatility, supply/demand shifts, freight rate changes).
3. 1 or 2 concise recommendations for the trader based on these figures.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )

        return response.choices[0].message.content

# importing the Excel file with freight costs and creating a dictionary of costs
import os
excel_path = "logistics_freight_trade_calc.xlsx"
freight_costs = {}


if os.path.exists(excel_path):
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = [str(col).strip().upper() for col in df_excel.columns]
    df_excel = df_excel[df_excel["CONTAINER"].astype(str).str.contains("20", na=False)]
    df_excel.loc[df_excel["CURRENCY"] == "USD", "ALL_IN"] = (
        df_excel.loc[df_excel["CURRENCY"] == "USD", "ALL_IN"].astype(float) * usd_to_eur
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

if freight_per_ton is not None:
    cost_per_ton = trade_data["buy_price"] + freight_per_ton

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
    else:
    # Sell price given → calculate actual margin
        margin_per_ton = trade_data["sell_price"] - cost_per_ton
        total_margin = margin_per_ton * trade_data["volume"]

        st.success(f"Margin per ton: **€{round(margin_per_ton, 2)}**")
        st.success(f"Total margin: **€{round(total_margin, 2)}**")

else:
    st.warning("⚠️ No freight cost available — cannot perform margin calculation.")

# AI-generated analysis
cocoa_market_price = 3500  # <-can be dynamic value
fx_rate = usd_to_eur 
margin_percent = (margin_per_ton / trade_data["sell_price"]) * 100 if trade_data["sell_price"] else 0

with st.expander("🧠 AI Analysis"):
    st.write("Generating AI commentary based on trade parameters...")
    ai_comment = generate_ai_comment(
        buy_price=round(trade_data["buy_price"], 2),
        sell_price=round(trade_data["sell_price"], 2),
        freight_cost=round(freight_per_ton, 2),
        cocoa_price=cocoa_market_price,
        fx_rate=round(fx_rate, 4),
        margin=margin_percent
    )
    st.markdown(ai_comment)

