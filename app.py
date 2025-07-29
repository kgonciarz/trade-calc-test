import streamlit as st
import re
import pandas as pd

st.set_page_config(layout="wide")
st.title("🧮 Cocoa Trade Assistant — Forward & Reverse Margin Calculator")
st.write("Calculate margin from a price — or compute required sale price to meet a target margin.")

st.caption("📍 Available routes: Abidjan → Antwerp, San Pedro → Hamburg, Accra → Rotterdam")

# --- Cost data ---
with st.sidebar:
    st.header("⚙️ Parameters")
    finance_rate = st.number_input("Annual finance rate (%)", value=8.0, step=0.1) / 100
    fx_currencies = ["EUR", "USD", "GBP"]
    base_currency = st.selectbox("Base currency:", fx_currencies, index=0)
    quote_currency = st.selectbox("Quote currency:", fx_currencies, index=1)
    fx_rate = st.number_input(f"Exchange rate ({base_currency}/{quote_currency})", value=1.08, step=0.01)
    eur_usd = fx_rate if base_currency == "EUR" and quote_currency == "USD" else 1.08  # fallback if needed
transport_to_port = {
    "Kumasi": {"Abidjan": 120, "San Pedro": 130},
    "Tamale": {"Accra": 140}
}

port_fobbing = {
    "Abidjan": 100,
    "San Pedro": 110,
    "Accra": 105
}

import os
excel_path = "logistics_freight_trade_calc.xlsx"
freight_costs = {}

if os.path.exists(excel_path):
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = [str(col).strip().upper() for col in df_excel.columns]
    df_excel = df_excel.dropna(subset=["POL", "POD", "SHIPPING LINE", "ALL_IN"])
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
else:
    st.warning("⚠️ Excel file with freight costs not found — using built-in sample.")
    freight_costs = {
        ("Abidjan", "Antwerp"): {"CMA CGM": 600, "MSC": 559},
        ("San Pedro", "Hamburg"): {"Simulated": 900},
        ("Accra", "Rotterdam"): {"Simulated": 880}
    }


# --- Freight helper ---
def get_freight_per_ton(port_from, port_to, selected_carrier=None):
    route = (port_from, port_to)
    if route in freight_costs:
        costs = freight_costs[route]
        if selected_carrier and selected_carrier in costs:
            return round(costs[selected_carrier] / 25, 2)
        min_value = min(costs.values())
        return round(min_value / 25, 2)
    else:
        return 90  # fallback

def select_freight_carrier(trade):
    route = (trade.get("port"), trade.get("destination"))
    carriers = list(freight_costs.get(route, {}).keys())
    if len(carriers) > 1:
        return st.selectbox("🚢 Choose shipping line:", carriers)
    elif carriers:
        st.caption(f"Only one freight option available: {carriers[0]}")
        return carriers[0]
    else:
        st.caption("No freight data available for this route.")
        return None
# --- Parse query ---
def parse_query(text):
    data = {}
    data["volume"] = int(re.findall(r"(\d+)\s?T", text)[0])
    data["buy_price"] = float(re.findall(r"(EXW|FCA|FOB).*?[€$]?(\d{3,5})", text)[0][1])
    data["buy_term"] = re.findall(r"(EXW|FCA|FOB)", text)[0]
    data["origin_city"] = re.findall(r"(?:EXW|FOB|FCA)\s+([A-Za-z\s]+)", text)
    data["port"] = re.findall(r"FOB\s+([A-Za-z\s]+)", text)
    dest_match = re.findall(r"(?:CIF|CFR)\s+([A-Za-z\s]+)", text)
    data["destination"] = dest_match[0].strip() if dest_match else None
    if not data["destination"]:
        via_match = re.findall(r"via\s+([A-Za-z\s]+)", text)
        data["destination"] = via_match[0].strip() if via_match else None
    data["payment_days"] = int(re.findall(r"(\d{1,3})[dD]", text)[0]) if "LC" in text else 0
    data["currency"] = "EUR" if "€" in text or "EUR" in text else "USD"
    data["is_reverse"] = "target" in text.lower()
    if data["is_reverse"]:
        data["target_margin"] = float(re.findall(r"target.*?€?(\d{2,5})", text.lower())[0])
        data["sell_term"] = "CIF"
    else:
        data["sell_price"] = float(re.findall(r"(CIF|CFR).*?[€$]?(\d{3,5})", text)[0][1])
        data["sell_term"] = re.findall(r"(CIF|CFR)", text)[0]
    data["origin_city"] = data["origin_city"][0].strip() if data["origin_city"] else None
    data["port"] = data["port"][0].strip() if data["port"] else None
    return data

# --- Cost builder ---
def build_cost(trade, selected_carrier=None):
    total = trade["buy_price"]
    costs = [("Base (" + trade["buy_term"] + ")", total)]

    if trade["buy_term"] == "EXW" and trade["origin_city"] and trade["port"]:
        inland = transport_to_port.get(trade["origin_city"], {}).get(trade["port"], 0)
        total += inland
        costs.append(("Inland (EXW→FCA)", inland))

    if trade["port"]:
        port_cost = port_fobbing.get(trade["port"], 100)
        total += port_cost
        costs.append(("Port (FCA→FOB)", port_cost))

    # Only add freight if sale term is CIF or CFR
    if trade["sell_term"] in ["CIF", "CFR"]:
        freight = get_freight_per_ton(trade["port"], trade["destination"], selected_carrier)
        total += freight
        costs.append(("Freight (FOB→" + trade["sell_term"] + ")", freight))

    finance = (total * finance_rate) * (trade["payment_days"] / 360)
    total += finance
    costs.append(("Finance", round(finance, 2)))

    return total, costs

# --- Margin logic ---
def compute_forward(trade, carrier):
    total_cost, details = build_cost(trade, carrier)
    if trade["currency"] == "USD":
        sell_price = trade["sell_price"] / eur_usd
    else:
        sell_price = trade["sell_price"]
    margin_per_ton = sell_price - total_cost
    total_margin = margin_per_ton * trade["volume"]
    return details, {
        "Sell Price": round(sell_price, 2),
        "Total Cost": round(total_cost, 2),
        "Margin/ton": round(margin_per_ton, 2),
        "Total Margin": round(total_margin, 2),
        "Selected Carrier": carrier if carrier else "Auto (cheapest)",
        "Freight €/ton": get_freight_per_ton(trade["port"], trade["destination"], carrier)
    }

def compute_reverse(trade, carrier):
    total_cost, details = build_cost(trade, carrier)
    required_sell = total_cost + trade["target_margin"]
    total_margin = trade["target_margin"] * trade["volume"]
    return details, {
        "Required Sale Price": round(required_sell, 2),
        "Target Margin/ton": trade["target_margin"],
        "Total Cost": round(total_cost, 2),
        "Total Target Margin": round(total_margin, 2),
        "Selected Carrier": carrier if carrier else "Auto (cheapest)",
        "Freight €/ton": get_freight_per_ton(trade["port"], trade["destination"], carrier)
    }

# --- Cost builder ---
def build_cost(trade, selected_carrier=None):
    total = trade["buy_price"]
    costs = [("Base (" + trade["buy_term"] + ")", total)]

    if trade["buy_term"] == "EXW" and trade["origin_city"] and trade["port"]:
        inland = transport_to_port.get(trade["origin_city"], {}).get(trade["port"], 0)
        total += inland
        costs.append(("Inland (EXW→FCA)", inland))

    if trade["buy_term"] in ["EXW", "FCA"] and trade["port"]:
        port_cost = port_fobbing.get(trade["port"], 100)
        total += port_cost
        costs.append(("Port (FCA→FOB)", port_cost))

    if trade["sell_term"] in ["CIF", "CFR"]:
        freight = get_freight_per_ton(trade["port"], trade["destination"], selected_carrier)
        total += freight
        costs.append(("Freight (FOB→" + trade["sell_term"] + ")", freight))

    finance = (total * finance_rate) * (trade["payment_days"] / 360)
    total += finance
    costs.append(("Finance", round(finance, 2)))

    return total, costs

# --- UI ---
query = st.text_input("✏️ Trade scenario:", 
                      "Buy 250T EXW Kumasi €2800, FOB Abidjan, target €200 margin, sell CIF Antwerp")

if query:
    try:
        trade = parse_query(query)

        # Carrier dropdown
        carrier_list = freight_costs.get((trade['port'], trade['destination']), {}).keys()
        selected_carrier = st.selectbox("🚢 Select Shipping Line (optional)", ["Auto (cheapest)"] + list(carrier_list))
        carrier_used = selected_carrier if selected_carrier != "Auto (cheapest)" else None

        containers_needed = round(trade["volume"] / 25)
        st.markdown(f"🧱 Estimated containers: **{containers_needed} x 40'**")

        if trade["is_reverse"]:
            breakdown, summary = compute_reverse(trade, carrier_used)
            st.subheader("🔁 Reverse Calculator")
            st.markdown(f"🎯 To hit a **{summary['Target Margin/ton']} EUR/ton** margin:")
            st.success(f"👉 You must sell at **{summary['Required Sale Price']} EUR/ton {trade['sell_term']} {trade['destination']}**")
        else:
            breakdown, summary = compute_forward(trade, carrier_used)
            st.subheader("📈 Forward Margin Calculation")
            st.success(f"💰 Margin: {summary['Margin/ton']} EUR/ton")

        st.markdown("### 🔍 Cost Breakdown")
        st.dataframe(pd.DataFrame(breakdown, columns=["Element", "EUR/ton"]))

        st.markdown("### 📊 Summary")
        st.json(summary)

    except Exception as e:
        st.error("Could not parse input. Example:")
        st.code("Buy 250T EXW Kumasi €2800, FOB Abidjan, sell CIF Rotterdam €3600, 45d LC\nor:\nBuy 250T EXW Kumasi €2800, via Abidjan, target €200 margin")
        st.exception(e)
