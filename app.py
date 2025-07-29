import streamlit as st
import re
import pandas as pd

st.set_page_config(layout="wide")
st.title("🧮 Cocoa Trade Assistant — Forward & Reverse Margin Calculator")
st.write("Calculate margin from a price — or compute required sale price to meet a target margin.")

# --- Cost data ---
transport_to_port = {
    "Kumasi": {"Abidjan": 120, "San Pedro": 130},
    "Tamale": {"Accra": 140}
}

port_fobbing = {
    "Abidjan": 100,
    "San Pedro": 110,
    "Accra": 105
}

freight_costs = {
    ("Abidjan", "Rotterdam"): 85,
    ("San Pedro", "Hamburg"): 90,
    ("Accra", "Antwerp"): 88
}

finance_rate = 0.08  # annual
eur_usd = 1.08

# --- Parse query ---
def parse_query(text):
    data = {}
    data["volume"] = int(re.findall(r"(\d+)\s?T", text)[0])
    data["buy_price"] = float(re.findall(r"(EXW|FCA|FOB).*?[€$]?(\d{3,5})", text)[0][1])
    data["buy_term"] = re.findall(r"(EXW|FCA|FOB)", text)[0]
    data["origin_city"] = re.findall(r"EXW\s+([A-Za-z\s]+)", text)
    data["port"] = re.findall(r"FOB\s+([A-Za-z\s]+)", text)
    # Safe extract from regex
    dest_match = re.findall(r"(?:CIF|CFR)\s+([A-Za-z\s]+)", text)
    data["destination"] = dest_match[0].strip() if dest_match else None
    data["destination"] = dest_match[0].strip() if dest_match else None
    if not data["destination"]:
        data["destination"] = data.get("port", "Unknown")
    data["payment_days"] = int(re.findall(r"(\d{1,3})[dD]", text)[0]) if "LC" in text else 0
    data["currency"] = "EUR" if "€" in text or "EUR" in text else "USD"
    data["is_reverse"] = "target" in text.lower()
    if data["is_reverse"]:
        data["target_margin"] = float(re.findall(r"target.*?€?(\d{2,5})", text.lower())[0])
        data["sell_term"] = "CIF"  # implied
    else:
        data["sell_price"] = float(re.findall(r"(CIF|CFR).*?[€$]?(\d{3,5})", text)[0][1])
        data["sell_term"] = re.findall(r"(CIF|CFR)", text)[0]
    data["origin_city"] = data["origin_city"][0].strip() if data["origin_city"] else None
    data["port"] = data["port"][0].strip() if data["port"] else None
    # Fallback if not found
    if not data["destination"]:
        via_match = re.findall(r"via\s+([A-Za-z\s]+)", text)
        data["destination"] = via_match[0].strip() if via_match else None
    return data

# --- Cost builder ---
def build_cost(trade):
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

    freight = freight_costs.get((trade["port"], trade["destination"]), 90)
    total += freight
    costs.append(("Freight (FOB→" + trade["sell_term"] + ")", freight))

    finance = (total * finance_rate) * (trade["payment_days"] / 360)
    total += finance
    costs.append(("Finance", round(finance, 2)))

    return total, costs

# --- Margin logic ---
def compute_forward(trade):
    total_cost, details = build_cost(trade)
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
        "Total Margin": round(total_margin, 2)
    }

def compute_reverse(trade):
    total_cost, details = build_cost(trade)
    required_sell = total_cost + trade["target_margin"]
    total_margin = trade["target_margin"] * trade["volume"]
    return details, {
        "Required Sale Price": round(required_sell, 2),
        "Target Margin/ton": trade["target_margin"],
        "Total Cost": round(total_cost, 2),
        "Total Target Margin": round(total_margin, 2)
    }

# --- UI ---
query = st.text_input("✏️ Trade scenario:", 
                      "Buy 250T EXW Kumasi €2800, export via Abidjan, target €200 margin")

if query:
    try:
        trade = parse_query(query)
        if trade["is_reverse"]:
            breakdown, summary = compute_reverse(trade)
            st.subheader("🔁 Reverse Calculator")
            st.markdown(f"🎯 To hit a **{summary['Target Margin/ton']} EUR/ton** margin:")
            st.success(f"👉 You must sell at **{summary['Required Sale Price']} EUR/ton CIF {trade['destination']}**")
        else:
            breakdown, summary = compute_forward(trade)
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
