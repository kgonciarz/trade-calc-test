import streamlit as st
import re

st.set_page_config(layout="wide")
st.title("💼 Cocoa Trade Margin Simulator (No Excel Needed)")
st.write("Simulate trade margin using hardcoded logic from your real-world pricing structure.")

# --- Sample cost tables ---
port_costs = {
    "Abidjan": 100,
    "San Pedro": 110,
    "Lagos": 120
}

freight_matrix = {
    ("Abidjan", "Amsterdam"): 80,
    ("San Pedro", "Hamburg"): 85,
    ("Lagos", "Rotterdam"): 95
}

finance_rate_annual = 0.08
eur_usd = 1.08

# --- Helpers ---
def parse_query(text):
    # Example: "Buy 250T FOB Abidjan €3300, sell CIF Amsterdam €3850, 45d LC"
    volume = int(re.findall(r"(\d+)\s?T", text)[0])
    origin = re.findall(r"FOB\s+([A-Za-z\s]+)", text)[0].strip()
    destination = re.findall(r"CIF\s+([A-Za-z\s]+)", text)[0].strip()
    fob_price = float(re.findall(r"FOB.*?[€$]?(\d{3,5})", text)[0])
    cif_price = float(re.findall(r"CIF.*?[€$]?(\d{3,5})", text)[0])
    payment_days = int(re.findall(r"(\d{1,3})[dD]", text)[0]) if "LC" in text else 0
    currency = "EUR" if "€" in text or "EUR" in text else "USD"
    return {
        "volume": volume,
        "origin": origin,
        "destination": destination,
        "fob_price": fob_price,
        "cif_price": cif_price,
        "currency": currency,
        "payment_days": payment_days
    }

def compute_margin(trade):
    fobbing = port_costs.get(trade["origin"], 100)
    freight = freight_matrix.get((trade["origin"], trade["destination"]), 90)
    finance_cost = (trade["fob_price"] * finance_rate_annual) * (trade["payment_days"] / 360)

    total_cost = trade["fob_price"] + fobbing + freight + finance_cost
    margin_per_ton = trade["cif_price"] - total_cost
    total_margin = margin_per_ton * trade["volume"]

    return {
        "FOB": trade["fob_price"],
        "Fobbing": fobbing,
        "Freight": freight,
        "Finance Cost": round(finance_cost, 2),
        "Total Cost (€/MT)": round(total_cost, 2),
        "CIF Price": trade["cif_price"],
        "Margin per Ton": round(margin_per_ton, 2),
        "Total Margin": round(total_margin, 2),
        "Volume (MT)": trade["volume"],
        "Currency": trade["currency"]
    }

# --- Streamlit UI ---
query = st.text_input("✏️ Type your trade scenario:",
                      "Buy 250T FOB Abidjan €3300, sell CIF Amsterdam €3850, 45d LC")

if query:
    try:
        trade_data = parse_query(query)
        margin_result = compute_margin(trade_data)

        st.subheader("📊 Result")
        st.json(margin_result)

        st.success(f"💰 Margin per Ton: {margin_result['Margin per Ton']} {margin_result['Currency']}")
    except Exception as e:
        st.error("Failed to interpret the trade query. Please use a format like:")
        st.code("Buy 100T FOB San Pedro €3100, sell CIF Hamburg €3650, 60d LC")
        st.exception(e)
