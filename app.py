import streamlit as st
import re
import pandas as pd

st.set_page_config(layout="wide")
st.title("💼 Cocoa Trade Margin Assistant (v2)")
st.write("Simulate margin based on Incoterms, FX, freight, finance, and fobbing logic — no Excel needed.")

# --- Base data ---
port_costs = {
    "Abidjan": 100,
    "San Pedro": 110,
    "Lagos": 120,
    "Accra": 105
}

freight_matrix = {
    ("Abidjan", "Amsterdam"): 80,
    ("San Pedro", "Hamburg"): 85,
    ("Lagos", "Rotterdam"): 95,
    ("Accra", "Antwerp"): 88
}

finance_rate_annual = 0.08  # 8%
eur_usd = 1.08

# --- Query parser ---
def parse_query(text):
    volume = int(re.findall(r"(\d+)\s?T", text)[0])
    origin = re.findall(r"FOB\s+([A-Za-z\s]+)", text)[0].strip()
    destination = re.findall(r"CIF\s+([A-Za-z\s]+)", text)[0].strip()
    fob_price = float(re.findall(r"FOB.*?[€$]?(\d{3,5})", text)[0].replace(",", ""))
    cif_price = float(re.findall(r"CIF.*?[€$]?(\d{3,5})", text)[0].replace(",", ""))
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

# --- Calculation logic ---
def compute_margin(trade):
    fobbing = port_costs.get(trade["origin"], 100)
    freight = freight_matrix.get((trade["origin"], trade["destination"]), 90)
    finance_cost = (trade["fob_price"] * finance_rate_annual) * (trade["payment_days"] / 360)

    # Assume CIF is in EUR; convert if needed
    if trade["currency"] == "USD":
        fob_eur = trade["fob_price"] / eur_usd
        cif_eur = trade["cif_price"] / eur_usd
        currency_used = "USD"
    else:
        fob_eur = trade["fob_price"]
        cif_eur = trade["cif_price"]
        currency_used = "EUR"

    total_cost_eur = fob_eur + fobbing + freight + finance_cost
    margin_per_ton = cif_eur - total_cost_eur
    total_margin = margin_per_ton * trade["volume"]

    breakdown = {
        "FOB": [fob_eur],
        "Fobbing": [fobbing],
        "Freight": [freight],
        "Finance": [round(finance_cost, 2)],
        "Total Cost": [round(total_cost_eur, 2)],
        "CIF Sale Price": [cif_eur],
        "Margin/ton": [round(margin_per_ton, 2)],
        "Total Margin": [round(total_margin, 2)],
        "Volume (Tons)": [trade["volume"]],
        "Currency": [currency_used]
    }

    return breakdown, margin_per_ton

# --- UI ---
query = st.text_input("✏️ Type your trade scenario:",
                      "Buy 250T FOB Abidjan €3300, sell CIF Amsterdam €3850, 45d LC")

if query:
    try:
        trade_data = parse_query(query)
        breakdown, margin_per_ton = compute_margin(trade_data)

        st.subheader("📊 Cost & Margin Breakdown")
        st.dataframe(pd.DataFrame(breakdown))

        # --- Alerts ---
        if margin_per_ton < 50:
            st.error(f"⚠️ Low margin: only {margin_per_ton:.2f} EUR/ton")
        else:
            st.success(f"💰 Healthy margin: {margin_per_ton:.2f} EUR/ton")

        st.markdown("---")

        st.caption("All costs shown in EUR per MT (converted if needed). FX rate: 1 EUR = 1.08 USD")
    except Exception as e:
        st.error("Failed to parse the trade query. Try this format:")
        st.code("Buy 250T FOB San Pedro €3200, sell CIF Hamburg €3800, 60d LC")
        st.exception(e)
