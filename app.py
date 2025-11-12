"""
app.py — Global FactCheck Network (Dark Theme)

This Streamlit app allows users to verify news headlines or claims using
the Google Fact Check Tools API. It presents results in a professional,
news-portal-style interface.
"""

# ===============================
# 🧩 Import required libraries
# ===============================
import streamlit as st
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from ftfy import fix_text

# ===============================
# 🔑 API Key Configuration
# ===============================
# Tries to read from Streamlit Secrets; fallback is None
API_KEY = st.secrets.get("GOOGLE_FACTCHECK_API_KEY", None)

# ===============================
# 🧹 Helper Function: Clean text
# ===============================
def clean_text(text):
    """
    Cleans and normalizes text strings:
    - Fixes unicode issues using ftfy
    - Removes extra whitespace
    """
    if not text:
        return ""
    text = fix_text(text)
    return " ".join(text.split()).strip()

# ===============================
# 🔍 Function: Fetch Fact-Check Results
# ===============================
def get_fact_check_results(query):
    """
    Calls the Google Fact Check Tools API to find
    fact-checks related to the given query string.
    Returns a list of dictionaries (publisher, title, rating, url).
    """
    if not API_KEY:
        st.error("⚠️ Missing API Key. Please set GOOGLE_FACTCHECK_API_KEY in Streamlit secrets.")
        return []

    api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": API_KEY}

    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        claims = data.get("claims", [])

        results = []
        for claim in claims:
            for review in claim.get("claimReview", []):
                results.append({
                    "publisher": review.get("publisher", {}).get("name", "Unknown"),
                    "title": review.get("title", "No title available"),
                    "rating": review.get("textualRating", "No Rating"),
                    "url": review.get("url", "#")
                })
        return results

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return []

# ===============================
# 🎨 Streamlit Page Setup
# ===============================
def configure_page():
    """Set up page settings and dark theme styling."""
    st.set_page_config(page_title="📰 Global FactCheck Network", layout="wide")

    # Inject custom dark theme CSS for a news-portal look
    st.markdown("""
    <style>
        /* General app background */
        .stApp {
            background: linear-gradient(180deg, #05070a, #0b0f12);
            color: #e9f4ff;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Top header banner */
        .topbar {
            background: linear-gradient(90deg, #071029, #3b0f34);
            padding: 18px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        }
        .topbar h1 {
            margin: 0;
            font-size: 2.2rem;
            color: #fff;
        }
        .topbar p {
            margin-top: 5px;
            color: #bcd7ef;
            opacity: 0.9;
        }

        /* Section containers */
        .panel {
            background: rgba(255,255,255,0.03);
            padding: 16px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 20px;
        }

        /* Result cards */
        .card {
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 12px;
            border: 1px solid rgba(255,255,255,0.05);
        }

        .small { color:#9fb1c6; font-size:0.85rem; }
        .muted { color:#99aebf; }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# 🧠 Main App Function
# ===============================
def app():
    """Main Streamlit application."""
    configure_page()

    # --- Header ---
    st.markdown("""
    <div class="topbar">
        <h1>AI vs. Fact — Global FactCheck Network</h1>
        <p>Verify any headline or claim using trusted global fact-checking sources.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Tabs ---
    tabs = st.tabs(["🏠 Home", "🔍 Fact Check"])

    # -----------------------------
    # 🏠 HOME TAB
    # -----------------------------
    with tabs[0]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Welcome to the Global FactCheck Network")
        st.write("""
        This tool lets you:
        - ✅ Search for claims verified by fact-checking organizations  
        - 🌐 Fetch results directly from Google Fact Check Tools API  
        - 📰 Explore trustworthy verification sources in a single portal  
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # 🔍 FACT-CHECK TAB
    # -----------------------------
    with tabs[1]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Search for a Claim or Headline")

        # Input area
        query = st.text_input("Enter your claim or news headline:")

        if st.button("Check Facts"):
            if not query.strip():
                st.warning("⚠️ Please enter a statement or headline to verify.")
            else:
                with st.spinner("Fetching verified fact-checks..."):
                    results = get_fact_check_results(query)

                # --- Display results ---
                if not results:
                    st.info("No verified fact-checks found for this claim.")
                else:
                    st.success(f"✅ Found {len(results)} fact-check(s):")

                    for item in results:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(f"**Source:** <span class='small'>{item['publisher']}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Verdict:** <span class='muted'>{item['rating']}</span>", unsafe_allow_html=True)
                        st.markdown(f"[Read Full Article]({item['url']})")
                        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# 🚀 Run the App
# ===============================
if __name__ == "__main__":
    app()
