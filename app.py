"""
app.py — Global FactCheck Network (Dark Theme)
Author: Ashish Jat
Description:
A Streamlit app to verify news claims using the Google Fact Check Tools API.
Cleaned and optimized for Streamlit Cloud deployment.
"""

# ===============================
# 🧩 Import Libraries
# ===============================
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from ftfy import fix_text
import time
import random

# ===============================
# 🔑 API Key Configuration
# ===============================
# Load API key from Streamlit secrets for safety
API_KEY = st.secrets.get("Factcheck_key", None)

# ===============================
# 🧹 Helper Function: Clean Text
# ===============================
def clean_text(text):
    """Fix text encoding and remove extra spaces."""
    if not text:
        return ""
    text = fix_text(text)
    return " ".join(text.split()).strip()

# ===============================
# 🔍 Fetch Fact Check Results
# ===============================
def get_fact_check_results(query):
    """
    Fetches fact-checking data from Google Fact Check Tools API
    based on the user's query.
    """
    if not API_KEY:
        st.warning("⚠️ No API key found. Add it under Streamlit Secrets.")
        return []

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": API_KEY}

    try:
        response = requests.get(url, params=params, timeout=15)
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
# 🎨 Page Configuration & Styling
# ===============================
def configure_page():
    """Set up page layout and custom dark theme."""
    st.set_page_config(page_title="📰 Global FactCheck Network", layout="wide")

    st.markdown("""
    <style>
        /* Background + font */
        .stApp {
            background: linear-gradient(180deg, #05070a, #0b0f12);
            color: #e9f4ff;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Header banner */
        .topbar {
            background: linear-gradient(90deg, #071029, #3b0f34);
            padding: 20px;
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

        /* Section panels */
        .panel {
            background: rgba(255,255,255,0.03);
            padding: 16px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 20px;
        }

        /* Result cards */
        .card {
            background: rgba(255,255,255,0.02);
            padding: 14px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 12px;
        }

        .small { color:#9fb1c6; font-size:0.85rem; }
        .muted { color:#99aebf; }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# 🧠 Main Streamlit App
# ===============================
def app():
    """Main app entry point."""
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
    # 🏠 Home Tab
    # -----------------------------
    with tabs[0]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Welcome to the Global FactCheck Network")
        st.write("""
        This tool allows you to:
        - ✅ Search and verify political or news claims  
        - 🌐 Fetch results from the **Google Fact Check Tools API**  
        - 📰 Explore verified information from trusted publishers  
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # 🔍 Fact Check Tab
    # -----------------------------
    with tabs[1]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Search a Headline or Statement")

        query = st.text_input("Enter a news headline or claim:")

        if st.button("Check Facts"):
            if not query.strip():
                st.warning("⚠️ Please enter a valid claim to check.")
            else:
                with st.spinner("Fetching fact-check results..."):
                    results = get_fact_check_results(query)

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
# 🚀 Run App
# ===============================
if __name__ == "__main__":
    app()
