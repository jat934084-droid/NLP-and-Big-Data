"""
google_factcheck_test.py — Global FactCheck Explorer (Dark Theme)

This standalone Streamlit app lets users verify a headline or statement
against verified sources using the Google Fact Check Tools API.
"""

# ===============================
# 🧩 Import Required Libraries
# ===============================
import streamlit as st
import requests
from urllib.parse import quote_plus

# ===============================
# 🔑 API Key Configuration
# ===============================
# Try to read API key from Streamlit secrets; fallback to your original hardcoded key
DEFAULT_API_KEY = "AIzaSyDmFciPOWcIuxDKilN1WO-SmMkwXUxZrUE"
API_KEY = st.secrets.get("Factcheck_key", DEFAULT_API_KEY)

# ===============================
# 🎨 Page Setup and Styling
# ===============================
st.set_page_config(page_title="📰 Global FactCheck Explorer", layout="wide")

# Dark theme styling for professional news portal look
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0a0d12, #0e1217);
    color: #E6EEF3;
    font-family: 'Segoe UI', sans-serif;
}

/* Header styling */
.header {
    background: linear-gradient(90deg, #0e1a3b, #410f2b);
    padding: 24px;
    border-radius: 10px;
    margin-bottom: 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}
.header h1 {
    margin: 0;
    font-size: 2.4rem;
    color: #ffffff;
}
.header p {
    color: #cfd9e1;
    margin-top: 5px;
    font-size: 1rem;
    opacity: 0.9;
}

/* Input + button section */
.search-container {
    background: rgba(255,255,255,0.03);
    padding: 20px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 20px;
}
input {
    color: #fff !important;
}

/* Button styling */
button[kind="primary"] {
    background: linear-gradient(90deg, #ff416c, #3f51b5);
    border: none;
    color: white !important;
}

/* Result cards */
.card {
    background: rgba(255,255,255,0.03);
    padding: 14px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 12px;
}
.card h3 {
    color: #f3f7ff;
}
.meta {
    color: #9fb1c6;
    font-size: 0.9rem;
    margin-bottom: 6px;
}
.link {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.06);
    color: #dceefc;
    text-decoration: none;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 🧭 Header Section
# ===============================
st.markdown("""
<div class="header">
    <h1>🧠 Global FactCheck Explorer</h1>
    <p>Verify any news headline or claim against trusted global fact-checking sources.</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# 💬 Input Section
# ===============================
st.markdown('<div class="search-container">', unsafe_allow_html=True)
query = st.text_input("Enter a news headline or claim to verify:")

col1, col2 = st.columns([1, 1])
with col1:
    check_btn = st.button("🔍 Check Fact")
with col2:
    clear_btn = st.button("🧹 Clear")
st.markdown('</div>', unsafe_allow_html=True)

# Reset the text field when "Clear" is pressed
if clear_btn:
    st.session_state["query"] = ""

# ===============================
# 🔍 API Query and Display Results
# ===============================
if check_btn:
    if not query or not query.strip():
        st.warning("⚠️ Please enter a valid headline or statement.")
    else:
        encoded_query = quote_plus(query.strip())
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={encoded_query}&key={API_KEY}"

        with st.spinner("Fetching fact-check results..."):
            try:
                response = requests.get(url, timeout=15)
                data = response.json()
            except Exception as e:
                st.error(f"❌ Failed to connect to API: {e}")
                data = {}

        # --- Display results ---
        if "claims" in data and data["claims"]:
            st.success(f"✅ Found {len(data['claims'])} related fact-checks:")
            for claim in data["claims"]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"### 🗞️ {claim.get('text', 'No claim text available')}")

                for review in claim.get("claimReview", []):
                    publisher = review.get("publisher", {}).get("name", "Unknown Source")
                    rating = review.get("textualRating", "No Rating")
                    review_url = review.get("url", "#")

                    st.markdown(f"<div class='meta'><b>Source:</b> {publisher}</div>", unsafe_allow_html=True)
                    st.markdown(f"<b>Verdict:</b> {rating}")
                    st.markdown(f"<a class='link' href='{review_url}' target='_blank'>Read Full Article</a>", unsafe_allow_html=True)
                    st.markdown("---")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No verified fact-checks found for this claim.")
