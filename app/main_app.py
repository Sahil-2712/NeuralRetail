from turtle import color
from webbrowser import BackgroundBrowser

import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NeuralRetail Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---------------- THEME SESSION STATE ----------------
if "theme" not in st.session_state:
    st.session_state.theme = True

# ---------------- THEME TOGGLE ----------------

theme = st.sidebar.toggle(
    "☀️ Light Mode" if st.session_state.theme else "🌙 Dark Mode",
    value=st.session_state.theme,
    key="theme_toggle"
)

# Save current theme
st.session_state.theme = theme

# Update theme immediately
st.session_state.theme = theme

# ---------------- COLORS ----------------
if st.session_state.theme:
    bg_color = "#0F172A"
    card_color = "#1E293B"
    text_color = "white"
    sub_text = "#94A3B8"
else:
    bg_color = "#F8FAFC"
    card_color = "#E2E8F0"
    text_color = "#0F172A"
    sub_text = "#475569"

# ---------------- CUSTOM CSS ----------------
st.markdown(f"""
<style>

.stApp {{
    background-color: {bg_color};
}}
/* Main Title */
.main-title {{
    font-size: 52px;
    font-weight: bold;
    color: {text_color};
    text-align: center;
    margin-top: 20px;
}}

.sub-title {{
    font-size: 22px;
    color: {sub_text};
    text-align: center;
    margin-bottom: 50px;
}}

/* Feature Boxes */
.feature-box {{
    background-color: {card_color};
    padding: 30px;
    border-radius: 20px;
    margin-top: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    transition: 0.3s;
}}

.feature-box:hover {{
    transform: translateY(-5px);
}}

.feature-title {{
    color: {text_color};
    font-size: 24px;
    font-weight: bold;
}}

.feature-text {{
    color: {sub_text};
    font-size: 16px;
    margin-top: 10px;
}}

/* Footer */
.footer {{
    text-align: center;
    color: {sub_text};
    margin-top: 60px;
    font-size: 16px;
}}
/* Big Buttons */
.stButton > button {{
    background-color: #1E293B;
    color: white;
    height: 90px;
    border-radius: 20px;
    border: none;
    font-size: 28px;
    font-weight: bold;
    transition: 0.3s;
}}
.stButton > button:hover {{
    background-color: #334155;
    transform: scale(1.02);
}}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🛒 NeuralRetail")

st.sidebar.markdown("---")


# ---------------- TITLE ----------------
st.markdown(f"""
<div class='main-title'>
🛒 NeuralRetail Dashboard
</div>

<div class='sub-title'>
AI Powered Retail Business Intelligence & Analytics Platform
</div>
""", unsafe_allow_html=True)

# ---------------- HOME BUTTON ----------------
st.page_link(
    "main_app.py",
    label="Home",
    icon="🏠"
)

st.write("")
st.write("")
# ---------------- MODULES ----------------
st.subheader("🚀 Dashboard Modules")

left, right = st.columns(2)

with left:

    if st.button("📈 Demand Forecasting", use_container_width=True):
        st.switch_page("pages/forecast_page.py")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("👥 Customer Segmentation", use_container_width=True):
        st.switch_page("pages/segment_page.py")

with right:

    if st.button("⚠️ Churn Prediction", use_container_width=True):
        st.switch_page("pages/churn_page.py")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("📦 Inventory Optimization", use_container_width=True):
        st.switch_page("pages/inventory_page.py")
# ---------------- FOOTER ----------------
st.markdown(f"""
<div class='footer'>
Built with ❤️ using Streamlit | NeuralRetail Analytics Platform
</div>
""", unsafe_allow_html=True)