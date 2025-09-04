import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import json
import sys
import codecs

# إصلاح ترميز الأحرف العربية للنظم المختلفة
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'UTF-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# --- إخفاء كامل لجميع عناصر Streamlit وGitHub ---
st.set_page_config(
    page_title="نظام إدارة الكوارث والأزمات",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# CSS مخصص لإخفاء كل العناصر غير المرغوب فيها مع تحسين الترميز
hide_all_elements = """
<meta charset="UTF-8">
<style>
/* إخفاء كل عناصر Streamlit والروابط الخارجية */
[data-testid="stDecoration"],
.stDeployButton,
footer,
[data-testid="baseButton-header"],
[href*="github"],
[href*="streamlit"],
[data-testid="stToolbar"],
[data-testid="stHeader"],
[aria-label="View app source"],
[aria-label="View app source code"],
[data-testid="stAppViewContainer"] > header,
[data-testid="stSidebarContent"] > div:first-child,
[data-testid="stSidebarContent"] > div:first-child + div {
    display: none !important;
}

/* تنسيقات مخصصة للتطبيق */
.stApp {
    background-image: url("https://github.com/workmeshari1/disaster-app/blob/6b907779e30e18ec6ebec68b90e2558d91e5339b/assets.png?raw=true");
    background-size: cover;
    background-position: center top;
    background-repeat: no-repeat;
    background-attachment: fixed;
    min-height: 100vh;
    padding-top: 80px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* تحسين الترميز للنصوص العربية */
* {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

@media only screen and (max-width: 768px) {
    .stApp {
        background-size: cover;
        background-position: center top;
        padding-top: 60px;
    }
}

h1 {
    font-size: 26px !important;
    color: #ffffff;
    text-align: center;
    margin-top: -60px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
h2 {
    font-size: 20px !important;
    color: #ffffff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
h3 {
    font-size: 18px !important;
    color: #ffffff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* تحسينات للهواتف */
@media (max-width: 480px) {
    .stTextInput input {
        font-size: 16px !important;
    }
    
    .stButton button {
        width: 100% !important;
        margin: 5px 0 !important;
    }
}
</style>
"""
st.markdown(hide_all_elements, unsafe_allow_html=True)

# ... rest of your existing code continues exactly as before ...