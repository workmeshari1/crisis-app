import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import json

# إعداد الصفحة
st.set_page_config(
    page_title="⚡ إدارة الكوارث والأزمات",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ✅ CSS مُعدّل: يخفي فقط العناصر غير المرغوب فيها بدون إخفاء الحقول
hide_streamlit_style = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton, [data-testid="stDecoration"], [data-testid="stToolbar"],
[href*="github"], [href*="streamlit"] {
    display: none !important;
}

.stApp {
    background-image: url("https://github.com/workmeshari1/disaster-app/blob/6b907779e30e18ec6ebec68b90e2558d91e5339b/assets.png?raw=true");
    background-size: cover;
    background-position: center top;
    background-repeat: no-repeat;
    background-attachment: fixed;
    min-height: 100vh;
    padding-top: 80px;
}

h1 {
    font-size: 26px !important;
    color: #ffffff;
    text-align: center;
    margin-top: -60px;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# تحميل الموديل
@st.cache_resource
def load_model():
    return SentenceTransformer("Omartificial-Intelligence-Space/Arabert-all-nli-triplet-Matryoshka")

# تحميل البيانات وكلمة المرور
@st.cache_data(ttl=600)
def load_data_and_password():
    creds_json = os.getenv("GOOGLE_CREDENTIALS")
    sheet_id = os.getenv("SHEET_ID")

    if not creds_json and hasattr(st, 'secrets') and "GOOGLE_CREDENTIALS" in st.secrets:
        creds_json = json.dumps(dict(st.secrets["GOOGLE_CREDENTIALS"]))
        sheet_id = st.secrets.SHEET.get("id")

    if not creds_json or not sheet_id:
        raise ValueError("❌ المتغيرات السرية غير موجودة.")

    creds_info = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    ws = sheet.sheet1

    data = ws.get_all_records()
    df = pd.DataFrame(data)
    password_value = ws.cell(1, 5).value
    return df, password_value

# ========== الواجهة ==========

st.markdown("<h1>⚡ دائرة إدارة الكوارث والأزمات الصناعية</h1>", unsafe_allow_html=True)

# تحميل البيانات
try:
    df, PASSWORD = load_data_and_password()
except Exception as e:
    st.error(f"❌ فشل في تحميل البيانات: {str(e)}")
    st.stop()

# ✅ التحقق من كلمة المرور
if not PASSWORD:
    st.error("⚠️ لم يتم تحميل كلمة المرور من Google Sheet (F1).")
    st.stop()

# التحقق من حالة الجلسة
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ✅ عرض حقل الرقم السري
if not st.session_state.authenticated:
    password_input = st.text_input("🔐 أدخل الرقم السري", type="password")
    if st.button("دخول"):
        if password_input == str(PASSWORD).strip():
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ الرقم السري غير صحيح")
    st.stop()

# ✅ بعد الدخول - ابدأ بعرض البحث أو أي مكونات أخرى
st.success("✅ تم تسجيل الدخول بنجاح. الآن يمكنك البحث 👇")
