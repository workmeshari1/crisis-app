import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import json

# ========== إعداد الصفحة ==========
st.set_page_config(
    page_title="⚡ إدارة الكوارث والأزمات",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== CSS لإخفاء كل ما يتعلق بـ Streamlit وGitHub ==========
hide_ui = """
<style>
#MainMenu, header, footer {visibility: hidden;}
.stDeployButton, [data-testid="stDecoration"], [data-testid="stToolbar"],
[href*="github"], [href*="streamlit"], [data-testid="stHeader"] {
    display: none !important;
}
.stApp {
    background-image: url("https://github.com/workmeshari1/disaster-app/blob/6b907779e30e18ec6ebec68b90e2558d91e5339b/assets.png?raw=true");
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
    min-height: 100vh;
    padding-top: 80px;
}
h1 { font-size: 26px !important; color: #fff; text-align: center; margin-top: -60px; }
</style>
"""
st.markdown(hide_ui, unsafe_allow_html=True)

# ========== منع كليك يمين وF11 وCtrl+F ========== (اختياري)
st.markdown("""
<script>
document.addEventListener('contextmenu', event => event.preventDefault());
document.addEventListener('keydown', function(event) {
    if (event.key === 'F11') event.preventDefault();
    if ((event.ctrlKey || event.metaKey) && event.key === 'f') {
        event.preventDefault();
    }
});
</script>
""", unsafe_allow_html=True)

# ========== تحميل الموديل ========== 
@st.cache_resource
def load_model():
    return SentenceTransformer("Omartificial-Intelligence-Space/Arabert-all-nli-triplet-Matryoshka")

# ========== تحميل البيانات وكلمة المرور ========== 
@st.cache_data(ttl=600)
def load_data_and_password():
    creds_json = os.getenv("GOOGLE_CREDENTIALS")
    sheet_id = os.getenv("SHEET_ID")

    if not creds_json and hasattr(st, 'secrets') and "GOOGLE_CREDENTIALS" in st.secrets:
        creds_json = json.dumps(dict(st.secrets["GOOGLE_CREDENTIALS"]))
        sheet_id = st.secrets.SHEET.get("id")

    if not creds_json or not sheet_id:
        raise ValueError("لم يتم تقديم بيانات الاعتماد لـ Google Sheet.")

    creds_info = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).sheet1

    df = pd.DataFrame(sheet.get_all_records())
    password = sheet.cell(1, 5).value
    return df, password

# ========== وظائف البحث ========== 
@st.cache_data
def compute_embeddings(descriptions: list[str]):
    model = load_model()
    return model.encode(descriptions, convert_to_tensor=True)

def is_number_in_range(number, syn):
    try:
        if "-" in syn:
            parts = syn.split("-")
            min_val = int(parts[0].strip())
            max_val = float('inf') if parts[1].strip() in ["∞", "inf"] else int(parts[1].strip())
            return min_val <= number <= max_val
        return number == int(syn.strip())
    except:
        return False

def process_number_input(q, df, syn_col, action_col, desc_col):
    try:
        number = int(q)
        for _, row in df.iterrows():
            syns = str(row.get(syn_col, "")).split(",")
            for syn in syns:
                if is_number_in_range(number, syn):
                    st.markdown(f"""
                    <div style='background:#1f1f1f;color:#fff;padding:14px;border-radius:10px;
                                direction:rtl;text-align:right;font-size:18px;margin-bottom:12px;'>
                        <div style="font-size:22px;margin-bottom:8px;">🔢 نتيجة رقمية</div>
                        <b>الوصف:</b> {row.get(desc_col,"—")}<br>
                        <b>الإجراء:</b>
                        <span style='background:#ff6600;color:#fff;padding:6px 10px;border-radius:6px;
                                     display:inline-block;margin-top:6px;'>{row[action_col]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    return True
        return False
    except:
        return False

# ========== عرض النتائج ========== 
def render_card(r, desc_col, action_col, icon):
    st.markdown(f"""
    <div style='background:#1f1f1f;color:#fff;padding:12px;border-radius:8px;
                direction:rtl;text-align:right;font-size:18px;margin-bottom:10px;'>
        <div style="font-size:22px;margin-bottom:6px;">{icon}</div>
        <b>الوصف:</b> {r[desc_col]}<br>
        <b>الإجراء:</b>
        <span style='background:#ff6600;color:#fff;padding:4px 8px;border-radius:6px;
                     display:inline-block;margin-top:4px;'>{r[action_col]}</span>
    </div>
    """, unsafe_allow_html=True)

# ========== واجهة التطبيق ==========
st.markdown("<h1>⚡ دائرة إدارة الكوارث والأزمات الصناعية</h1>", unsafe_allow_html=True)

# تحميل البيانات
try:
    df, PASSWORD = load_data_and_password()
except Exception as e:
    st.error(f"فشل تحميل البيانات: {e}")
    st.stop()

DESC_COL = "وصف الحالة أو الحدث"
ACTION_COL = "الإجراء"
SYN_COL = "مرادفات للوصف"

if df.empty or DESC_COL not in df.columns or ACTION_COL not in df.columns:
    st.error("تحقق من وجود الأعمدة الأساسية (الوصف و الإجراء) وبيانات صالحة.")
    st.stop()
if SYN_COL not in df.columns:
    df[SYN_COL] = ""

# تحقق من كلمة المرور
if not PASSWORD:
    st.error("لم يتم تحميل كلمة المرور من Google Sheet (F1).")
    st.stop()

# حالة الجلسة
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# خانة الرقم السري
if not st.session_state.authenticated:
    pwd = st.text_input("🔐 أدخل الرقم السري", type="password")
    if st.button("دخول"):
        if pwd.strip() == str(PASSWORD).strip():
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("الرقم السري غير صحيح.")
    st.stop()

# بعد تسجيل الدخول
query = st.text_input("📝 اكتب وصف الحالة هنا:", placeholder="مثال: تسرب غاز سام")
if not query:
    st.stop()

q = query.strip().lower()
# === بحث رقمي أولًا ===
if process_number_input(q, df, SYN_COL, ACTION_COL, DESC_COL):
    st.stop()

# === البحث النصي ===
words = q.split()
literal = [row for _, row in df.iterrows() if all(w in str(row[DESC_COL]).lower() for w in words)]
synonyms = []
if not literal:
    for _, row in df.iterrows():
        if any(w in str(row.get(SYN_COL, "")).lower() for w in words):
            synonyms.append(row)

if literal:
    st.markdown("<h3 style='text-align:right;'>🔍 النتائج المطابقة:</h3>", unsafe_allow_html=True)
    for r in literal[:5]:
        render_card(r, DESC_COL, ACTION_COL, "🔍")
elif synonyms:
    st.markdown("<h3 style='text-align:right;'>📌 يمكن قصدك:</h3>", unsafe_allow_html=True)
    for r in synonyms[:3]:
        render_card(r, DESC_COL, ACTION_COL, "📌")
else:
    st.warning("❌ لم يتم العثور على نتائج.. وش رايك تستخدم البحث الذكي 👇")
    if st.button("🤖 البحث الذكي"):
        descriptions = df[DESC_COL].fillna("").astype(str).tolist()
        embeddings = compute_embeddings(descriptions)
        query_emb = load_model().encode(query, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
        top_scores, top_idxs = torch.topk(cosine_scores, k=min(5, len(df)))
        found = False
        for score, idx in zip(top_scores, top_idxs):
            for score, idx in zip(top_scores, top_idxs):
                found = True
                r = df.iloc[int(idx.item())]
                render_card(r, DESC_COL, ACTION_COL, "🤖")
        if not found:
            st.info("لم نتمكن من العثور على نتائج مشابهة كافية.")




