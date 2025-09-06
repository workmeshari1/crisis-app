import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import json

# --- كود debugging للتحقق من أن التطبيق يعمل ---
st.set_page_config(
    page_title="⚡ إدارة الكوارث والأزمات",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# رسالة تأكيد أن التطبيق يعمل
st.success("✅ التطبيق يعمل! جاري التحميل...")

# تحقق بسيط من المكتبات
try:
    st.write("🔍 جاري التحقق من المكتبات...")
    st.write(f"pandas version: {pd.__version__}")
    st.write(f"gspread version: {gspread.__version__}")
except Exception as e:
    st.error(f"❌ خطأ في المكتبات: {e}")

# استمر بالكود الأصلي هنا...
# ========== إعداد الصفحة ==========
st.set_page_config(
    page_title="⚡ إدارة الكوارث والأزمات",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# ========== إخفاء كامل لكل عناصر Streamlit وGitHub ==========
hide_ui = """
<style>
#MainMenu, header, footer {visibility: hidden;}
.stDeployButton, [data-testid="stDecoration"],
[data-testid="baseButton-header"], [href*="github"],
[href*="streamlit"], [data-testid="stToolbar"],
[data-testid="stHeader"], div[data-testid="stToolbar"],
div[role="tooltip"], button[title="View fullscreen"],
section[data-testid="stSidebar"] {
    display: none !important;
    visibility: hidden !important;
}
div[data-testid="stAppViewContainer"] > footer,
div[data-testid="stAppViewContainer"] > div:last-child {
    display: none !important;
    height: 0 !important;
}
.stApp {
    background-image: url("https://github.com/workmeshari1/disaster-app/blob/6b907779e30e18ec6ebec68b90e2558d91e5339b/assets.png?raw=true");
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
    min-height: 100vh;
    padding-top: 80px;
    user-select: none;
}
@media only screen and (max-width: 768px) {
    .stApp { padding-top: 60px; }
}
h1 { font-size: 26px !important; color: #fff; text-align: center; margin-top: -60px; }
h2 { font-size: 20px !important; color: #fff; }
h3 { font-size: 18px !important; color: #fff; }
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #ff6600, #e55a00) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: bold !important;
}
.stTextInput > div > div > input {
    border-radius: 8px !important;
    padding: 12px !important;
    background: rgba(255,255,255,0.9) !important;
}
div[data-testid="stVerticalBlock"] > div {
    background: rgba(31, 31, 31, 0.9) !important;
    border-radius: 10px !important;
    padding: 15px !important;
    margin-bottom: 15px !important;
}
a { display: none !important; }
.embedded .stApp { margin-top: -50px; }
.stMarkdown { text-align: right !important; direction: rtl !important; }
</style>
"""
st.markdown(hide_ui, unsafe_allow_html=True)

# ========== تعطيل كليك يمين + بعض الاختصارات ==========
st.markdown("""
<script>
document.addEventListener('contextmenu', event => event.preventDefault());
document.addEventListener('keydown', function(event) {
    if (event.key === 'F11') { event.preventDefault(); }
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
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

@st.cache_data(ttl=600)
def load_data_and_password():
    try:
        creds_json = os.getenv("GOOGLE_CREDENTIALS")
        sheet_id = os.getenv("SHEET_ID")

        if not creds_json and hasattr(st, 'secrets') and "GOOGLE_CREDENTIALS" in st.secrets:
            creds_json = json.dumps(dict(st.secrets["GOOGLE_CREDENTIALS"]))
            sheet_id = st.secrets.SHEET.get("id")

        if not creds_json or not sheet_id:
            raise ValueError("❌ بيانات Google Sheet مفقودة.")

        creds_info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id)
        ws = sheet.sheet1

        df = pd.DataFrame(ws.get_all_records())
        password_value = ws.cell(1, 5).value  # خلية F1
        return df, password_value
    except Exception as e:
        st.error(f"❌ فشل تحميل البيانات: {e}")
        st.stop()

# ========== Embeddings ==========
@st.cache_data
def compute_embeddings(descriptions: list[str]):
    model = load_model()
    return model.encode(descriptions, convert_to_tensor=True)

# ========== البحث بالرقم ==========
def is_number_in_range(number, synonym):
    try:
        if "-" in synonym:
            min_val, max_val = synonym.split("-")
            min_val = int(min_val.strip())
            max_val = float('inf') if max_val.strip() in ["∞", "inf"] else int(max_val.strip())
            return min_val <= number <= max_val
        return number == int(synonym.strip())
    except:
        return False

def process_number_input(q, df, syn_col, action_col):
    try:
        number = int(q)
        for _, row in df.iterrows():
            synonyms = str(row.get(syn_col, "")).strip()
            if not synonyms:
                continue
            for syn in synonyms.split(","):
                if is_number_in_range(number, syn):
                    st.markdown(f"""
                    <div style='background:#1f1f1f;color:#fff;padding:14px;border-radius:10px;
                                direction:rtl;text-align:right;font-size:18px;margin-bottom:12px;'>
                        <div style="font-size:22px;margin-bottom:8px;">🔢 نتيجة رقمية</div>
                        <b>الوصف:</b> {row.get("وصف الحالة أو الحدث", "—")}<br>
                        <b>الإجراء:</b>
                        <span style='background:#ff6600;color:#fff;padding:6px 10px;border-radius:6px;
                                     display:inline-block;margin-top:6px;'>{row[action_col]}</span>
                    </div>""", unsafe_allow_html=True)
                    return True
        st.warning("❌ لم يتم العثور على تطابق للرقم.")
        return False
    except:
        return False

# ========== واجهة المستخدم ==========
st.markdown("<h1>⚡ دائرة إدارة الكوارث والأزمات الصناعية</h1>", unsafe_allow_html=True)

# تحميل البيانات وكلمة المرور
df, PASSWORD = load_data_and_password()

DESC_COL = "وصف الحالة أو الحدث"
ACTION_COL = "الإجراء"
SYN_COL = "مرادفات للوصف"

if df.empty or DESC_COL not in df.columns or ACTION_COL not in df.columns:
    st.error("❌ تحقق من صحة الأعمدة ووجود بيانات.")
    st.stop()

if SYN_COL not in df.columns:
    df[SYN_COL] = ""

# ========== التحقق من الجلسة ==========
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    if not PASSWORD:
        st.error("⚠️ لم يتم تحميل كلمة المرور من Google Sheet (F1).")
        st.stop()

    password = st.text_input("الرقم السري", type="password")
    if st.button("دخول"):
        if password == str(PASSWORD):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ الرقم السري غير صحيح")
    st.stop()

# ========== بعد الدخول ==========
query = st.text_input("ابحث هنا:", placeholder="اكتب وصف الحالة…")
if not query:
    st.stop()

q = query.strip().lower()

# 🔢 بحث رقمي
if process_number_input(q, df, SYN_COL, ACTION_COL):
    st.stop()

# 📝 بحث نصي
words = q.split()
literal_results = []
synonym_results = []

for _, row in df.iterrows():
    text = str(row[DESC_COL]).lower()
    if all(w in text for w in words):
        literal_results.append(row)

if not literal_results:
    for _, row in df.iterrows():
        syn_text = str(row.get(SYN_COL, "")).lower()
        if any(w in syn_text for w in words):
            synonym_results.append(row)

def render_card(r, icon="🔶"):
    st.markdown(f"""
    <div style='background:#1f1f1f;color:#fff;padding:12px;border-radius:8px;
                direction:rtl;text-align:right;font-size:18px;margin-bottom:10px;'>
        <div style="font-size:22px;margin-bottom:6px;">{icon}</div>
        <b>الوصف:</b> {r[DESC_COL]}<br>
        <b>الإجراء:</b>
        <span style='background:#ff6600;color:#fff;padding:4px 8px;border-radius:6px;
                     display:inline-block;margin-top:4px;'>{r[ACTION_COL]}</span>
    </div>
    """, unsafe_allow_html=True)

if literal_results:
    st.markdown("<h3 style='text-align: right;'>🔍 النتائج المطابقة:</h3>", unsafe_allow_html=True)
    for r in literal_results[:5]:
        render_card(r, "🔍")
elif synonym_results:
    st.markdown("<h3 style='text-align: right;'>📌 يمكن قصدك:</h3>", unsafe_allow_html=True)
    for r in synonym_results[:3]:
        render_card(r, "📌")
else:
    st.warning("❌ لم يتم العثور على نتائج.. جرب البحث الذكي 👇")
    if st.button("🤖 البحث الذكي"):
        try:
            with st.spinner("جاري البحث الذكي..."):
                model = load_model()
                descriptions = df[DESC_COL].fillna("").astype(str).tolist()
                embeddings = compute_embeddings(descriptions)
                query_embedding = model.encode(query, convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
                top_scores, top_indices = torch.topk(cosine_scores, k=min(5, len(df)))
                found = False
                for score, idx in zip(top_scores, top_indices):
                    if float(score) > 0.3:
                        found = True
                        r = df.iloc[int(idx.item())]
                        st.markdown(f"""
                        <div style='background:#444;color:#fff;padding:12px;border-radius:8px;
                                    direction:rtl;text-align:right;font-size:18px;margin-bottom:10px;'>
                            <div style="font-size:22px;margin-bottom:6px;">🤖</div>
                            <b>الوصف:</b> {r[DESC_COL]}<br>
                            <b>الإجراء:</b>
                            <span style='background:#ff6600;color:#fff;padding:4px 8px;border-radius:6px;
                                         display:inline-block;margin-top:4px;'>{r[ACTION_COL]}</span><br>
                            <span style='font-size:14px;color:orange;'>درجة التشابه: {float(score):.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                if not found:
                    st.info("🤖 لم يتم العثور على نتائج مشابهة كافية.")
        except Exception as e:
            st.error(f"❌ خطأ في البحث الذكي: {e}")

# ========== زر تسجيل الخروج ==========
if st.button("🔒 تسجيل خروج", use_container_width=True):
    st.session_state.authenticated = False
    st.rerun()

