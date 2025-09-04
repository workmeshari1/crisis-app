import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import json

# --- إخفاء كامل لعناصر Streamlit وGitHub ---
st.set_page_config(
    page_title="نظام إدارة الكوارث والأزمات",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# إخفاء جميع العناصر غير المرغوب فيها
hide_ui_elements = """
<style>
/* إخفاء عناصر Streamlit */
#MainMenu, header, footer {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="stDecoration"] {display: none;}
[data-testid="baseButton-header"] {display: none;}
[data-testid="stToolbar"] {display: none;}
[href*="github"] {display: none;}
[href*="streamlit"] {display: none;}

/* تنسيقات التطبيق */
.stApp {
    background-image: url("https://github.com/workmeshari1/disaster-app/blob/6b907779e30e18ec6ebec68b90e2558d91e5339b/assets.png?raw=true");
    background-size: cover;
    background-position: center top;
    background-repeat: no-repeat;
    background-attachment: fixed;
    min-height: 100vh;
    padding-top: 80px;
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
}
h2 {
    font-size: 20px !important;
    color: #ffffff;
}
h3 {
    font-size: 18px !important;
    color: #ffffff;
}

/* تحسينات للأداء */
.stSpinner > div {display: none;}
</style>
"""
st.markdown(hide_ui_elements, unsafe_allow_html=True)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# --- تحميل الموديل مرة واحدة ---
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {str(e)}")
        return None

# --- قراءة البيانات + كلمة المرور من الشيت ---
@st.cache_data(ttl=600, show_spinner=False)
def load_data_and_password():
    try:
        creds_json = os.getenv("GOOGLE_CREDENTIALS")
        sheet_id = os.getenv("SHEET_ID")
        
        if not creds_json and hasattr(st, 'secrets') and "GOOGLE_CREDENTIALS" in st.secrets:
            creds_json = json.dumps(dict(st.secrets["GOOGLE_CREDENTIALS"]))
            if "id" in st.secrets["SHEET"]:
                sheet_id = st.secrets.SHEET["id"]
            else:
                raise ValueError("❌ 'id' is missing in the secrets.toml SHEET section.")

        if not creds_json or not sheet_id:
            raise ValueError("❌ لم يتم العثور على المتغيرات السرية")
        
        creds_info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id)
        ws = sheet.sheet1

        data = ws.get_all_records()
        df = pd.DataFrame(data)
        password_value = ws.cell(1, 5).value
        return df, password_value
    except Exception as e:
        st.error(f"❌ فشل الاتصال بقاعدة البيانات: {str(e)}")
        st.info("تأكد من إعداد متغيرات البيئة GOOGLE_CREDENTIALS و SHEET_ID")
        st.stop()

# --- حساب إمبادنج للوصف ---
@st.cache_data(show_spinner=False)
def compute_embeddings(descriptions: list[str]):
    model = load_model()
    if model is None:
        return None
    return model.encode(descriptions, convert_to_tensor=True)

# --- دالة للتحقق من الأرقام ضمن نطاق أو قيمة مفردة ---
def is_number_in_range(number, synonym):
    try:
        if "-" in synonym:
            parts = synonym.split("-")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                return False
            min_val = int(parts[0])
            max_val = float('inf') if parts[1] in ["∞", "inf"] else int(parts[1])
            return min_val <= number <= max_val
        else:
            return number == int(synonym)
    except ValueError:
        return False

def process_number_input(q, df, syn_col, action_col):
    try:
        number = int(q)
        matched_row = None

        for _, row in df.iterrows():
            synonyms = str(row.get(syn_col, "")).strip()
            if not synonyms:
                continue

            for syn in synonyms.split(","):
                syn = syn.strip()
                if not syn:
                    continue
                if is_number_in_range(number, syn):
                    matched_row = row
                    break
            if matched_row is not None:
                break

        if matched_row is not None:
            st.markdown(
                f"""
                <div style='background:#1f1f1f;color:#fff;padding:14px;border-radius:10px;
                            direction:rtl;text-align:right;font-size:18px;margin-bottom:12px;'>
                    <div style="font-size:22px;margin-bottom:8px;">🔢 نتيجة رقمية</div>
                    <b>الوصف:</b> {matched_row.get("وصف الحالة أو الحدث", "—")}<br>
                    <b>الإجراء:</b>
                    <span style='background:#ff6600;color:#fff;padding:6px 10px;border-radius:6px;
                                 display:inline-block;margin-top:6px;'>{matched_row[action_col]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return True
        else:
            st.warning("❌ لم يتم العثور على تطابق للرقم المدخل.")
            return False
    except ValueError:
        return False

# ============== واجهة التطبيق ==============
def main():
    st.title("⚡ دائرة إدارة الكوارث والأزمات الصناعية")

    # جرب تحميل البيانات
    try:
        df, PASSWORD = load_data_and_password()
    except Exception as e:
        st.error(f"❌ فشل الاتصال بقاعدة البيانات: {str(e)}")
        st.stop()

    # التحقق من الأعمدة المطلوبة
    DESC_COL = "وصف الحالة أو الحدث"
    ACTION_COL = "الإجراء"
    SYN_COL = "مرادفات للوصف"

    if df.empty:
        st.error("❌ لا توجد بيانات في الجدول.")
        st.stop()

    for col in [DESC_COL, ACTION_COL]:
        if col not in df.columns:
            st.error(f"عمود مفقود في Google Sheet: '{col}'.")
            st.stop()

    if SYN_COL not in df.columns:
        df[SYN_COL] = ""

    # تسجيل الدخول
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        password = st.text_input("الرقم السري", type="password")
        if st.button("دخول"):
            if password == str(PASSWORD):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ الرقم السري غير صحيح")
        st.stop()

    # بعد التحقق
    query = st.text_input("ابحث هنا:", placeholder="اكتب وصف الحالة…")
    if not query:
        st.stop()

    q = query.strip().lower()

    # --------- 🔢 معالجة الأرقام ---------
    if process_number_input(q, df, SYN_COL, ACTION_COL):
        st.stop()

    # --------- 📝 البحث النصي ---------
    words = [w for w in q.split() if w]
    literal_results = []
    synonym_results = []

    # 1) الحرفي من الوصف
    for _, row in df.iterrows():
        text = str(row[DESC_COL]).lower()
        if all(w in text for w in words):
            literal_results.append(row)

    # 2) الحرفي من المرادفات
    if not literal_results:
        for _, row in df.iterrows():
            syn_text = str(row.get(SYN_COL, "")).lower()
            if any(w in syn_text for w in words):
                synonym_results.append(row)

    # عرض النتائج
    def render_card(r, icon="🔶"):
        st.markdown(
            f"""
            <div style='background:#1f1f1f;color:#fff;padding:12px;border-radius:8px;direction:rtl;text-align:right;font-size:18px;margin-bottom:10px;'>
                <div style="font-size:22px;margin-bottom:6px;">{icon} </div>
                <b>الوصف:</b> {r[DESC_COL]}<br>
                <b>الإجراء:</b>
                <span style='background:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:6px;display:inline-block;margin-top:4px;'>
                    {r[ACTION_COL]}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if literal_results:
        st.subheader("🔍 النتائج المطابقة:")
        for r in literal_results[:5]:
            render_card(r, "🔍")
    elif synonym_results:
        st.subheader("📌 يمكن قصدك:")
        for r in synonym_results[:3]:
            render_card(r, "📌")
    else:
        st.warning("❌ لم يتم العثور على نتائج.. وش رايك تستخدم البحث الذكي 👇")
        if st.button("🤖 البحث الذكي"):
            try:
                with st.spinner("جاري البحث الذكي..."):
                    model = load_model()
                    if model is None:
                        st.error("❌ النموذج غير متاح")
                        return
                        
                    descriptions = df[DESC_COL].fillna("").astype(str).tolist()
                    if not descriptions:
                        st.error("❌ لا توجد أوصاف صالحة في البيانات.")
                        return
                        
                    embeddings = compute_embeddings(descriptions)
                    if embeddings is None:
                        st.error("❌ فشل في حساب التضمينات")
                        return
                        
                    query_embedding = model.encode(query, convert_to_tensor=True)
                    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
                    top_scores, top_indices = torch.topk(cosine_scores, k=min(5, len(df)))
                    st.subheader("🧐 يمكن قصدك:")
                    found_results = False
                    for score, idx in zip(top_scores, top_indices):
                        if float(score) > 0.3:
                            found_results = True
                            r = df.iloc[int(idx.item())]
                            st.markdown(
                                f"""
                                <div style='background:#444;color:#fff;padding:12px;border-radius:8px;direction:rtl;text-align:right;font-size:18px;margin-bottom:10px;'>
                                    <div style="font-size:22px;margin-bottom:6px;">🤖 </div>
                                    <b>الوصف:</b> {r[DESC_COL]}<br>
                                    <b>الإجراء:</b>
                                    <span style='background:#ff6600;color:#0a1e3f;padding:4px 8px;border-radius:6px;display:inline-block;margin-top:4px;'>
                                        {r[ACTION_COL]}
                                    </span><br>
                                    <span style='font-size:14px;color:orange;'>درجة التشابه: {float(score):.2f}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    if not found_results:
                        st.info("🤖 لم يتم العثور على نتائج مشابهة كافية. حاول إعادة صياغة سؤالك.")
            except Exception as e:
                st.error(f"❌ خطأ في البحث الذكي: {str(e)}")

    # شريط جانبي
    with st.sidebar:
        st.markdown("### معلومات النظام")
        st.info(f"📊 عدد الحالات المسجلة: {len(df)}")
        st.info("🔄 تحديث البيانات: كل 10 دقائق")
        if st.button("🔒 تسجيل خروج"):
            st.session_state.authenticated = False
            st.rerun()

    # Footer مخصص
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; direction: rtl;'>
        نظام إدارة الكوارث والأزمات الذكية
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
