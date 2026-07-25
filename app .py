# -*- coding: utf-8 -*-
"""
app.py
تطبيق Streamlit لموسوعة الأعشاب الطبية — RAG عربي (BM25 + بحث دلالي + Groq للتوليد).
"""

import streamlit as st
import pandas as pd

from rag_core import build_herb_dataframe, HerbRetriever, build_context_package
from embedder import Embedder
from generation import generate_answer

DATA_PATH = "data/herbs_encyclopedia.txt"

st.set_page_config(
    page_title="موسوعة الأعشاب الطبية",
    page_icon="🌿",
    layout="centered",
)

# ============================================================
# الهوية البصرية — خطوط + ألوان + بطاقات مخصصة
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Tajawal:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        direction: rtl;
        font-family: 'Tajawal', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(76,107,79,0.06), transparent 40%),
            radial-gradient(circle at 88% 92%, rgba(139,111,62,0.07), transparent 40%),
            #F7F3EA;
    }

    /* ===== الترويسة ===== */
    .herbal-hero {
        text-align: center;
        padding: 1.4rem 0 1rem 0;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid #D9CFB0;
    }
    .herbal-hero h1 {
        font-family: 'Amiri', serif;
        font-weight: 700;
        font-size: 2.6rem;
        color: #2B3A2F;
        margin-bottom: 0.15rem;
        letter-spacing: 0.5px;
    }
    .herbal-hero p {
        font-family: 'Tajawal', sans-serif;
        font-weight: 400;
        color: #6B5B3A;
        font-size: 1rem;
        margin-top: 0;
    }
    .herbal-divider {
        text-align: center;
        color: #8B6F3E;
        letter-spacing: 6px;
        font-size: 0.85rem;
        margin: 0.4rem 0 1.6rem 0;
        opacity: 0.75;
    }

    /* ===== مربع السؤال ===== */
    div[data-testid="stTextInput"] input {
        background-color: #FFFFFF;
        border: 1.5px solid #D9CFB0;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-family: 'Tajawal', sans-serif;
        font-size: 1.02rem;
        color: #2B3A2F;
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #4C6B4F;
        box-shadow: 0 0 0 3px rgba(76,107,79,0.15);
    }

    /* ===== الزرار ===== */
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #4C6B4F, #3B5540);
        color: #F7F3EA;
        font-family: 'Tajawal', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 0;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
        box-shadow: 0 3px 10px rgba(43,58,47,0.18);
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(43,58,47,0.25);
        color: #F7F3EA;
    }

    /* ===== بطاقة الإجابة ===== */
    .answer-card {
        background: #FFFFFF;
        border: 1px solid #E4DCC8;
        border-right: 5px solid #4C6B4F;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
        box-shadow: 0 2px 12px rgba(43,58,47,0.06);
    }
    .answer-card h3 {
        font-family: 'Amiri', serif;
        color: #2B3A2F;
        margin-top: 0;
        font-size: 1.35rem;
    }
    .answer-card p, .answer-card div {
        font-family: 'Tajawal', sans-serif;
        color: #33402F;
        line-height: 2;
        font-size: 1.02rem;
    }

    .disclaimer-box {
        background: #EFE8D8;
        border-right: 4px solid #8B6F3E;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-top: 0.9rem;
        color: #6B5B3A;
        font-size: 0.9rem;
    }

    /* ===== بطاقات المصادر ===== */
    .sources-title {
        font-family: 'Amiri', serif;
        font-size: 1.25rem;
        color: #2B3A2F;
        margin: 1.6rem 0 0.7rem 0;
    }
    .source-card {
        background: #FFFFFF;
        border: 1px solid #E4DCC8;
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.8rem;
    }
    .source-name {
        font-family: 'Tajawal', sans-serif;
        font-weight: 700;
        color: #2B3A2F;
        font-size: 1rem;
    }
    .source-meta {
        color: #8A8071;
        font-size: 0.82rem;
    }
    .badge {
        display: inline-block;
        padding: 0.22rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        white-space: nowrap;
    }
    .badge-high {
        background: #F7E4E0;
        color: #A13D2E;
    }
    .badge-caution {
        background: #F6ECD3;
        color: #92701F;
    }
    .badge-normal {
        background: #E4EDE3;
        color: #4C6B4F;
    }

    footer, #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# تحميل الموارد الثقيلة مرة واحدة فقط (تُخزَّن في الكاش)
# ============================================================

@st.cache_resource(show_spinner="جاري تحميل موديل الفهم الدلالي...")
def load_embedder():
    return Embedder()


@st.cache_resource(show_spinner="جاري تجهيز فهرس البحث...")
def load_retriever(_embedder):
    df = build_herb_dataframe(DATA_PATH)
    return HerbRetriever(df, _embedder)


embedder = load_embedder()
retriever = load_retriever(embedder)


# ============================================================
# الترويسة
# ============================================================

st.markdown(
    """
    <div class="herbal-hero">
        <h1>🌿 موسوعة الأعشاب الطبية</h1>
        <p>مائة عشبة موثّقة — إجابات مبنية على المصدر فقط، بلا تخمين</p>
    </div>
    <div class="herbal-divider">﹀ ﹀ ﹀</div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# الإعدادات (تُقرأ تلقائيًا من Secrets، بلا واجهة ظاهرة)
# ============================================================

api_key = st.secrets.get("GROQ_API_KEY", "")
top_k = 5
alpha = 0.45

query = st.text_input(
    "❓ اكتب سؤالك",
    placeholder="مثال: ما فوائد البابونج؟ / علاج الأرق بالأعشاب / هل الحرمل آمن؟",
    label_visibility="collapsed",
)

ask = st.button("🔍  اسأل", type="primary", use_container_width=True)

RISK_LABELS = {
    "HIGH_RISK": ("⚠️ خطورة عالية", "badge-high"),
    "CAUTION": ("تحذير", "badge-caution"),
    "NORMAL": ("عادي", "badge-normal"),
}

if ask:
    if not query.strip():
        st.warning("من فضلك اكتب سؤال أولاً.")
    elif not api_key:
        st.error("مفيش GROQ_API_KEY متضاف في Secrets. أضيفه من إعدادات التطبيق على Streamlit (Manage app → Settings → Secrets).")
    else:
        with st.spinner("جاري البحث في الموسوعة..."):
            package = build_context_package(retriever, query, k=top_k, alpha=alpha)

        with st.spinner("جاري توليد الإجابة..."):
            result = generate_answer(query, package["context_text"], api_key=api_key)

        st.markdown(
            f"""
            <div class="answer-card">
                <h3>📋 الإجابة</h3>
                <div>{result['answer']}</div>
            </div>
            <div class="disclaimer-box">{result['disclaimer']}</div>
            """,
            unsafe_allow_html=True,
        )

        if package["sources"]:
            st.markdown('<div class="sources-title">📚 المصادر المسترجعة</div>', unsafe_allow_html=True)
            for src in package["sources"]:
                label, css_class = RISK_LABELS.get(src["risk"], RISK_LABELS["NORMAL"])
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div>
                            <div class="source-name">{src['herb_name']}</div>
                            <div class="source-meta">درجة التطابق: {src['score']}</div>
                        </div>
                        <span class="badge {css_class}">{label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("لم يتم العثور على عشبة مرتبطة بالسؤال في الموسوعة.")

st.markdown(
    """
    <div class="herbal-divider" style="margin-top:2.2rem;">﹀ ﹀ ﹀</div>
    """,
    unsafe_allow_html=True,
)
st.caption("مبني على notebook: Arabic Herbal RAG — محوّل ليشتغل بدون Ollama، عبر Streamlit + Groq.")
