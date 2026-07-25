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
    page_title="موسوعة الأعشاب الطبية — RAG",
    page_icon="🌿",
    layout="centered",
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
# الواجهة
# ============================================================

st.title("🌿 موسوعة الأعشاب الطبية")
st.caption("نظام RAG عربي — اسأل عن أي عشبة أو مرض وهيتم الرد بناءً على 100 عشبة موثّقة فقط.")

# الإعدادات بتتاخد من secrets.toml (محليًا) أو Secrets (على Streamlit Cloud) تلقائيًا،
# من غير ما تظهر للمستخدم في الواجهة.
api_key = st.secrets.get("GROQ_API_KEY", "")
top_k = 5
alpha = 0.45

query = st.text_input(
    "❓ اكتب سؤالك",
    placeholder="مثال: ما فوائد البابونج؟ / علاج الأرق بالأعشاب / هل الحرمل آمن؟",
)

ask = st.button("اسأل", type="primary", use_container_width=True)

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

        st.markdown("### 📋 الإجابة")
        st.write(result["answer"])
        st.info(result["disclaimer"])

        if package["sources"]:
            st.markdown("### 📚 المصادر المسترجعة")
            src_df = pd.DataFrame(package["sources"])
            src_df = src_df.rename(
                columns={
                    "herb_name": "العشبة",
                    "chunk_id": "الرقم",
                    "score": "درجة التطابق",
                    "risk": "التصنيف",
                }
            )
            st.dataframe(src_df, use_container_width=True, hide_index=True)
        else:
            st.warning("لم يتم العثور على عشبة مرتبطة بالسؤال في الموسوعة.")

st.divider()
st.caption("مبني على notebook: Arabic Herbal RAG — محوّل ليشتغل بدون Ollama، عبر Streamlit + Groq.")
