# -*- coding: utf-8 -*-
"""
rag_core.py
منطق الـ RAG الأساسي: تحميل البيانات، تنظيف عربي، بحث لفظي (BM25) + بحث دلالي (Embeddings)،
دمج هجين، وبناء "حزمة سياق" (context package) جاهزة للـ LLM.

مبني على نفس منطق الـ notebook الأصلي، لكن مبسّط ومحوّل لملف .py عادي
يشتغل من غير Ollama (بديل الـ Embeddings هنا: sentence-transformers، وبديل التوليد: Groq API في generation.py).
"""

import re
import numpy as np
import pandas as pd

import pyarabic.araby as araby
import arabicstopwords.arabicstopwords as arabicstopwords
from tashaphyne.stemming import ArabicLightStemmer

from rank_bm25 import BM25Okapi

# ============================================================
# 1) تحميل و تقسيم الملف إلى إدخالات (كل عشبة = سجل)
# ============================================================

ENTRY_PATTERN = re.compile(r"📌\s*(\d+)\.\s*(.+)")


def load_herb_entries(path: str) -> list[dict]:
    """يقرأ ملف herbs_encyclopedia.txt ويرجّع قائمة إدخالات خام (رقم + نص كامل)."""
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    matches = list(ENTRY_PATTERN.finditer(raw_text))
    entries = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        entry_num = int(m.group(1))
        entry_text = raw_text[start:end].strip()
        entries.append({"entry_id": entry_num, "text": entry_text})
    return entries


# ============================================================
# 2) استخراج الحقول (metadata) من كل إدخال
# ============================================================

def extract_metadata(text: str) -> dict:
    metadata = {
        "herb_name": "",
        "herb_name_en": "",
        "benefits": "",
        "diseases": "",
        "usage": "",
        "warnings": "",
        "price": "",
    }

    header = re.match(r"📌\s*\d+\.\s*(.+)", text)
    if header:
        title = header.group(1).strip()
        en_match = re.search(r"\((.*?)\)", title)
        if en_match:
            metadata["herb_name_en"] = en_match.group(1).strip()
            metadata["herb_name"] = title[: en_match.start()].strip(" /")
        else:
            metadata["herb_name"] = title.strip()

    def grab(field_label, stop_labels):
        stop_pattern = "|".join(re.escape(s) for s in stop_labels)
        pattern = rf"{field_label}:\s*(.*?)(?:{stop_pattern}|$)"
        m = re.search(pattern, text, re.S)
        if not m:
            return ""
        value = m.group(1)
        # شيل أي سطر جديد + علامة التعداد "•" متسربة من السطر التالي، ومسافات زيادة
        value = re.sub(r"\s*[\n•]+\s*$", "", value)
        return value.strip(" \n\t.،")

    labels = [
        "الفوائد الرئيسية",
        "الأمراض التي تعالجها",
        "طريقة الاستخدام",
        "التحذيرات والموانع",
        "السعر الاسترشادي",
    ]

    metadata["benefits"] = grab(labels[0], labels[1:])
    metadata["diseases"] = grab(labels[1], labels[2:])
    metadata["usage"] = grab(labels[2], labels[3:])
    metadata["warnings"] = grab(labels[3], labels[4:])
    m = re.search(r"السعر الاسترشادي.*?:\s*(.*)", text)
    metadata["price"] = m.group(1).strip() if m else ""

    return metadata


# ============================================================
# 3) تنظيف وتطبيع النص العربي
# ============================================================

ARABIC_LETTER_STEMMER = ArabicLightStemmer()


def normalize_arabic(text: str) -> str:
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[ؤئ]", "ء", text)
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)  # إبقاء الحروف العربية فقط
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_lexical_form(text: str) -> str:
    """إزالة كلمات الوقف + تجذيع خفيف — يُستخدم في نسخة BM25 فقط."""
    words = text.split()
    out = []
    for w in words:
        if arabicstopwords.is_stop(w):
            continue
        try:
            ARABIC_LETTER_STEMMER.light_stem(w)
            stemmed = ARABIC_LETTER_STEMMER.get_stem()
            out.append(stemmed if stemmed else w)
        except Exception:
            out.append(w)
    return " ".join(out)


# ============================================================
# 4) بناء جدول العشاب (DataFrame) بكل الحقول + النصوص المعالجة
# ============================================================

def build_herb_dataframe(path: str) -> pd.DataFrame:
    entries = load_herb_entries(path)
    rows = []
    for e in entries:
        meta = extract_metadata(e["text"])
        clean = clean_text(normalize_arabic(e["text"]))
        lexical = to_lexical_form(clean)

        retrieval_text = " ".join(
            [
                meta["herb_name"],
                meta["benefits"],
                meta["diseases"],
                meta["warnings"],
                meta["usage"],
            ]
        )
        embedding_text = (
            f"اسم العشبة: {meta['herb_name']}\n"
            f"الفوائد: {meta['benefits']}\n"
            f"الأمراض: {meta['diseases']}\n"
            f"طريقة الاستخدام: {meta['usage']}\n"
            f"التحذيرات: {meta['warnings']}"
        )

        rows.append(
            {
                "chunk_id": e["entry_id"],
                "raw_text": e["text"],
                "clean": clean,
                "lexical": lexical,
                **meta,
                "retrieval_text": clean_text(normalize_arabic(retrieval_text)),
                "embedding_text": embedding_text,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 5) تصنيف درجة الخطورة (علشان الأولوية في السياق)
# ============================================================

HIGH_RISK_MARKERS = ["سام", "قاتل", "الوفاة", "تسمم", "شلل", "فشل تنفسي"]
CAUTION_MARKERS = [
    "يحظر", "يمنع", "الحوامل", "المرضعات", "الأطفال",
    "مرضى السكر", "مرضى الضغط", "الوارفارين", "مضادات التخثر",
    "يتفاعل", "يخفض السكر", "يخفض الضغط",
]


def classify_risk(row) -> str:
    warnings = str(row.get("warnings", ""))
    if any(marker in warnings for marker in HIGH_RISK_MARKERS):
        return "HIGH_RISK"
    if any(marker in warnings for marker in CAUTION_MARKERS):
        return "CAUTION"
    return "NORMAL"


# ============================================================
# 6) الفهرسة: BM25 + Embeddings (دلاليّ)
# ============================================================

class HerbRetriever:
    def __init__(self, df: pd.DataFrame, embedder):
        """
        df: جدول الأعشاب من build_herb_dataframe
        embedder: كائن عنده .encode(list[str]) -> np.ndarray (مُطبَّع L2)
        """
        self.df = df.reset_index(drop=True)
        self.embedder = embedder

        # BM25
        tokenized = [t.split() for t in self.df["lexical"].tolist()]
        self.bm25 = BM25Okapi(tokenized)

        # Embeddings دلالية (تُحسب مرة واحدة عند بدء التطبيق)
        self.doc_embeddings = self.embedder.encode(
            self.df["embedding_text"].tolist()
        )

        self.df["risk"] = self.df.apply(classify_risk, axis=1)

    def _bm25_scores(self, query: str) -> np.ndarray:
        q_clean = clean_text(normalize_arabic(query))
        q_lexical = to_lexical_form(q_clean).split()
        scores = np.array(self.bm25.get_scores(q_lexical), dtype="float32")
        return scores

    def _semantic_scores(self, query: str) -> np.ndarray:
        q_vec = self.embedder.encode([query])[0]
        sims = self.doc_embeddings @ q_vec
        return sims.astype("float32")

    @staticmethod
    def _min_max(scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.45) -> pd.DataFrame:
        """
        alpha: وزن البحث الدلالي (0 = لفظي بالكامل BM25 فقط، 1 = دلالي بالكامل).
        alpha أقل من 0.5 معناه اعتماد أكبر على المطابقة اللفظية الدقيقة للاسم/المرض.
        """
        bm25_raw = self._bm25_scores(query)
        sem_raw = self._semantic_scores(query)

        bm25_norm = self._min_max(bm25_raw)
        sem_norm = self._min_max(sem_raw)

        hybrid = alpha * sem_norm + (1 - alpha) * bm25_norm

        out = self.df.copy()
        out["bm25_score"] = bm25_raw
        out["semantic_score"] = sem_raw
        out["hybrid_score"] = hybrid

        out = out.sort_values("hybrid_score", ascending=False).head(k)
        return out.reset_index(drop=True)


# ============================================================
# 7) بناء "حزمة السياق" اللي هتتبعت للـ LLM
# ============================================================

CONTEXT_SEP = "\n" + ("=" * 40) + "\n"


def build_context_package(
    retriever: HerbRetriever,
    query: str,
    k: int = 5,
    alpha: float = 0.45,
    min_score_ratio: float = 0.15,
) -> dict:
    results = retriever.hybrid_search(query, k=k, alpha=alpha)

    if len(results) == 0 or results["hybrid_score"].max() <= 0:
        return {"context_text": "", "sources": [], "num_sources": 0}

    max_score = results["hybrid_score"].max()

    # إعطاء أولوية لأي نتيجة HIGH_RISK حتى لو نقاطها أقل شوية
    def keep(row):
        if row["risk"] == "HIGH_RISK":
            return True
        return row["hybrid_score"] >= max_score * min_score_ratio

    kept = results[results.apply(keep, axis=1)]

    blocks = []
    sources = []
    for _, row in kept.iterrows():
        tag = f"[{row['risk']}]" if row["risk"] != "NORMAL" else ""
        block = (
            f"العشبة: {row['herb_name']} ({row.get('herb_name_en','')}) {tag}\n"
            f"الفوائد الرئيسية: {row['benefits']}\n"
            f"الأمراض التي تعالجها: {row['diseases']}\n"
            f"طريقة الاستخدام: {row['usage']}\n"
            f"التحذيرات والموانع: {row['warnings']}\n"
            f"السعر الاسترشادي: {row['price']}"
        )
        blocks.append(block)
        sources.append(
            {
                "herb_name": row["herb_name"],
                "chunk_id": int(row["chunk_id"]),
                "score": round(float(row["hybrid_score"]), 3),
                "risk": row["risk"],
            }
        )

    return {
        "context_text": CONTEXT_SEP.join(blocks),
        "sources": sources,
        "num_sources": len(sources),
    }
