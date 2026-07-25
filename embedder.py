# -*- coding: utf-8 -*-
"""
embedder.py
بديل خطوة الـ Embeddings اللي كانت بتشتغل عن طريق Ollama (bge-m3).
هنا بنستخدم موديل صغير من sentence-transformers بيشتغل جوه نفس سيرفر Streamlit،
من غير أي سيرفر خارجي أو تثبيت إضافي على الجهاز.

الموديل: intfloat/multilingual-e5-small
- خفيف نسبيًا (~470 ميجا)، بيغطي العربي كويس، وبيتحمّل مرة واحدة ويتخزن في الكاش.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        # موديلات e5 بتحتاج بادئة query:/passage: لأفضل أداء، لكن هنا هنبسطها
        # ونعامل كل النصوص بنفس الطريقة (passage) لتفادي التعقيد، وده كافٍ لحجم
        # الموسوعة الصغير ده (100 عشبة).
        prefixed = [f"query: {t}" for t in texts]
        vectors = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")
