# -*- coding: utf-8 -*-
"""
embedder.py

خطوة الـ Embeddings لتطبيق Arabic Herbal RAG
باستخدام Sentence Transformers بدلًا من Ollama.

الموديل:
intfloat/multilingual-e5-small
"""

from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "intfloat/multilingual-e5-small"


class Embedder:

    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):

        # موديلات E5 بتحتاج بادئة query:/passage:
        # نستخدم query هنا بشكل موحد.

        prefixed = [
            f"query: {text}"
            for text in texts
        ]

        vectors = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            vectors,
            dtype="float32"
        )
