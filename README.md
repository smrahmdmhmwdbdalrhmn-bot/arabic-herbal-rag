# 🌿 موسوعة الأعشاب الطبية — RAG عربي (Streamlit + Groq)

نظام RAG (Retrieval-Augmented Generation) بيرد على أسئلة عن 100 عشبة طبية، بيدمج بحث لفظي (BM25)
وبحث دلالي (Embeddings) عشان يجيب أدق النتائج، وبعدين بيولّد إجابة مبنية **فقط** على المصدر
(مفيش اختلاق معلومات)، مع الاهتمام الخاص بتحذيرات الأعشاب الخطيرة.

> ملحوظة: النسخة دي محوّلة من notebook أصلي كان بيستخدم **Ollama** (سيرفر محلي). بما إن Ollama
> مش بيشتغل على Streamlit Cloud، استبدلنا:
> - الـ Embeddings: من `bge-m3` عبر Ollama → موديل `intfloat/multilingual-e5-small` عبر مكتبة
>   `sentence-transformers` (شغال جوه نفس السيرفر، من غير أي تثبيت خارجي).
> - التوليد: من `qwen2.5`/`silma` عبر Ollama → **Groq API** (مجاني وسريع جدًا).

---

## هيكل المشروع

```
herbal-rag/
├── app.py                          # واجهة Streamlit الرئيسية
├── rag_core.py                     # تحميل البيانات، تنظيف عربي، BM25 + بحث هجين
├── embedder.py                     # الـ embeddings الدلالية (sentence-transformers)
├── generation.py                   # التوليد عبر Groq + الـ prompt الصارم
├── data/
│   └── herbs_encyclopedia.txt      # بيانات الأعشاب (100 عشبة)
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example        # نموذج لملف الأسرار (بدون مفتاح حقيقي)
├── .gitignore
└── README.md
```

## تحديث المشروع لاحقًا

أي تعديل تعمله (مثلاً تضيف أعشاب جديدة في `data/herbs_encyclopedia.txt` أو تعدّل الـ prompt):

```bash
git add .
git commit -m "وصف التعديل"
git push
```

Streamlit Cloud هيلاقظ التحديث ويعيد التشغيل تلقائيًا خلال ثواني.
