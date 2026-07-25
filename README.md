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

## الخطوة 1 — جرّب المشروع على جهازك (اختياري بس مفيد)

```bash
# 1. ادخل مجلد المشروع
cd herbal-rag

# 2. اعمل بيئة افتراضية (مستحسن)
python -m venv venv
source venv/bin/activate        # على ويندوز: venv\Scripts\activate

# 3. ثبّت المكتبات
pip install -r requirements.txt

# 4. اعمل نسخة من ملف الأسرار
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# افتح الملف وحط فيه مفتاح Groq بتاعك (خطوة 2 تحت توضحلك تجيبه إزاي)

# 5. شغّل التطبيق
streamlit run app.py
```

هيفتحلك المتصفح على `http://localhost:8501`.

---

## الخطوة 2 — جيب مفتاح Groq API (مجاني)

1. ادخل على **https://console.groq.com**
2. سجّل دخول (بإيميلك أو بحساب Google)
3. من القائمة الجانبية اضغط **API Keys**
4. اضغط **Create API Key**، اديله اسم زي `herbal-rag`، وانسخ المفتاح
   (هيبان لك مرة واحدة بس، احفظه في مكان آمن)
5. حط المفتاح في `.streamlit/secrets.toml` كالتالي:

```toml
GROQ_API_KEY = "gsk_...المفتاح اللي نسخته..."
```

> بديل: تقدر كمان تكتب المفتاح مباشرة في شريط "Groq API Key" اللي في التطبيق نفسه من غير ما
> تحفظه في ملف، مفيد لو بتجرب بسرعة.

---

## الخطوة 3 — ارفع المشروع على GitHub

### أ) لو أول مرة تستخدم GitHub من التيرمينال

```bash
git config --global user.name "اسمك"
git config --global user.email "بريدك الإلكتروني"
```

### ب) اعمل الريبو

1. روح على **https://github.com/new**
2. اكتب اسم الريبو، مثلاً: `arabic-herbal-rag`
3. سيبه **Public** (لازم يكون Public عشان Streamlit Cloud المجاني يقدر يوصله)
4. **متعملش** تحديد "Add README" ولا "Add .gitignore" (هما موجودين عندنا خلاص)
5. اضغط **Create repository**

### ج) ارفع الكود من جهازك

جوه مجلد المشروع (`herbal-rag`)، نفّذ بالترتيب:

```bash
git init
git add .
git commit -m "Initial commit: Arabic Herbal RAG app"
git branch -M main
git remote add origin https://github.com/USERNAME/arabic-herbal-rag.git
git push -u origin main
```

⚠️ استبدل `USERNAME` باسم حسابك على GitHub، و `arabic-herbal-rag` باسم الريبو اللي عملته.

> تأكد إن `.streamlit/secrets.toml` (اللي فيه مفتاحك الحقيقي) **متترفعش** على GitHub —
> ملف `.gitignore` اللي في المشروع بيمنع ده تلقائيًا. اللي هيترفع بس هو `secrets.toml.example`
> (من غير مفتاح حقيقي جواه).

---

## الخطوة 4 — Deploy على Streamlit Cloud (اللينك الدائم)

1. روح على **https://share.streamlit.io**
2. سجّل دخول بحساب GitHub بتاعك (هيطلب صلاحية وصول لريبوهاتك)
3. اضغط **Create app** → **Deploy a public app from GitHub**
4. اختار:
   - **Repository**: `USERNAME/arabic-herbal-rag`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. قبل ما تضغط Deploy، افتح **Advanced settings** وحط الـ Secret بتاعك هناك:
   ```toml
   GROQ_API_KEY = "gsk_...المفتاح بتاعك..."
   ```
   (ده بديل آمن لملف secrets.toml — بياخده Streamlit Cloud مباشرة ومش بيتخزن في الكود)
6. اضغط **Deploy**

هياخد كام دقيقة أول مرة (بيثبّت المكتبات وبيحمّل موديل الـ embeddings)، وبعدها هتاخد لينك دائم شكله زي:

```
https://arabic-herbal-rag-USERNAME.streamlit.app
```

اللينك ده ثابت وشغال طول ما الريبو موجود — أي تحديث تعمله وترفعه على GitHub (`git push`)
هيتحدث تلقائيًا على نفس اللينك من غير ما تعمل Deploy تاني.

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
