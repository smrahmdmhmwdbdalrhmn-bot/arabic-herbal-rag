# -*- coding: utf-8 -*-
"""
generation.py
بديل خطوة التوليد اللي كانت بتشتغل عن طريق Ollama (qwen2.5 / silma) — هنا بنستخدم Groq API
(مجاني وسريع، وموديلاته بتفهم عربي كويس).
"""

from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"

DISCLAIMER = (
    "هذه المعلومات مستقاة من مصدر مرجعي في الأعشاب الطبية، "
    "ولا تُغني عن استشارة طبيب مختص."
)


def build_strict_prompt(query: str, context_text: str) -> str:
    if not context_text.strip():
        return None

    return f"""أنت مساعد متخصص في الأعشاب الطبية، تجاوب فقط بناءً على السياق المرفق أدناه ولا شيء غيره.

قواعد صارمة يجب الالتزام بها:
1. استخدم فقط المعلومات الموجودة في السياق. لا تخترع أي معلومة غير موجودة فيه.
2. إذا كان السياق لا يحتوي على إجابة كافية للسؤال، قل بوضوح: "المعلومات المتاحة لا تكفي للإجابة على هذا السؤال بدقة."
3. إذا وُجدت عشبة مصنّفة [HIGH_RISK] ضمن السياق، يجب ذكر تحذيرها بشكل صريح وواضح، ولا يجوز التقليل من خطورته أو حذفه أبدًا.
4. لا تقدّم أي نصيحة طبية قاطعة (مثل "استخدم" أو "لا تستخدم" بشكل حاسم) — قدّم المعلومة كما وردت في المصدر واذكر أنها ليست بديلاً عن استشارة طبيب.
5. لا تُصنّف الإجابة إلى "إيجابية" أو "سلبية" — فقط اعرض المعلومة كما هي.
6. اذكر في نهاية إجابتك أسماء الأعشاب التي استندت إليها من السياق (كمصادر).

السياق:
{context_text}

السؤال:
{query}

الإجابة (بالعربية، منظمة وواضحة):"""


def generate_answer(
    query: str,
    context_text: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    prompt = build_strict_prompt(query, context_text)

    if prompt is None:
        return {
            "answer": "المعلومات المتاحة لا تكفي للإجابة على هذا السؤال بدقة — لم يتم العثور على أعشاب مرتبطة بالسؤال في الموسوعة.",
            "disclaimer": DISCLAIMER,
        }

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=800,
    )

    answer = response.choices[0].message.content

    return {"answer": answer, "disclaimer": DISCLAIMER}
