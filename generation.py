# -*- coding: utf-8 -*-

from groq import Groq

DEFAULT_MODEL = "openai/gpt-oss-20b"

DISCLAIMER = (
    "هذه المعلومات مستقاة من مصدر مرجعي في الأعشاب الطبية، "
    "ولا تُغني عن استشارة طبيب مختص."
)


def generate_answer(query, context_text, api_key, model=DEFAULT_MODEL):

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": context_text + "\n\nالسؤال: " + query,
            }
        ],
        temperature=0,
        max_tokens=800,
    )

    answer = response.choices[0].message.content

    if not answer:
        answer = "المعلومات المتاحة لا تكفي للإجابة على هذا السؤال بدقة."

    return {
        "answer": answer.strip(),
        "disclaimer": DISCLAIMER,
    }
