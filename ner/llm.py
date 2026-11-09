import json
import os
from openai import OpenAI

def format_prompt_for_ner(text: str, labels: list[dict]) -> str:
    return f"""
    Extrae las siguientes entidades del texto: {labels}.
    Devuelve ÚNICAMENTE un objeto JSON válido con este formato exacto:
    {{
        "entities": [
    {{"start": 0, "end": 5, "label": "LABEL", "text": "texto"}}
        ]
    }}
    Texto:
    {text}
    """
def parse_llm_response(raw_response: str, original_text: str) -> list[dict]:
    try:
        data = json.loads(raw_response)
        raw_entities = data.get("entities", [])
    except Exception:
        return []

    parsed_entities = []
    for ent in raw_entities:
        label = ent.get("label")
        text = ent.get("text")
        if label and text:
            start = original_text.find(text)
            if start != -1:
                end = start + len(text)
                parsed_entities.append({
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": text
                })
    return parsed_entities


def predict_llm_entities(text: str, labels: list[str], client: OpenAI = None, model: str = "gpt-4o-mini") -> list[dict]:
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = format_prompt_for_ner(text, labels)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Eres un asistente experto en extracción de entidades nombradas (NER). "
                           "Debes responder estrictamente en formato JSON válido."
            },
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    raw_content = response.choices[0].message.content
    return parse_llm_response(raw_content, text)