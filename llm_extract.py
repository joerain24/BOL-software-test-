# llm_extract.py — call OpenAI to extract BOL JSON from OCR text
import os, json, httpx

SCHEMA = {
  "type": "object",
  "properties": {
    "bol_number": {"type":"string"},
    "pro_number": {"type":"string"},
    "ship_date": {"type":"string"},
    "carrier": {"type":"object","properties":{"name":{"type":"string"},"scac":{"type":"string"}}},
    "freight_lines": {"type":"array","items":{"type":"object","properties":{
      "description":{"type":"string"},"quantity":{"type":"number"},"package_type":{"type":"string"},
      "weight":{"type":"number"},"weight_unit":{"type":"string"}}}},
    "total_weight":{"type":"number"},
    "total_packages":{"type":"number"}
  },
  "additionalProperties": True
}

def _prompt():
    return "Extract Bill of Lading fields from OCR text. Return ONLY valid JSON matching the schema. If unsure, use null."

def _coerce_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(s[a:b+1])
        raise

async def extract_openai(ocr_text: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # set your preferred model name
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
      "model": model,
      "messages": [
        {"role":"system","content": _prompt()},
        {"role":"user","content": ocr_text[:12000]}
      ],
      "response_format": {"type":"json_schema","json_schema": SCHEMA}
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return _coerce_json(content)
