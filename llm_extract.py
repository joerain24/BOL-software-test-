# llm_extract.py — OpenAI caller with clean fail-fast on insufficient_quota
import os, json, httpx, time, random

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
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()  # set to gpt-5 if you have it
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Add it as a repo secret and pass it in the workflow env.")

    # Keep the payload small to reduce cost and avoid quota usage
    text = ocr_text[:6000]  # trim long OCR blobs
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content": _prompt()},
            {"role":"user","content": text}
        ],
        "response_format": {"type":"json_schema","json_schema": SCHEMA}
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    max_attempts = 4
    base_sleep = 2.0

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, max_attempts + 1):
            resp = await client.post(url, headers=headers, json=payload)
            status = resp.status_code

            # Try to parse any error body for hints
            body_text = resp.text
            try:
                body = resp.json()
            except Exception:
                body = {}

            if status == 200:
                content = body["choices"][0]["message"]["content"]
                return _coerce_json(content)

            # Fail fast on quota problems
            if status in (429, 400) and isinstance(body, dict):
                err = (body.get("error") or {})
                if err.get("type") == "insufficient_quota" or "quota" in (err.get("message","").lower()):
                    raise RuntimeError("INSUFFICIENT_QUOTA: Your OpenAI plan/credits are exhausted.")

            # Gentle backoff for real throttling
            if status == 429 or 500 <= status < 600:
                if attempt == max_attempts:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.6)
                print(f"[extract_openai] HTTP {status} attempt {attempt}/{max_attempts}; backing off {sleep_s:.1f}s…")
                time.sleep(sleep_s)
                continue

            # Other client errors → show snippet and stop
            if 400 <= status < 500:
                raise httpx.HTTPStatusError(f"OpenAI HTTP {status}. Body: {body_text[:300]}", request=resp.request, response=resp)

            # Unknown server error
            resp.raise_for_status()


