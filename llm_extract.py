# llm_extract.py — OpenAI caller, returns Waybill fields, fail-fast on insufficient_quota
import os, json, httpx, time, random

SCHEMA = {
  "type": "object",
  "properties": {
    "waybill_number": {"type":"string"},
    "date": {"type":"string"},             # prefer YYYY-MM-DD
    "shipper": {"type":"string"},
    "carrier": {"type":"string"},
    "po_number": {"type":"string"},
    "material": {"type":"string"},
    "gross_weight": {"type":"number"},     # lb
    "tare_weight": {"type":"number"},      # lb
    "net_weight": {"type":"number"},       # lb
    "location": {"type":"string"},
    "ticket_number": {"type":"string"},
    "vehicle_number": {"type":"string"},
    "signature_present": {"type":"string"},   # "Yes" / "No" / "Not Indicated"
    "radiation_checked": {"type":"string"}    # "Yes" / "No" / "Not Indicated"
  },
  "additionalProperties": True
}

def _prompt():
  return (
    "Extract these fields from the provided OCR text of a waybill/scale ticket. "
    "Return ONLY JSON with exactly these keys. If a value is missing/unclear, use null. "
    "Keys: waybill_number, date (YYYY-MM-DD if possible), shipper, carrier, po_number, material, "
    "gross_weight (lb), tare_weight (lb), net_weight (lb), location, ticket_number, vehicle_number, "
    "signature_present ('Yes'/'No'/'Not Indicated'), radiation_checked ('Yes'/'No'/'Not Indicated')."
  )

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
  model = (os.getenv("OPENAI_MODEL") or "gpt-5").strip()  # change if your account needs a different id
  if not api_key:
    raise RuntimeError("OPENAI_API_KEY missing. Add it as a repo secret and pass it in the workflow env.")

  # keep payload modest
  text = ocr_text[:6000]
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

  max_attempts = int(os.getenv("OPENAI_MAX_ATTEMPTS", "4"))
  base_sleep   = float(os.getenv("OPENAI_BASE_SLEEP", "2.0"))

  async with httpx.AsyncClient(timeout=30) as client:
    for attempt in range(1, max_attempts + 1):
      resp = await client.post(url, headers=headers, json=payload)
      status = resp.status_code

      try:
        body = resp.json()
      except Exception:
        body = {}
      body_text = resp.text

      if status == 200:
        content = body["choices"][0]["message"]["content"]
        return _coerce_json(content)

      # fail fast if out of quota
      if status in (429, 400) and isinstance(body, dict):
        err = (body.get("error") or {})
        if err.get("type") == "insufficient_quota" or "quota" in (err.get("message","").lower()):
          raise RuntimeError("INSUFFICIENT_QUOTA: Your OpenAI plan/credits are exhausted.")

      # gentle backoff for throttling/server errors
      if status == 429 or 500 <= status < 600:
        if attempt == max_attempts:
          resp.raise_for_status()
        retry_after = resp.headers.get("Retry-After")
        sleep_s = float(retry_after) if retry_after else base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.6)
        print(f"[extract_openai] HTTP {status} attempt {attempt}/{max_attempts}; backing off {sleep_s:.1f}s…")
        time.sleep(sleep_s)
        continue

      # other 4xx → show snippet and stop
      if 400 <= status < 500:
        raise httpx.HTTPStatusError(f"OpenAI HTTP {status}. Body: {body_text[:300]}", request=resp.request, response=resp)

      resp.raise_for_status()


