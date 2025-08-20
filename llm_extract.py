# llm_extract.py — OpenAI caller, returns Waybill fields; robust JSON; fail-fast on quota
import os, json, httpx, time, random

FIELDS = [
  "waybill_number","date","shipper","carrier","po_number","material",
  "gross_weight","tare_weight","net_weight","location","ticket_number",
  "vehicle_number","signature_present","radiation_checked"
]

def _prompt():
  return (
    "You are a structured data extractor for US waybills / scale tickets.\n"
    "Rules:\n"
    f"• Return ONLY a valid JSON object with these keys exactly: {', '.join(FIELDS)}.\n"
    "• If a value is unknown, use null (do NOT invent).\n"
    "• Dates: prefer YYYY-MM-DD if possible.\n"
    "• Weights are numeric in pounds. If only gross/tare present, compute net_weight = gross_weight - tare_weight.\n"
    "• Signature fields are 'Yes', 'No', or 'Not Indicated'.\n"
    "• Keep output strictly JSON; no extra text.\n"
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
  model = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()  # switch to gpt-5 later if you have access
  if not api_key:
    raise RuntimeError("OPENAI_API_KEY missing. Add it as a repo secret and pass it in the workflow env.")

  text = ocr_text[:9000]  # keep payload modest
  url = "https://api.openai.com/v1/chat/completions"
  payload = {
    "model": model,
    "temperature": 0.1,
    "messages": [
      {"role":"system","content": _prompt()},
      {"role":"user","content": text}
    ],
    # broader support than json_schema
    "response_format": {"type": "json_object"}
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

      # other 4xx
      if 400 <= status < 500:
        raise httpx.HTTPStatusError(f"OpenAI HTTP {status}. Body: {body_text[:300]}", request=resp.request, response=resp)

      resp.raise_for_status()
