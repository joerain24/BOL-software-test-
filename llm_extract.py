# llm_extract.py — OpenAI caller with retries/backoff + clear auth checks
import os, json, httpx, time, random

SCHEMA = {
  "type": "object",
  "properties": {
    "bol_number": {"type":"string"},
    "pro_number": {"type":"string"},
    "ship_date": {"type":"string"},
    "carrier": {"type":"object","properties":{
      "name":{"type":"string"},
      "scac":{"type":"string"}
    }},
    "freight_lines": {"type":"array","items":{"type":"object","properties":{
      "description":{"type":"string"},
      "quantity":{"type":"number"},
      "package_type":{"type":"string"},
      "weight":{"type":"number"},
      "weight_unit":{"type":"string"}
    }}},
    "total_weight":{"type":"number"},
    "total_packages":{"type":"number"}
  },
  "additionalProperties": True
}

def _prompt():
    return (
        "Extract Bill of Lading fields from the provided OCR text. "
        "Return ONLY valid JSON matching the schema. If unsure, use null."
    )

def _coerce_json(s: str):
    # be forgiving if model adds extra text; try to slice the outermost JSON block
    try:
        return json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(s[a:b+1])
        raise

async def extract_openai(ocr_text: str) -> dict:
    # -------- Auth & config checks --------
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "gpt-4.1-mini").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing/empty. "
            "Add a repo secret named OPENAI_API_KEY and expose it in the workflow env."
        )

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _prompt()},
            {"role": "user", "content": ocr_text[:12000]}  # trim huge OCR blobs
        ],
        "response_format": {"type": "json_schema", "json_schema": SCHEMA}
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # -------- Retry settings (tuneable via env) --------
    max_attempts = int(os.getenv("OPENAI_MAX_ATTEMPTS", "7"))
    base_sleep = float(os.getenv("OPENAI_BASE_SLEEP", "2.0"))  # seconds

    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                status = resp.status_code

                # Handle rate limits & server errors with backoff
                if status == 429 or 500 <= status < 600:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        sleep_s = float(retry_after)
                    else:
                        sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.6)
                    print(f"[extract_openai] HTTP {status} (attempt {attempt}/{max_attempts}). "
                          f"Backing off {sleep_s:.1f}s…")
                    if attempt == max_attempts:
                        # no more retries — raise with brief body snippet
                        try:
                            print(f"[extract_openai] Response body: {resp.text[:300]}")
                        except Exception:
                            pass
                        resp.raise_for_status()
                    time.sleep(sleep_s)
                    continue

                # For other 4xx (e.g., 401/403/400), fail fast with a helpful hint
                if 400 <= status < 500:
                    snippet = resp.text[:400]
                    raise httpx.HTTPStatusError(
                        f"OpenAI HTTP {status}. Likely bad/missing auth or invalid request. "
                        f"Body: {snippet}",
                        request=resp.request, response=resp
                    )

                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return _coerce_json(content)

            except httpx.HTTPStatusError as e:
                # 401/403 most common cause: missing/invalid Authorization header
                if e.response is not None and e.response.status_code in (401, 403):
                    raise RuntimeError(
                        "OpenAI auth failed (401/403). "
                        "Verify your repo secret OPENAI_API_KEY and that the workflow passes it:\n"
                        "env:\n  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}\n"
                        "Also ensure there are no extra quotes/spaces in the secret."
                    ) from e
                # Other non-retriable 4xx errors
                raise
            except Exception as e:
                # Network hiccup; backoff and retry
                if attempt == max_attempts:
                    raise
                sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.6)
                print(f"[extract_openai] Exception {type(e).__name__} on attempt {attempt}; "
                      f"sleeping {sleep_s:.1f}s then retrying…")
                time.sleep(sleep_s)

