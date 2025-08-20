import os
import io
import re
import csv
import json
import glob
import time
import asyncio
from typing import List

from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from dateutil import parser

# LLM caller (OpenAI). Make sure llm_extract.py is in the repo root.
from llm_extract import extract_openai

OUT_DIR = "outputs"
JSON_DIR = os.path.join(OUT_DIR, "json")
CSV_HEADERS = os.path.join(OUT_DIR, "bol_headers.csv")
CSV_LINES = os.path.join(OUT_DIR, "bol_lines.csv")

os.makedirs(JSON_DIR, exist_ok=True)

# ---------- Simple regex patterns (fallback PoC) ----------
BOL_RE = re.compile(r"\b(?:BOL|B\.O\.L\.?|Bill of Lading)[:#\s-]*([A-Z0-9-]{6,})", re.I)
PRO_RE = re.compile(r"\b(?:PRO|Pro No\.?|Pro#)[:#\s-]*([A-Z0-9-]{5,})", re.I)
DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
SCAC_RE = re.compile(r"\b[A-Z]{2,4}\b")
WEIGHT_RE = re.compile(r"\b(\d{2,6})\s?(lb|lbs|pounds|kg|kgs)\b", re.I)


def to_iso_date(s: str | None) -> str | None:
    if not s:
        return None
    try:
        return parser.parse(s, dayfirst=False).date().isoformat()
    except Exception:
        return None


def ocr_any(path: str) -> str:
    """OCR a PDF or image path and return plain text."""
    with open(path, "rb") as fh:
        data = fh.read()
    # Try PDF first
    try:
        pages = convert_from_bytes(data, dpi=300)
        texts = [pytesseract.image_to_string(p) for p in pages]
        return "\n\n".join(texts)
    except Exception:
        # Fallback: image
        im = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(im)


def extract_regex(text: str) -> dict:
    """Very basic regex fallback for PoC."""
    out = {
        "bol_number": None,
        "pro_number": None,
        "ship_date": None,
        "carrier": {"name": None, "scac": None},
        "freight_lines": [],
        "total_weight": None,
        "total_packages": None,
    }
    if m := BOL_RE.search(text):
        out["bol_number"] = m.group(1).strip()
    if m := PRO_RE.search(text):
        out["pro_number"] = m.group(1).strip()
    if m := DATE_RE.search(text):
        out["ship_date"] = to_iso_date(m.group(1))

    # crude SCAC guess
    scacs = SCAC_RE.findall(text)
    if scacs:
        out["carrier"]["scac"] = scacs[0]

    # crude weights (pick largest as total)
    weights: List[float] = []
    for w, unit in WEIGHT_RE.findall(text):
        v = float(w)
        unit = unit.lower()
        if unit.startswith("kg"):
            v *= 2.20462  # kg → lb
        weights.append(v)
    if weights:
        out["total_weight"] = max(weights)
        out["freight_lines"].append({
            "description": "Freight",
            "quantity": 1,
            "package_type": "PKG",
            "weight": out["total_weight"],
            "weight_unit": "lb"
        })
    return out


def write_csvs(id_: str, data: dict) -> None:
    # headers CSV
    header_exists = os.path.exists(CSV_HEADERS)
    with open(CSV_HEADERS, "a", newline="") as f:
        w = csv.writer(f)
        if not header_exists:
            w.writerow(["id", "bol_number", "pro_number", "ship_date", "carrier_scac", "total_weight", "total_packages"])
        w.writerow([
            id_,
            data.get("bol_number") or "",
            data.get("pro_number") or "",
            data.get("ship_date") or "",
            (data.get("carrier") or {}).get("scac") or "",
            data.get("total_weight") or "",
            data.get("total_packages") or "",
        ])

    # lines CSV
    lines_exists = os.path.exists(CSV_LINES)
    with open(CSV_LINES, "a", newline="") as f:
        w = csv.writer(f)
        if not lines_exists:
            w.writerow(["id", "description", "quantity", "package_type", "weight", "weight_unit"])
        for fl in data.get("freight_lines", []):
            w.writerow([
                id_,
                fl.get("description", ""),
                fl.get("quantity", ""),
                fl.get("package_type", ""),
                fl.get("weight", ""),
                fl.get("weight_unit", ""),
            ])


def process_one(path: str, idx: int) -> None:
    print("Processing", path)
    text = ocr_any(path)

    MODE = os.getenv("EXTRACTOR_MODE", "REGEX").upper()
    if MODE == "OPENAI":
        try:
            # OpenAI JSON extraction
            data = asyncio.get_event_loop().run_until_complete(extract_openai(text))
        except RuntimeError as e:
            # Clean fallback when quota is exhausted (or other explicit fail-fast)
            if "INSUFFICIENT_QUOTA" in str(e):
                print("[process_one] OpenAI quota exhausted — falling back to regex for this file.")
                data = extract_regex(text)
            else:
                raise
    else:
        data = extract_regex(text)

    base = os.path.splitext(os.path.basename(path))[0]
    id_ = f"{base}-{idx:04d}"

    os.makedirs(JSON_DIR, exist_ok=True)
    with open(os.path.join(JSON_DIR, f"{id_}.json"), "w") as f:
        json.dump(data, f, indent=2)

    write_csvs(id_, data)
    print("Done →", id_)


def main() -> None:
    files = sorted(glob.glob("samples/*.pdf")) \
          + sorted(glob.glob("samples/*.png")) \
          + sorted(glob.glob("samples/*.jpg"))

    if not files:
        print("No sample files in samples/")
        return

    # TEMP while testing: only process first file to avoid rate limits
    files = files[:1]

    os.makedirs(OUT_DIR, exist_ok=True)

    for idx, path in enumerate(files, 1):
        process_one(path, idx)
        time.sleep(5)  # pause to avoid hitting API rate limits


if __name__ == "__main__":
    main()


