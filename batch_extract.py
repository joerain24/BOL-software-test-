import os, io, re, csv, json, glob, time, asyncio
from typing import List
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from dateutil import parser

from llm_extract import extract_openai

OUT_DIR = "outputs"
JSON_DIR = os.path.join(OUT_DIR, "json")
WAYBILLS_CSV = os.path.join(OUT_DIR, "waybills.csv")

os.makedirs(JSON_DIR, exist_ok=True)

# minimal regex fallback (fills only what we can quickly guess)
DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")

def to_iso_date(s: str | None) -> str | None:
    if not s: return None
    try:
        return parser.parse(s, dayfirst=False).date().isoformat()
    except Exception:
        return None

def ocr_any(path: str) -> str:
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        pages = convert_from_bytes(data, dpi=300)
        texts = [pytesseract.image_to_string(p) for p in pages]
        return "\n\n".join(texts)
    except Exception:
        im = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(im)

def extract_regex(text: str) -> dict:
    out = {
        "waybill_number": None,
        "date": None,
        "shipper": None,
        "carrier": None,
        "po_number": None,
        "material": None,
        "gross_weight": None,
        "tare_weight": None,
        "net_weight": None,
        "location": None,
        "ticket_number": None,
        "vehicle_number": None,
        "signature_present": None,
        "radiation_checked": None,
    }
    if m := DATE_RE.search(text):
        out["date"] = to_iso_date(m.group(1))
    return out

def write_waybill_row(data: dict, source_file: str) -> None:
    header = [
        "Waybill #","Date","Shipper","Carrier","PO #","Material",
        "Gross Wt","Tare Wt","Net Wt","Location","Ticket #","Vehicle #",
        "Signature Present","Radiation Checked","Source File"
    ]
    file_exists = os.path.exists(WAYBILLS_CSV)
    with open(WAYBILLS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow([
            data.get("waybill_number") or "",
            data.get("date") or "",
            data.get("shipper") or "",
            data.get("carrier") or "",
            data.get("po_number") or "",
            data.get("material") or "",
            data.get("gross_weight") or "",
            data.get("tare_weight") or "",
            data.get("net_weight") or "",
            data.get("location") or "",
            data.get("ticket_number") or "",
            data.get("vehicle_number") or "",
            data.get("signature_present") or "",
            data.get("radiation_checked") or "",
            source_file,
        ])

def process_one(path: str, idx: int) -> None:
    print("Processing", path)
    text = ocr_any(path)
    mode = os.getenv("EXTRACTOR_MODE", "REGEX").upper()

    if mode == "OPENAI":
        try:
            data = asyncio.get_event_loop().run_until_complete(extract_openai(text))
        except RuntimeError as e:
            if "INSUFFICIENT_QUOTA" in str(e):
                print("[process_one] OpenAI quota exhausted — falling back to regex for this file.")
                data = extract_regex(text)
            else:
                raise
    else:
        data = extract_regex(text)

    # Save raw JSON (audit/debug)
    base = os.path.splitext(os.path.basename(path))[0]
    job_id = f"{base}-{idx:04d}"
    os.makedirs(JSON_DIR, exist_ok=True)
    with open(os.path.join(JSON_DIR, f"{job_id}.json"), "w") as f:
        json.dump(data, f, indent=2)

    # Append a row to waybills.csv
    write_waybill_row(data, source_file=os.path.basename(path))
    print("Done →", job_id)

def main() -> None:
    files = sorted(glob.glob("samples/*.pdf")) \
          + sorted(glob.glob("samples/*.png")) \
          + sorted(glob.glob("samples/*.jpg"))

    if not files:
        print("No sample files in samples/")
        return

    # While testing, process just the first file to avoid rate limits
    files = files[:1]

    os.makedirs(OUT_DIR, exist_ok=True)
    for idx, path in enumerate(files, 1):
        process_one(path, idx)
        time.sleep(5)

if __name__ == "__main__":
    main()


