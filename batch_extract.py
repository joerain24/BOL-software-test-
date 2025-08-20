import os, io, re, csv, json, glob, time, asyncio
from typing import List
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from dateutil import parser

from llm_extract import extract_openai

OUT_DIR = "outputs"
JSON_DIR = os.path.join(OUT_DIR, "json")
DEBUG_DIR = os.path.join(OUT_DIR, "debug")
WAYBILLS_CSV = os.path.join(OUT_DIR, "waybills.csv")

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")

def to_iso_date(s: str | None) -> str | None:
    if not s: return None
    try:
        return parser.parse(s, dayfirst=False).date().isoformat()
    except Exception:
        return None

def _ocr_image(im: Image.Image) -> str:
    im = im.convert("L")  # grayscale
    cfg = "--oem 1 --psm 6"
    return pytesseract.image_to_string(im, config=cfg)

def ocr_any(path: str) -> str:
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        pages = convert_from_bytes(data, dpi=300)
        texts = [_ocr_image(p) for p in pages]
        return "\n\n".join(texts)
    except Exception:
        im = Image.open(io.BytesIO(data))
        return _ocr_image(im)

def extract_regex(text: str) -> dict:
    out = {k: None for k in [
        "waybill_number","date","shipper","carrier","po_number","material",
        "gross_weight","tare_weight","net_weight","location","ticket_number",
        "vehicle_number","signature_present","radiation_checked"
    ]}
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

    # DEBUG: save OCR text so you can inspect if nulls happen
    with open(os.path.join(DEBUG_DIR, f"ocr_{os.path.basename(path)}.txt"), "w", encoding="utf-8") as df:
        df.write(text[:20000])

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

    # derive net if missing
    try:
        if (data.get("net_weight") in (None, "") and
            data.get("gross_weight") not in (None, "") and
            data.get("tare_weight") not in (None, "")):
            gw = float(data["gross_weight"]); tw = float(data["tare_weight"])
            data["net_weight"] = gw - tw
    except Exception:
        pass

    base = os.path.splitext(os.path.basename(path))[0]
    job_id = f"{base}-{idx:04d}"

    # save raw JSON
    with open(os.path.join(JSON_DIR, f"{job_id}.json"), "w") as f:
        json.dump(data, f, indent=2)

    # append one row to the single CSV
    write_waybill_row(data, source_file=os.path.basename(path))
    print("Done →", job_id)

def main() -> None:
    files = sorted(glob.glob("samples/*.pdf")) \
          + sorted(glob.glob("samples/*.png")) \
          + sorted(glob.glob("samples/*.jpg"))

    if not files:
        print("No sample files in samples/")
        return

    # while testing, just do first file to avoid rate limits
    files = files[:1]

    os.makedirs(OUT_DIR, exist_ok=True)
    for idx, path in enumerate(files, 1):
        process_one(path, idx)
        time.sleep(3)

if __name__ == "__main__":
    main()


