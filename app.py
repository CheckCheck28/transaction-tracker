"""
Transaction Tracker - backend.

Flow for each click on "Analyze":
  1. The browser uploads a PDF to /analyze.
  2. We try to read selectable text from the PDF (downloaded bank statements).
     If there is almost no text, the PDF is a scan, so we turn the pages into one image.
  3. We make ONE Gemma call (ask or ask_with_image). Gemma answers in a compact
     line-by-line format (short answers are much faster than long JSON).
  4. Python (not the AI) adds everything up and sends the stats back to the page.
"""

import io
import os
import re
import tempfile
from collections import defaultdict

from dotenv import load_dotenv

# Load .env BEFORE importing the helper, in case the helper reads the key when it is imported.
load_dotenv()

import pymupdf  # PyMuPDF: reads PDF text and turns PDF pages into pictures
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image  # used to stick several scanned pages into one image

from gemma_client import ask, ask_with_image  # your helper

# --- Safety check: the key must come from the environment. We never print it. ---
if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit("GEMINI_API_KEY is missing. Copy .env.example to .env and add your key.")

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # refuse uploads bigger than 10 MB

MIN_TEXT_CHARS = 50      # fewer characters than this = we treat the PDF as a scan
MAX_TEXT_CHARS = 30000   # cap how much text we send to Gemma
MAX_SCAN_PAGES = 3       # cap how many scanned pages we send as an image

CATEGORIES = [
    "Food & Dining", "Groceries", "Shopping", "Transport", "Bills & Utilities",
    "Rent & Housing", "Entertainment", "Health", "Travel", "Transfers",
    "Fees & Charges", "Income", "Other",
]

# Gemma writes the category as a NUMBER (1, 2, 3...) instead of a long name.
# Fewer words to write = a faster answer.
CATEGORY_LIST = ", ".join(f"{i} {name}" for i, name in enumerate(CATEGORIES, start=1))

PROMPT = f"""Read this bank or card statement and list every transaction in a compact text format.

Output ONLY these lines, nothing else (no markdown, no explanation):
CURRENCY|<3-letter ISO code such as INR, USD, EUR>
SUMMARY|<2 short sentences about the spending habits you can see>
Then ONE line per transaction, exactly like this:
date|description|amount|D or C|category number

Rules:
- date: YYYY-MM-DD if you can tell the full date, otherwise exactly as written.
- description: the merchant or purpose, at most 4 words.
- amount: a positive number with no currency symbol or commas.
- D = money going out (debit). C = money coming in (credit).
- category number from this list: {CATEGORY_LIST}
- Skip opening/closing balances and running totals. Only real transactions.
"""


# ---------------------------------------------------------------- reading the PDF

def pdf_to_image_file(doc):
    """Render the first few pages as pictures and stack them into ONE jpg file."""
    pages = []
    for i in range(min(len(doc), MAX_SCAN_PAGES)):
        pixmap = doc[i].get_pixmap(dpi=110)
        pages.append(Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB"))

    width = max(p.width for p in pages)
    height = sum(p.height for p in pages)
    canvas = Image.new("RGB", (width, height), "white")

    y = 0
    for p in pages:
        canvas.paste(p, (0, y))
        y += p.height

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    canvas.save(tmp.name, "JPEG", quality=80)  # JPEG is much smaller than PNG = faster upload
    return tmp.name


def tidy_text(text):
    """Remove extra spaces and blank lines. Fewer characters = fewer tokens = faster."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def call_gemma(pdf_bytes):
    """Make the single Gemma call for this click and return its raw text answer."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    text = tidy_text("".join(page.get_text() for page in doc))
    print(f"PDF has {len(doc)} page(s), {len(text)} characters of text.")

    if len(text) >= MIN_TEXT_CHARS:
        # Downloaded PDF with real text -> plain text call.
        print("Text PDF: sending text to Gemma...")
        return ask(PROMPT + "\n\nSTATEMENT TEXT:\n" + text[:MAX_TEXT_CHARS])

    # Scanned PDF -> send a picture instead.
    print("Scanned PDF: sending image to Gemma...")
    image_path = pdf_to_image_file(doc)
    try:
        return ask_with_image(PROMPT, image_path)
    finally:
        os.remove(image_path)  # clean up the temporary picture


# ---------------------------------------------------------------- reading Gemma's answer

def parse_answer(raw):
    """Turn Gemma's pipe-separated lines into currency, summary and a list of transactions."""
    currency, summary, transactions = "", "", []

    for line in raw.splitlines():
        line = line.strip().strip("`").strip()
        if "|" not in line:
            continue

        upper = line.upper()
        if upper.startswith("CURRENCY|"):
            currency = line.split("|", 1)[1].strip()
            continue
        if upper.startswith("SUMMARY|"):
            summary = line.split("|", 1)[1].strip()
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue  # not a transaction line

        # Read from the right, so a "|" inside the description can't break the line.
        amount_text, kind_text, category_text = parts[-3:]
        description = "|".join(parts[1:-3])

        try:
            amount = abs(float(re.sub(r"[^\d.\-]", "", amount_text)))
        except ValueError:
            continue  # skip rows we can't read an amount from

        # Category: a number from our list (or, if Gemma wrote the name, accept the name).
        category = "Other"
        digits = re.sub(r"\D", "", category_text)
        if category_text in CATEGORIES:
            category = category_text
        elif digits and 1 <= int(digits) <= len(CATEGORIES):
            category = CATEGORIES[int(digits) - 1]

        transactions.append({
            "date": parts[0],
            "description": description[:80],
            "amount": round(amount, 2),
            "type": "credit" if kind_text.upper().startswith("C") else "debit",
            "category": category,
        })

    return currency, summary, transactions


# ---------------------------------------------------------------- the maths (plain Python)

def build_stats(transactions):
    debits = [t for t in transactions if t["type"] == "debit"]
    credits = [t for t in transactions if t["type"] == "credit"]

    total_spent = round(sum(t["amount"] for t in debits), 2)
    total_income = round(sum(t["amount"] for t in credits), 2)

    # Add up spending per category.
    per_category = defaultdict(float)
    for t in debits:
        per_category[t["category"]] += t["amount"]

    categories = [
        {
            "name": name,
            "total": round(total, 2),
            "percent": round(total / total_spent * 100, 1) if total_spent else 0,
        }
        for name, total in sorted(per_category.items(), key=lambda kv: kv[1], reverse=True)
    ]

    biggest = sorted(debits, key=lambda t: t["amount"], reverse=True)[:5]

    return {
        "total_spent": total_spent,
        "total_income": total_income,
        "net": round(total_income - total_spent, 2),
        "count": len(transactions),
        "average_spend": round(total_spent / len(debits), 2) if debits else 0,
        "categories": categories,
        "biggest": biggest,
    }


# ---------------------------------------------------------------- web routes

@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("pdf")
    if file is None or file.filename == "":
        return jsonify(error="Choose a PDF file first."), 400

    pdf_bytes = file.read()
    if not pdf_bytes.startswith(b"%PDF"):
        return jsonify(error="That file is not a PDF."), 400

    try:
        raw = call_gemma(pdf_bytes)
        print("Gemma answered.")
        currency, summary, transactions = parse_answer(raw)
    except Exception:
        # We deliberately don't echo details back, so nothing sensitive leaks.
        return jsonify(error="Could not read that statement. Try a clearer PDF."), 500

    if not transactions:
        return jsonify(error="No transactions found in that PDF."), 422

    return jsonify(
        currency=currency[:3].upper(),
        summary=summary,
        stats=build_stats(transactions),
        transactions=transactions,
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)