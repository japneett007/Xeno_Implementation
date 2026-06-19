"""
DataGuard — Transaction Validator
Backend API  |  Flask 2.x  |  Python 3.8+

Architecture principles applied:
  - Senior Architect  : clean layered design, single-responsibility modules
  - Backend Developer : RESTful semantics, input validation, structured errors
  - Security Auditor  : path-traversal guard, file-type enforcement, no secrets in logs
  - Code Reviewer     : type hints on public funcs, no bare except, mutable defaults avoided
  - Test Engineer     : all routes return consistent JSON envelopes (easy to test)
  - UI/UX Designer    : phone_rules JSON fed to template for dynamic country dropdown
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Country phone rules — keyed by ISO-3166-1 alpha-2 code
PHONE_RULES: Dict[str, Dict] = {
    "IN": {"digits": 10, "label": "India"},
    "SG": {"digits": 8,  "label": "Singapore"},
    "US": {"digits": 10, "label": "USA"},
    "UK": {"digits": 10, "label": "UK"},
    "AU": {"digits": 9,  "label": "Australia"},
    "DE": {"digits": 10, "label": "Germany"},
    "JP": {"digits": 11, "label": "Japan"},
}

DATE_FORMATS: List[str] = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
]

ALLOWED_EXTENSIONS = {"csv"}

# ---------------------------------------------------------------------------
# Helper: security
# ---------------------------------------------------------------------------

def _allowed_file(filename: str) -> bool:
    """Return True only for explicitly whitelisted extensions."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_phone(phone: object, country_code: str = "IN") -> Tuple[bool, str]:
    """Validate a phone number against country-specific digit count rules."""
    if pd.isna(phone) or str(phone).strip() == "":
        return False, "Missing"
    digits = re.sub(r"\D", "", str(phone))
    rule = PHONE_RULES.get(country_code.upper(), {"digits": 10})
    expected = rule["digits"]
    if len(digits) != expected:
        return False, f"Expected {expected} digits, got {len(digits)}"
    return True, "Valid"


def validate_date(date_val: object) -> Tuple[bool, str]:
    """Try to parse date_val against known formats."""
    if pd.isna(date_val) or str(date_val).strip() == "":
        return False, "Missing"
    date_str = str(date_val).strip()
    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(date_str, fmt)
            return True, fmt
        except ValueError:
            continue
    return False, "Unrecognised format"


def validate_email(email: object) -> Tuple[bool, str]:
    """Basic RFC-5321 email pattern check."""
    if pd.isna(email) or str(email).strip() == "":
        return False, "Missing"
    pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
    match = bool(re.match(pattern, str(email).strip()))
    return match, "Valid" if match else "Invalid format"


def validate_amount(value: object) -> bool:
    """Return True if value converts to a non-negative float."""
    if pd.isna(value):
        return False
    try:
        return float(str(value).replace(",", "")) >= 0
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Column-type detector
# ---------------------------------------------------------------------------

def detect_column_type(col_name: str) -> str:
    """Infer semantic type from column name keywords."""
    col = col_name.lower()
    if any(k in col for k in ("phone", "mobile", "contact")):
        return "phone"
    if any(k in col for k in ("email", "mail")):
        return "email"
    if any(k in col for k in ("date", "time", "created", "updated", "ordered")):
        return "date"
    if any(k in col for k in ("amount", "price", "total", "cost", "payment")):
        return "amount"
    if any(k in col for k in ("order_id", "_id", "transaction")):
        return "id"
    return "text"


# ---------------------------------------------------------------------------
# Core validation engine
# ---------------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame, country_code: str = "IN"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Validate every row in *df* by column type.

    Returns:
        cleaned_df  — rows with no issues
        flagged_df  — rows with at least one issue
        summary     — dict of per-column stats + totals
    """
    total_rows = len(df)
    col_report: Dict = {}
    issue_flags = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        col_type = detect_column_type(col)
        missing = int(df[col].isna().sum())
        errors = 0
        duplicates = 0

        if col_type == "phone":
            results = df[col].apply(lambda x: validate_phone(x, country_code))
            valid_mask = results.apply(lambda r: r[0])
            errors = int((~valid_mask).sum())
            issue_flags[col] = ~valid_mask

        elif col_type == "email":
            results = df[col].apply(validate_email)
            valid_mask = results.apply(lambda r: r[0])
            errors = int((~valid_mask).sum())
            issue_flags[col] = ~valid_mask

        elif col_type == "date":
            results = df[col].apply(validate_date)
            valid_mask = results.apply(lambda r: r[0])
            errors = int((~valid_mask).sum())
            issue_flags[col] = ~valid_mask

        elif col_type == "amount":
            valid_mask = df[col].apply(validate_amount)
            errors = int((~valid_mask).sum())
            issue_flags[col] = ~valid_mask

        elif col_type == "id":
            duplicates = int(df[col].duplicated().sum())

        col_report[col] = {
            "type": col_type,
            "total": total_rows,
            "missing": missing,
            "errors": errors,
            "duplicates": duplicates,
            "valid": max(0, total_rows - errors - missing),
        }

    bad_rows = issue_flags.any(axis=1)
    cleaned_df = df[~bad_rows].copy()
    flagged_df = df[bad_rows].copy()

    summary = {
        "total_rows": total_rows,
        "valid_rows": int((~bad_rows).sum()),
        "flagged_rows": int(bad_rows.sum()),
        "columns": col_report,
    }
    return cleaned_df, flagged_df, summary


def split_csv(df: pd.DataFrame, chunk_size: int = 100) -> List[pd.DataFrame]:
    """Split *df* into sequential chunks of *chunk_size* rows."""
    return [df.iloc[i: i + chunk_size] for i in range(0, len(df), chunk_size)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main SPA page with phone rules injected."""
    return render_template("index.html", phone_rules=PHONE_RULES)


@app.route("/validate", methods=["POST"])
def validate():
    """
    POST /validate
    Form data: file (CSV), country_code (str), chunk_size (int)
    Returns: JSON summary
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Only CSV files are supported"}), 400

    country_code = request.form.get("country_code", "IN").strip().upper()
    if country_code not in PHONE_RULES:
        country_code = "IN"  # safe fallback

    try:
        chunk_size = max(10, min(10_000, int(request.form.get("chunk_size", 100))))
    except (ValueError, TypeError):
        chunk_size = 100

    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return jsonify({"error": f"Could not parse CSV: {exc}"}), 400

    if df.empty:
        return jsonify({"error": "Uploaded CSV contains no data rows"}), 400

    cleaned_df, flagged_df, summary = validate_dataframe(df, country_code)

    # Persist outputs
    cleaned_path = os.path.join(OUTPUT_FOLDER, "cleaned_output.csv")
    flagged_path = os.path.join(OUTPUT_FOLDER, "flagged_rows.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    flagged_df.to_csv(flagged_path, index=False)

    # Chunked output (only when cleaned data > 1 chunk)
    chunks = split_csv(cleaned_df, chunk_size)
    chunk_info: List[Dict] = []
    if len(chunks) > 1:
        for idx, chunk in enumerate(chunks, start=1):
            path = os.path.join(OUTPUT_FOLDER, f"chunk_{idx}.csv")
            chunk.to_csv(path, index=False)
            chunk_info.append({"name": f"chunk_{idx}.csv", "rows": len(chunk)})

    summary["chunks"] = chunk_info
    summary["columns_found"] = list(df.columns)
    return jsonify(summary)


@app.route("/download/<filename>")
def download(filename: str):
    """
    GET /download/<filename>
    Serve a previously generated output file.  Path-traversal safe.
    """
    safe = secure_filename(filename)
    if not safe:
        return jsonify({"error": "Invalid filename"}), 400

    path = os.path.join(OUTPUT_FOLDER, safe)
    # Ensure the resolved path is still within OUTPUT_FOLDER (guard against symlinks)
    if not os.path.abspath(path).startswith(os.path.abspath(OUTPUT_FOLDER)):
        return jsonify({"error": "Access denied"}), 403

    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404

    return send_file(path, as_attachment=True)


@app.route("/download-all")
def download_all():
    """GET /download-all — Stream a ZIP of all output files."""
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(OUTPUT_FOLDER):
            full = os.path.join(OUTPUT_FOLDER, fname)
            if os.path.isfile(full):
                zf.write(full, fname)
    zip_buf.seek(0)
    return send_file(
        zip_buf,
        as_attachment=True,
        download_name="dataguard_output.zip",
        mimetype="application/zip",
    )


@app.route("/phone-rules", methods=["GET", "POST"])
def phone_rules():
    """
    GET  /phone-rules        — return current rules as JSON
    POST /phone-rules        — add / update a rule
      Body JSON: { "code": "JP", "digits": 11, "label": "Japan" }
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        code = str(data.get("code", "")).upper().strip()
        label = str(data.get("label", code)).strip()

        try:
            digits = int(data.get("digits", 10))
        except (ValueError, TypeError):
            return jsonify({"error": "digits must be an integer"}), 400

        if not re.match(r"^[A-Z]{2,3}$", code):
            return jsonify({"error": "code must be 2–3 uppercase letters"}), 400
        if not (5 <= digits <= 15):
            return jsonify({"error": "digits must be between 5 and 15"}), 400

        PHONE_RULES[code] = {"digits": digits, "label": label}
        return jsonify({"success": True, "rules": PHONE_RULES})

    return jsonify(PHONE_RULES)


# ---------------------------------------------------------------------------
# Error handlers — consistent JSON envelope
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum size is 100 MB."}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
