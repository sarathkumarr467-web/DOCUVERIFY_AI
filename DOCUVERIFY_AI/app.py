import os
import re
import sqlite3
import json
from datetime import datetime
from functools import wraps

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_file,
    abort
)

from werkzeug.utils import secure_filename
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

# =========================================================
# OPTIONAL DOCUMENT LIBRARIES
# =========================================================

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


# =========================================================
# DATA SCIENCE / ML
# =========================================================

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import joblib
except ImportError:
    joblib = None


# =========================================================
# PDF REPORT
# =========================================================

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )
except ImportError:
    A4 = None
    colors = None
    getSampleStyleSheet = None
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    Table = None
    TableStyle = None


# =========================================================
# FLASK CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = "DOCUVERIFY_AI_SECRET_KEY_2026"

BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

DATABASE = os.path.join(
    BASE_DIR,
    "docuverify.db"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "reports"
)

# =========================================================
# DATASET
# =========================================================

DATASET_PATH = os.path.join(
    BASE_DIR,
    "docuverify_dataset.csv"
)

DATASET_FOLDER_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "docuverify_dataset.csv"
)


# =========================================================
# ML MODEL FILES
# =========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "docuverify_model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "feature_names.pkl"
)


# =========================================================
# EDA
# =========================================================

EDA_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "eda"
)


# =========================================================
# FILE SETTINGS
# =========================================================

ALLOWED_EXTENSIONS = {
    "pdf",
    "docx",
    "txt",
    "png",
    "jpg",
    "jpeg"
}

MAX_FILE_SIZE = 16 * 1024 * 1024


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# =========================================================
# CREATE REQUIRED FOLDERS
# =========================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)

os.makedirs(
    EDA_FOLDER,
    exist_ok=True
)


# =========================================================
# FIND DATASET
# =========================================================

def get_dataset_path():

    if os.path.exists(DATASET_PATH):

        return DATASET_PATH

    if os.path.exists(DATASET_FOLDER_PATH):

        return DATASET_FOLDER_PATH

    return None


# =========================================================
# LOAD ML MODEL
# =========================================================

model = None
label_encoder = None
feature_names = []

MODEL_ERROR = None


if joblib is not None:

    try:

        if os.path.exists(MODEL_PATH):

            model = joblib.load(
                MODEL_PATH
            )

        if os.path.exists(ENCODER_PATH):

            label_encoder = joblib.load(
                ENCODER_PATH
            )

        if os.path.exists(FEATURE_PATH):

            feature_names = joblib.load(
                FEATURE_PATH
            )

            # Make sure feature_names is a list
            if feature_names is None:

                feature_names = []

            elif not isinstance(
                feature_names,
                (list, tuple)
            ):

                try:

                    feature_names = list(
                        feature_names
                    )

                except Exception:

                    feature_names = []

    except Exception as error:

        MODEL_ERROR = str(error)

else:

    MODEL_ERROR = (
        "joblib is not installed."
    )


# =========================================================
# DATABASE
# =========================================================

def get_db():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    return connection


# =========================================================
# CHECK COLUMN
# =========================================================

def column_exists(
    connection,
    table_name,
    column_name
):

    columns = connection.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    return any(
        column["name"] == column_name
        for column in columns
    )


# =========================================================
# ADD COLUMN IF MISSING
# =========================================================

def add_column_if_missing(
    connection,
    table_name,
    column_name,
    definition
):

    if not column_exists(
        connection,
        table_name,
        column_name
    ):

        connection.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN {column_name} {definition}
            """
        )


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_db():

    connection = get_db()

    # =====================================================
    # USERS TABLE
    # =====================================================

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL,

            created_at TEXT NOT NULL

        )
        """
    )

    # =====================================================
    # DOCUMENTS TABLE
    # =====================================================

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT,

            filename TEXT NOT NULL,

            file_type TEXT,

            file_size INTEGER DEFAULT 0,

            extracted_text TEXT,

            result TEXT DEFAULT 'Genuine',

            risk_level TEXT DEFAULT 'Low',

            risk_score REAL DEFAULT 0,

            confidence REAL DEFAULT 0,

            reasons TEXT,

            text_length REAL DEFAULT 0,

            text_quality REAL DEFAULT 0,

            metadata_score REAL DEFAULT 0,

            structure_score REAL DEFAULT 0,

            field_score REAL DEFAULT 0,

            suspicious_keywords REAL DEFAULT 0,

            repeated_lines REAL DEFAULT 0,

            abnormal_symbols REAL DEFAULT 0,

            missing_fields REAL DEFAULT 0,

            metadata_anomaly REAL DEFAULT 0,

            tampering_indicators REAL DEFAULT 0,

            analyzed_at TEXT,

            updated_at TEXT

        )
        """
    )

    # =====================================================
    # MIGRATION
    # =====================================================

    columns = {

        "username": "TEXT",

        "filename": "TEXT",

        "file_type": "TEXT",

        "file_size": "INTEGER DEFAULT 0",

        "extracted_text": "TEXT",

        "result": "TEXT DEFAULT 'Genuine'",

        "risk_level": "TEXT DEFAULT 'Low'",

        "risk_score": "REAL DEFAULT 0",

        "confidence": "REAL DEFAULT 0",

        "reasons": "TEXT",

        "text_length": "REAL DEFAULT 0",

        "text_quality": "REAL DEFAULT 0",

        "metadata_score": "REAL DEFAULT 0",

        "structure_score": "REAL DEFAULT 0",

        "field_score": "REAL DEFAULT 0",

        "suspicious_keywords": "REAL DEFAULT 0",

        "repeated_lines": "REAL DEFAULT 0",

        "abnormal_symbols": "REAL DEFAULT 0",

        "missing_fields": "REAL DEFAULT 0",

        "metadata_anomaly": "REAL DEFAULT 0",

        "tampering_indicators": "REAL DEFAULT 0",

        "analyzed_at": "TEXT",

        "updated_at": "TEXT"
    }

    for name, definition in columns.items():

        add_column_if_missing(
            connection,
            "documents",
            name,
            definition
        )

    # =====================================================
    # FIX OLD NULL VALUES
    # =====================================================

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    connection.execute(
        """
        UPDATE documents
        SET username = 'admin'
        WHERE username IS NULL
        """
    )

    connection.execute(
        """
        UPDATE documents
        SET filename = 'Unknown Document'
        WHERE filename IS NULL
        """
    )

    connection.execute(
        """
        UPDATE documents
        SET analyzed_at = ?
        WHERE analyzed_at IS NULL
        """,
        (now,)
    )

    connection.execute(
        """
        UPDATE documents
        SET updated_at = ?
        WHERE updated_at IS NULL
        """,
        (now,)
    )

    # =====================================================
    # DEFAULT ADMIN
    # =====================================================

    existing = connection.execute(
        """
        SELECT id
        FROM users
        WHERE username = ?
        """,
        ("admin",)
    ).fetchone()

    if existing is None:

        connection.execute(
            """
            INSERT INTO users
            (
                username,
                password,
                created_at
            )
            VALUES (?, ?, ?)
            """,
            (
                "admin",
                generate_password_hash(
                    "admin123"
                ),
                now
            )
        )

    connection.commit()

    connection.close()


init_db()


# =========================================================
# LOGIN REQUIRED
# =========================================================

def login_required(function):

    @wraps(function)
    def decorated_function(
        *args,
        **kwargs
    ):

        if "username" not in session:

            return redirect(
                url_for(
                    "login",
                    next=request.path
                )
            )

        return function(
            *args,
            **kwargs
        )

    return decorated_function


# =========================================================
# FILE HELPERS
# =========================================================

def allowed_file(filename):

    if not filename:

        return False

    if "." not in filename:

        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


def get_extension(filename):

    if "." not in filename:

        return ""

    return filename.rsplit(
        ".",
        1
    )[1].lower()


# =========================================================
# TEXT EXTRACTION - PDF
# =========================================================

def extract_pdf_text(path):

    if PdfReader is None:

        return ""

    try:

        reader = PdfReader(path)

        text_parts = []

        for page in reader.pages:

            try:

                page_text = page.extract_text()

                if page_text:

                    text_parts.append(
                        page_text
                    )

            except Exception:

                continue

        return "\n".join(
            text_parts
        )

    except Exception as error:

        print(
            "PDF EXTRACTION ERROR:",
            error
        )

        return ""


# =========================================================
# TEXT EXTRACTION - DOCX
# =========================================================

def extract_docx_text(path):

    if DocxDocument is None:

        return ""

    try:

        document = DocxDocument(path)

        paragraphs = []

        for paragraph in document.paragraphs:

            if paragraph.text:

                paragraphs.append(
                    paragraph.text
                )

        return "\n".join(
            paragraphs
        )

    except Exception as error:

        print(
            "DOCX EXTRACTION ERROR:",
            error
        )

        return ""


# =========================================================
# TEXT EXTRACTION - TXT
# =========================================================

def extract_txt_text(path):

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            return file.read()

    except Exception as error:

        print(
            "TXT EXTRACTION ERROR:",
            error
        )

        return ""


# =========================================================
# TEXT EXTRACTION - IMAGE OCR
# =========================================================

def extract_image_text(path):

    if (
        Image is None
        or pytesseract is None
    ):

        return ""

    try:

        image = Image.open(path)

        return pytesseract.image_to_string(
            image
        )

    except Exception as error:

        print(
            "OCR ERROR:",
            error
        )

        return ""


# =========================================================
# MAIN TEXT EXTRACTION
# =========================================================

def extract_text(
    path,
    extension
):

    if extension == "pdf":

        return extract_pdf_text(
            path
        )

    elif extension == "docx":

        return extract_docx_text(
            path
        )

    elif extension == "txt":

        return extract_txt_text(
            path
        )

    elif extension in {
        "png",
        "jpg",
        "jpeg"
    }:

        return extract_image_text(
            path
        )

    return ""


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(
    filename,
    text,
    file_size
):

    text = text or ""

    clean_text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    text_length = len(
        clean_text
    )

    # =====================================================
    # TEXT QUALITY
    # =====================================================

    if text_length == 0:

        text_quality = 5

    else:

        alphabetic = len(
            re.findall(
                r"[A-Za-z]",
                clean_text
            )
        )

        text_quality = (
            alphabetic /
            max(
                text_length,
                1
            )
        ) * 100

        text_quality = min(
            100,
            round(
                text_quality,
                2
            )
        )

    # =====================================================
    # SUSPICIOUS KEYWORDS
    # =====================================================

    suspicious_list = [

        "fake",
        "forged",
        "forgery",
        "tampered",
        "tampering",
        "edited",
        "invalid",
        "duplicate",
        "not genuine",
        "fraud",
        "fraudulent",
        "manipulated",
        "altered",
        "counterfeit"

    ]

    lower_text = text.lower()

    suspicious_keywords = sum(

        lower_text.count(
            keyword
        )

        for keyword in suspicious_list
    )

    suspicious_keywords = min(
        suspicious_keywords,
        10
    )

    # =====================================================
    # REPEATED LINES
    # =====================================================

    lines = [

        line.strip()

        for line in text.splitlines()

        if line.strip()
    ]

    repeated_lines = 0

    if len(lines) > 1:

        repeated_lines = (

            len(lines)

            -

            len(
                set(lines)
            )
        )

    repeated_lines = min(
        repeated_lines,
        10
    )

    # =====================================================
    # ABNORMAL SYMBOLS
    # =====================================================

    abnormal_matches = re.findall(
        r"[@#$%^&*]{2,}",
        text
    )

    abnormal_symbols = min(
        len(abnormal_matches),
        10
    )

    # =====================================================
    # COMMON DOCUMENT FIELDS
    # =====================================================

    fields = [

        "name",
        "date",
        "signature",
        "address"

    ]

    missing_fields = sum(

        1

        for field in fields

        if field not in lower_text
    )

    # =====================================================
    # METADATA SCORE
    # =====================================================

    extension = get_extension(
        filename
    )

    metadata_score = 85

    if extension not in {
        "pdf",
        "docx"
    }:

        metadata_score -= 10

    if file_size <= 0:

        metadata_score -= 30

    metadata_score = max(
        0,
        min(
            100,
            metadata_score
        )
    )

    # =====================================================
    # METADATA ANOMALY
    # =====================================================

    metadata_anomaly = 0

    if file_size < 1000:

        metadata_anomaly += 2

    # Images naturally have different metadata
    # so do not heavily penalize them.

    metadata_anomaly = min(
        metadata_anomaly,
        10
    )

    # =====================================================
    # STRUCTURE SCORE
    # =====================================================

    structure_score = 85

    if text_length == 0:

        structure_score -= 45

    elif text_length < 100:

        structure_score -= 25

    if repeated_lines > 0:

        structure_score -= (
            repeated_lines * 3
        )

    if abnormal_symbols > 0:

        structure_score -= (
            abnormal_symbols * 2
        )

    structure_score = max(
        0,
        min(
            100,
            structure_score
        )
    )

    # =====================================================
    # FIELD SCORE
    # =====================================================

    field_score = (

        100

        -

        (
            missing_fields * 15
        )
    )

    field_score = max(
        0,
        min(
            100,
            field_score
        )
    )

    # =====================================================
    # TAMPERING INDICATORS
    # =====================================================

    tampering_indicators = (

        suspicious_keywords

        +

        repeated_lines

        +

        abnormal_symbols
    )

    tampering_indicators = min(
        tampering_indicators,
        10
    )

    # =====================================================
    # RISK SCORE
    # =====================================================
    #
    # Lower score = safer
    # Higher score = more suspicious
    #
    # =====================================================

    risk_score = 0

    # Text quality contribution
    risk_score += (
        (100 - text_quality)
        * 0.10
    )

    # Metadata contribution
    risk_score += (
        (100 - metadata_score)
        * 0.05
    )

    # Structure contribution
    risk_score += (
        (100 - structure_score)
        * 0.10
    )

    # Missing fields contribution
    risk_score += (
        missing_fields
        * 2
    )

    # Suspicious keywords
    risk_score += (
        suspicious_keywords
        * 5
    )

    # Repeated lines
    risk_score += (
        repeated_lines
        * 2
    )

    # Abnormal symbols
    risk_score += (
        abnormal_symbols
        * 2
    )

    # Metadata anomaly
    risk_score += (
        metadata_anomaly
        * 2
    )

    # Tampering
    risk_score += (
        tampering_indicators
        * 2
    )

    risk_score = round(
        min(
            100,
            max(
                0,
                risk_score
            )
        ),
        2
    )

    return {

        "text_length":
            text_length,

        "text_quality":
            text_quality,

        "metadata_score":
            metadata_score,

        "structure_score":
            structure_score,

        "field_score":
            field_score,

        "suspicious_keywords":
            suspicious_keywords,

        "repeated_lines":
            repeated_lines,

        "abnormal_symbols":
            abnormal_symbols,

        "missing_fields":
            missing_fields,

        "metadata_anomaly":
            metadata_anomaly,

        "tampering_indicators":
            tampering_indicators,

        "risk_score":
            risk_score
    }


# =========================================================
# ML PREDICTION + FINAL CLASSIFICATION
# =========================================================

def predict_document(features):

    if model is None:

        raise RuntimeError(
            "ML model not loaded. "
            "Check docuverify_model.pkl."
        )

    # =====================================================
    # FEATURE DICTIONARY
    # =====================================================

    feature_values = {

        "text_length":
            features.get(
                "text_length",
                0
            ),

        "text_quality":
            features.get(
                "text_quality",
                0
            ),

        "metadata_score":
            features.get(
                "metadata_score",
                0
            ),

        "structure_score":
            features.get(
                "structure_score",
                0
            ),

        "field_score":
            features.get(
                "field_score",
                0
            ),

        "suspicious_keywords":
            features.get(
                "suspicious_keywords",
                0
            ),

        "repeated_lines":
            features.get(
                "repeated_lines",
                0
            ),

        "abnormal_symbols":
            features.get(
                "abnormal_symbols",
                0
            ),

        "missing_fields":
            features.get(
                "missing_fields",
                0
            ),

        "metadata_anomaly":
            features.get(
                "metadata_anomaly",
                0
            ),

        "tampering_indicators":
            features.get(
                "tampering_indicators",
                0
            ),

        "risk_score":
            features.get(
                "risk_score",
                0
            )
    }

    # =====================================================
    # FEATURE ORDER
    # =====================================================

    if feature_names:

        values = [

            feature_values.get(
                str(feature),
                0
            )

            for feature in feature_names
        ]

    else:

        values = [

            feature_values[
                "text_length"
            ],

            feature_values[
                "text_quality"
            ],

            feature_values[
                "metadata_score"
            ],

            feature_values[
                "structure_score"
            ],

            feature_values[
                "field_score"
            ],

            feature_values[
                "suspicious_keywords"
            ],

            feature_values[
                "repeated_lines"
            ],

            feature_values[
                "abnormal_symbols"
            ],

            feature_values[
                "missing_fields"
            ],

            feature_values[
                "metadata_anomaly"
            ],

            feature_values[
                "tampering_indicators"
            ],

            feature_values[
                "risk_score"
            ]
        ]

    # =====================================================
    # ML MODEL PREDICTION
    # =====================================================

    try:

        prediction = model.predict(
            [values]
        )[0]

    except Exception as error:

        print(
            "MODEL PREDICTION ERROR:",
            error
        )

        raise RuntimeError(
            "Model feature mismatch. "
            "Make sure feature_names.pkl "
            "matches the trained model."
        )

    # =====================================================
    # DECODE ML RESULT
    # =====================================================

    if label_encoder is not None:

        try:

            ml_result = label_encoder.inverse_transform(
                [prediction]
            )[0]

        except Exception:

            ml_result = str(
                prediction
            )

    else:

        ml_result = str(
            prediction
        )

    ml_result = str(
        ml_result
    ).strip()

    # =====================================================
    # MODEL CONFIDENCE
    # =====================================================

    model_confidence = 0

    try:

        probabilities = model.predict_proba(
            [values]
        )[0]

        model_confidence = round(
            float(
                max(probabilities)
            ) * 100,
            2
        )

    except Exception:

        model_confidence = 0

    # =====================================================
    # RISK SCORE
    # =====================================================

    risk_score = float(
        features.get(
            "risk_score",
            0
        )
    )

    risk_score = max(
        0,
        min(
            100,
            risk_score
        )
    )

    # =====================================================
    # FINAL CLASSIFICATION
    # =====================================================
    #
    # 0 - 34  = Genuine
    # 35 - 64 = Suspicious
    # 65 -100 = Fake
    #
    # =====================================================

    if risk_score >= 65:

        result = "Fake"

        risk_level = "High"

    elif risk_score >= 35:

        result = "Suspicious"

        risk_level = "Medium"

    else:

        result = "Genuine"

        risk_level = "Low"

    # =====================================================
    # CONFIDENCE
    # =====================================================

    if model_confidence > 0:

        confidence = model_confidence

    else:

        if result == "Genuine":

            confidence = (
                100 - risk_score
            )

        elif result == "Suspicious":

            confidence = (
                70 -
                abs(
                    risk_score - 50
                )
            )

        else:

            confidence = risk_score

    confidence = round(
        max(
            0,
            min(
                100,
                confidence
            )
        ),
        2
    )

    # =====================================================
    # REASONS
    # =====================================================

    reasons = []

    # Text quality

    if features["text_quality"] < 50:

        reasons.append(
            "Low text quality was detected."
        )

    else:

        reasons.append(
            "Text quality is within an acceptable range."
        )

    # Suspicious keywords

    if features["suspicious_keywords"] > 0:

        reasons.append(
            f'{features["suspicious_keywords"]} '
            'suspicious keyword indicator(s) detected.'
        )

    else:

        reasons.append(
            "No major suspicious keywords were detected."
        )

    # Repeated lines

    if features["repeated_lines"] > 0:

        reasons.append(
            "Repeated text patterns were detected."
        )

    else:

        reasons.append(
            "No significant repeated-line pattern was detected."
        )

    # Abnormal symbols

    if features["abnormal_symbols"] > 0:

        reasons.append(
            "Abnormal symbol patterns were detected."
        )

    else:

        reasons.append(
            "No abnormal symbol pattern was detected."
        )

    # Missing fields

    if features["missing_fields"] >= 3:

        reasons.append(
            "Several common document fields are missing."
        )

    else:

        reasons.append(
            "Most common document fields were detected."
        )

    # Metadata

    if features["metadata_anomaly"] > 0:

        reasons.append(
            "Potential metadata anomalies were detected."
        )

    else:

        reasons.append(
            "No significant metadata anomaly was detected."
        )

    # Tampering

    if features["tampering_indicators"] > 0:

        reasons.append(
            "Potential tampering indicators were identified."
        )

    else:

        reasons.append(
            "No strong tampering indicator was identified."
        )

    # ML result

    reasons.append(
        f"ML Model Prediction: {ml_result}"
    )

    reasons.append(
        f"Risk Score: {risk_score:.2f}/100"
    )

    # =====================================================
    # DEBUG
    # =====================================================

    print()
    print(
        "======================================"
    )
    print(
        "       DOCUVERIFY AI ANALYSIS"
    )
    print(
        "======================================"
    )
    print(
        "ML MODEL RESULT :",
        ml_result
    )
    print(
        "FINAL RESULT    :",
        result
    )
    print(
        "RISK LEVEL      :",
        risk_level
    )
    print(
        "RISK SCORE      :",
        risk_score
    )
    print(
        "CONFIDENCE      :",
        confidence,
        "%"
    )
    print(
        "======================================"
    )
    print()

    return {

        "result":
            result,

        "risk_level":
            risk_level,

        "risk_score":
            round(
                risk_score,
                2
            ),

        "confidence":
            confidence,

        "reasons":
            reasons,

        "ml_result":
            ml_result
    }


# =========================================================
# SAVE DOCUMENT
# =========================================================

def save_document(
    username,
    filename,
    file_type,
    file_size,
    extracted_text,
    features,
    prediction,
    document_id=None
):

    connection = get_db()

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    reasons_text = "\n".join(
        prediction.get(
            "reasons",
            []
        )
    )

    values = (

        username,

        filename,

        file_type,

        file_size,

        extracted_text,

        prediction["result"],

        prediction["risk_level"],

        prediction["risk_score"],

        prediction["confidence"],

        reasons_text,

        features["text_length"],

        features["text_quality"],

        features["metadata_score"],

        features["structure_score"],

        features["field_score"],

        features["suspicious_keywords"],

        features["repeated_lines"],

        features["abnormal_symbols"],

        features["missing_fields"],

        features["metadata_anomaly"],

        features["tampering_indicators"],

        now,

        now
    )

    # =====================================================
    # INSERT
    # =====================================================

    if document_id is None:

        cursor = connection.execute(
            """
            INSERT INTO documents
            (
                username,
                filename,
                file_type,
                file_size,
                extracted_text,
                result,
                risk_level,
                risk_score,
                confidence,
                reasons,
                text_length,
                text_quality,
                metadata_score,
                structure_score,
                field_score,
                suspicious_keywords,
                repeated_lines,
                abnormal_symbols,
                missing_fields,
                metadata_anomaly,
                tampering_indicators,
                analyzed_at,
                updated_at
            )
            VALUES
            (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?
            )
            """,
            values
        )

        document_id = cursor.lastrowid

    # =====================================================
    # UPDATE
    # =====================================================

    else:

        connection.execute(
            """
            UPDATE documents
            SET
                username = ?,
                filename = ?,
                file_type = ?,
                file_size = ?,
                extracted_text = ?,
                result = ?,
                risk_level = ?,
                risk_score = ?,
                confidence = ?,
                reasons = ?,
                text_length = ?,
                text_quality = ?,
                metadata_score = ?,
                structure_score = ?,
                field_score = ?,
                suspicious_keywords = ?,
                repeated_lines = ?,
                abnormal_symbols = ?,
                missing_fields = ?,
                metadata_anomaly = ?,
                tampering_indicators = ?,
                analyzed_at = ?,
                updated_at = ?
            WHERE id = ?
            AND username = ?
            """,
            values + (
                document_id,
                username
            )
        )

    connection.commit()

    connection.close()

    return document_id


# =========================================================
# STATISTICS
# =========================================================

def get_statistics(username):

    connection = get_db()

    total = connection.execute(
        """
        SELECT COUNT(*)
        FROM documents
        WHERE username = ?
        """,
        (username,)
    ).fetchone()[0]

    genuine = connection.execute(
        """
        SELECT COUNT(*)
        FROM documents
        WHERE username = ?
        AND result = 'Genuine'
        """,
        (username,)
    ).fetchone()[0]

    suspicious = connection.execute(
        """
        SELECT COUNT(*)
        FROM documents
        WHERE username = ?
        AND result = 'Suspicious'
        """,
        (username,)
    ).fetchone()[0]

    fake = connection.execute(
        """
        SELECT COUNT(*)
        FROM documents
        WHERE username = ?
        AND result = 'Fake'
        """,
        (username,)
    ).fetchone()[0]

    low = connection.execute(
        """
        SELECT COUNT(*)
        FROM documents
        WHERE username = ?
        AND risk_level = 'Low'
        """,
        (username,)
    ).fetchone()[0]

    medium = connection.execute(
        """
        SELECT COUNT(*)
        FROM documents
        WHERE username = ?
        AND risk_level = 'Medium'
        """,
        (username,)
    ).fetchone()[0]

    high = connection.execute(
        """
        SELECT COUNT(*)
        FROM documents
        WHERE username = ?
        AND risk_level = 'High'
        """,
        (username,)
    ).fetchone()[0]

    connection.close()

    return {

        "total":
            total,

        "genuine":
            genuine,

        "suspicious":
            suspicious,

        "fake":
            fake,

        "low":
            low,

        "medium":
            medium,

        "high":
            high
    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    if "username" not in session:

        return redirect(
            url_for("login")
        )

    return redirect(
        url_for("dashboard")
    )


# =========================================================
# LOGIN
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if not username:

            return render_template(
                "login.html",
                error="Username is required."
            )

        if not password:

            return render_template(
                "login.html",
                error="Password is required."
            )

        if len(username) > 12:

            return render_template(
                "login.html",
                error=(
                    "Username must not exceed "
                    "12 characters."
                )
            )

        if len(password) > 8:

            return render_template(
                "login.html",
                error=(
                    "Password must not exceed "
                    "8 characters."
                )
            )

        connection = get_db()

        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            """,
            (username,)
        ).fetchone()

        connection.close()

        if user and check_password_hash(
            user["password"],
            password
        ):

            session.clear()

            session["logged_in"] = True

            session["username"] = username

            return redirect(
                url_for("dashboard")
            )

        return render_template(
            "login.html",
            error=(
                "Invalid username or password."
            )
        )

    return render_template(
        "login.html"
    )


# =========================================================
# REGISTER
# =========================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not username or not password:

            return render_template(
                "register.html",
                error=(
                    "All fields are required."
                )
            )

        if len(username) > 12:

            return render_template(
                "register.html",
                error=(
                    "Username must be maximum "
                    "12 characters."
                )
            )

        if len(password) > 8:

            return render_template(
                "register.html",
                error=(
                    "Password must be maximum "
                    "8 characters."
                )
            )

        if re.search(
            r"\s",
            username
        ):

            return render_template(
                "register.html",
                error=(
                    "Username cannot contain spaces."
                )
            )

        if re.search(
            r"\s",
            password
        ):

            return render_template(
                "register.html",
                error=(
                    "Password cannot contain spaces."
                )
            )

        if password != confirm_password:

            return render_template(
                "register.html",
                error=(
                    "Passwords do not match."
                )
            )

        connection = get_db()

        existing = connection.execute(
            """
            SELECT id
            FROM users
            WHERE username = ?
            """,
            (username,)
        ).fetchone()

        if existing:

            connection.close()

            return render_template(
                "register.html",
                error=(
                    "Username already exists."
                )
            )

        connection.execute(
            """
            INSERT INTO users
            (
                username,
                password,
                created_at
            )
            VALUES (?, ?, ?)
            """,
            (
                username,

                generate_password_hash(
                    password
                ),

                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )

        connection.commit()

        connection.close()

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
@login_required
def dashboard():

    username = session["username"]

    stats = get_statistics(
        username
    )

    # =====================================================
    # DATASET INFORMATION
    # =====================================================

    dataset_rows = 0
    dataset_columns = 0

    dataset_fake = 0
    dataset_genuine = 0
    dataset_suspicious = 0

    dataset_path = get_dataset_path()

    if (
        dataset_path
        and pd is not None
    ):

        try:

            dataset = pd.read_csv(
                dataset_path
            )

            dataset_rows = len(
                dataset
            )

            dataset_columns = len(
                dataset.columns
            )

            if "label" in dataset.columns:

                counts = dataset[
                    "label"
                ].astype(str).str.strip()

                dataset_fake = int(
                    (
                        counts.str.lower()
                        == "fake"
                    ).sum()
                )

                dataset_genuine = int(
                    (
                        counts.str.lower()
                        == "genuine"
                    ).sum()
                )

                dataset_suspicious = int(
                    (
                        counts.str.lower()
                        == "suspicious"
                    ).sum()
                )

        except Exception as error:

            print(
                "DATASET ERROR:",
                error
            )

    # =====================================================
    # RECENT DOCUMENTS
    # =====================================================

    connection = get_db()

    recent_documents = connection.execute(
        """
        SELECT
            id,
            filename,
            file_type,
            result,
            risk_level,
            risk_score,
            confidence,
            analyzed_at

        FROM documents

        WHERE username = ?

        ORDER BY id DESC

        LIMIT 8
        """,
        (username,)
    ).fetchall()

    connection.close()

    return render_template(

        "dashboard.html",

        username=username,

        total=stats["total"],

        low=stats["low"],

        medium=stats["medium"],

        high=stats["high"],

        genuine_documents=
            stats["genuine"],

        suspicious_documents=
            stats["suspicious"],

        fake_documents=
            stats["fake"],

        dataset_size=
            dataset_rows,

        dataset_rows=
            dataset_rows,

        dataset_columns=
            dataset_columns,

        dataset_fake=
            dataset_fake,

        dataset_genuine=
            dataset_genuine,

        dataset_suspicious=
            dataset_suspicious,

        # These values should be replaced
        # with actual training metrics
        # if you have them.
        accuracy="N/A",

        precision="N/A",

        recall="N/A",

        f1_score="N/A",

        dataset_updated=
            datetime.now().strftime(
                "%d %b %Y"
            ),

        recent_documents=
            recent_documents,

        model_loaded=
            model is not None,

        model_error=
            MODEL_ERROR
    )


# =========================================================
# DATASET DETAILS
# =========================================================

@app.route("/dataset-details")
@login_required
def dataset_details():

    dataset_path = get_dataset_path()

    if dataset_path is None:

        return render_template(
            "dataset_details.html",
            error=(
                "Dataset not found. "
                "Place docuverify_dataset.csv "
                "inside the project folder or "
                "project/dataset folder."
            )
        )

    if pd is None:

        return render_template(
            "dataset_details.html",
            error=(
                "Pandas is not installed."
            )
        )

    try:

        df = pd.read_csv(
            dataset_path
        )

        rows = len(df)

        columns = len(
            df.columns
        )

        target_column = (

            "label"

            if "label" in df.columns

            else df.columns[-1]
        )

        feature_columns = [

            col

            for col in df.columns

            if col != target_column
        ]

        missing_values = int(
            df.isnull()
            .sum()
            .sum()
        )

        duplicate_rows = int(
            df.duplicated()
            .sum()
        )

        label_counts = (
            df[target_column]
            .value_counts()
            .to_dict()
        )

        total = len(df)

        label_distribution = []

        for label, count in label_counts.items():

            percentage = round(
                (
                    count /
                    max(
                        total,
                        1
                    )
                ) * 100,
                2
            )

            label_distribution.append({

                "label":
                    str(label),

                "count":
                    int(count),

                "percentage":
                    percentage
            })

        column_details = []

        for column in df.columns:

            column_details.append({

                "name":
                    column,

                "dtype":
                    str(
                        df[column].dtype
                    ),

                "unique":
                    int(
                        df[column].nunique()
                    ),

                "missing":
                    int(
                        df[column]
                        .isnull()
                        .sum()
                    )
            })

        return render_template(

            "dataset_details.html",

            rows=rows,

            columns=columns,

            target_column=
                target_column,

            feature_count=
                len(feature_columns),

            feature_columns=
                feature_columns,

            missing_values=
                missing_values,

            duplicate_rows=
                duplicate_rows,

            label_distribution=
                label_distribution,

            column_details=
                column_details
        )

    except Exception as error:

        return render_template(

            "dataset_details.html",

            error=str(error)
        )


# =========================================================
# EDA
# =========================================================

@app.route("/eda")
@login_required
def eda():

    dataset_path = get_dataset_path()

    if dataset_path is None:

        return render_template(

            "eda.html",

            error=(
                "Dataset not found. "
                "Please add docuverify_dataset.csv."
            )
        )

    if pd is None:

        return render_template(

            "eda.html",

            error=(
                "Pandas is not installed."
            )
        )

    try:

        # =================================================
        # LOAD DATASET
        # =================================================

        df = pd.read_csv(
            dataset_path
        )

        rows = len(df)

        columns = len(
            df.columns
        )

        # =================================================
        # TARGET COLUMN
        # =================================================

        if "label" in df.columns:

            target_column = "label"

        elif "result" in df.columns:

            target_column = "result"

        elif "target" in df.columns:

            target_column = "target"

        else:

            target_column = df.columns[-1]

        # =================================================
        # MISSING / DUPLICATE
        # =================================================

        missing_values = int(
            df.isnull()
            .sum()
            .sum()
        )

        duplicate_rows = int(
            df.duplicated()
            .sum()
        )

        # =================================================
        # TARGET DISTRIBUTION
        # =================================================

        target_counts = (
            df[target_column]
            .value_counts()
        )

        labels = [

            str(label)

            for label
            in target_counts.index
        ]

        values = [

            int(value)

            for value
            in target_counts.values
        ]

        # =================================================
        # GRAPH 1
        # TARGET DISTRIBUTION
        # =================================================

        graph1 = os.path.join(
            EDA_FOLDER,
            "target_distribution.png"
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            labels,
            values
        )

        plt.title(
            "Document Classification Distribution",
            fontsize=15,
            fontweight="bold"
        )

        plt.xlabel(
            "Document Class"
        )

        plt.ylabel(
            "Number of Documents"
        )

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        plt.savefig(
            graph1,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        # =================================================
        # GRAPH 2
        # DATA QUALITY
        # =================================================

        graph2 = os.path.join(
            EDA_FOLDER,
            "data_quality.png"
        )

        quality_labels = [

            "Missing Values",

            "Duplicate Rows"
        ]

        quality_values = [

            missing_values,

            duplicate_rows
        ]

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            quality_labels,
            quality_values
        )

        plt.title(
            "Dataset Quality Analysis",
            fontsize=15,
            fontweight="bold"
        )

        plt.ylabel(
            "Count"
        )

        plt.tight_layout()

        plt.savefig(
            graph2,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        # =================================================
        # GRAPH 3
        # NUMERIC FEATURE
        # =================================================

        graph3 = os.path.join(
            EDA_FOLDER,
            "numeric_distribution.png"
        )

        numeric_columns = (
            df.select_dtypes(
                include="number"
            ).columns
        )

        plt.figure(
            figsize=(10, 5)
        )

        if len(
            numeric_columns
        ) > 0:

            first_numeric = (
                numeric_columns[0]
            )

            df[first_numeric].hist(
                bins=20
            )

            plt.title(
                f"Distribution of {first_numeric}",
                fontsize=15,
                fontweight="bold"
            )

            plt.xlabel(
                first_numeric
            )

            plt.ylabel(
                "Frequency"
            )

        else:

            plt.text(
                0.5,
                0.5,
                "No Numeric Features",
                ha="center",
                va="center",
                fontsize=16
            )

            plt.axis(
                "off"
            )

        plt.tight_layout()

        plt.savefig(
            graph3,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        # =================================================
        # COLUMN DETAILS
        # =================================================

        column_details = []

        for column in df.columns:

            column_details.append({

                "name":
                    column,

                "dtype":
                    str(
                        df[column].dtype
                    ),

                "unique":
                    int(
                        df[column]
                        .nunique()
                    ),

                "missing":
                    int(
                        df[column]
                        .isnull()
                        .sum()
                    )
            })

        # =================================================
        # NUMERIC SUMMARY
        # =================================================

        numeric_summary = []

        for column in numeric_columns:

            numeric_summary.append({

                "name":
                    column,

                "mean":
                    round(
                        float(
                            df[column]
                            .mean()
                        ),
                        2
                    ),

                "min":
                    round(
                        float(
                            df[column]
                            .min()
                        ),
                        2
                    ),

                "max":
                    round(
                        float(
                            df[column]
                            .max()
                        ),
                        2
                    )
            })

        # =================================================
        # RENDER
        # =================================================

        return render_template(

            "eda.html",

            rows=rows,

            columns=columns,

            missing_values=
                missing_values,

            duplicate_rows=
                duplicate_rows,

            target_column=
                target_column,

            labels=labels,

            values=values,

            column_details=
                column_details,

            numeric_summary=
                numeric_summary,

            target_graph=
                "eda/target_distribution.png",

            quality_graph=
                "eda/data_quality.png",

            numeric_graph=
                "eda/numeric_distribution.png"
        )

    except Exception as error:

        print(
            "EDA ERROR:",
            error
        )

        return render_template(

            "eda.html",

            error=str(error)
        )


# =========================================================
# UPLOAD PAGE
# =========================================================

@app.route(
    "/upload",
    methods=["GET"]
)
@login_required
def upload_page():

    return render_template(
        "upload.html"
    )


# =========================================================
# UPLOAD DOCUMENT
# =========================================================

@app.route(
    "/upload_document",
    methods=["POST"]
)
@login_required
def upload_document():

    # =====================================================
    # CHECK FILE
    # =====================================================

    if "document" not in request.files:

        return render_template(

            "upload.html",

            error=(
                "Please select a document."
            )
        )

    file = request.files[
        "document"
    ]

    if not file or not file.filename:

        return render_template(

            "upload.html",

            error=(
                "Please select a document."
            )
        )

    # =====================================================
    # CHECK EXTENSION
    # =====================================================

    if not allowed_file(
        file.filename
    ):

        return render_template(

            "upload.html",

            error=(
                "Unsupported file format. "
                "Use PDF, DOCX, TXT, PNG, JPG or JPEG."
            )
        )

    # =====================================================
    # CHECK MODEL
    # =====================================================

    if model is None:

        return render_template(

            "upload.html",

            error=(
                "ML model is not loaded. "
                "Please check docuverify_model.pkl."
            )
        )

    # =====================================================
    # SAFE FILENAME
    # =====================================================

    original_filename = secure_filename(
        file.filename
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d%H%M%S%f"
    )

    filename = (
        timestamp
        + "_"
        + original_filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # =====================================================
    # SAVE FILE
    # =====================================================

    file.save(
        filepath
    )

    file_size = os.path.getsize(
        filepath
    )

    extension = get_extension(
        original_filename
    )

    # =====================================================
    # EXTRACT TEXT
    # =====================================================

    extracted_text = extract_text(
        filepath,
        extension
    )

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    features = extract_features(

        original_filename,

        extracted_text,

        file_size
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    try:

        prediction = predict_document(
            features
        )

    except Exception as error:

        print(
            "PREDICTION ERROR:",
            error
        )

        return render_template(

            "upload.html",

            error=str(error)
        )

    # =====================================================
    # SAVE DATABASE
    # =====================================================

    document_id = save_document(

        session["username"],

        original_filename,

        extension.upper(),

        file_size,

        extracted_text,

        features,

        prediction
    )

    # =====================================================
    # RESULT
    # =====================================================

    return redirect(

        url_for(

            "result",

            document_id=document_id
        )
    )


# =========================================================
# UPDATE DOCUMENT
# =========================================================

@app.route(
    "/update",
    methods=["GET", "POST"]
)
@login_required
def update_document():

    if request.method == "GET":

        return render_template(
            "update.html"
        )

    if "document" not in request.files:

        return render_template(

            "update.html",

            error=(
                "Please select a document."
            )
        )

    file = request.files[
        "document"
    ]

    if not file or not file.filename:

        return render_template(

            "update.html",

            error=(
                "Please select a document."
            )
        )

    if not allowed_file(
        file.filename
    ):

        return render_template(

            "update.html",

            error=(
                "Unsupported file format."
            )
        )

    if model is None:

        return render_template(

            "update.html",

            error=(
                "ML model is not loaded."
            )
        )

    original_filename = secure_filename(
        file.filename
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d%H%M%S%f"
    )

    filename = (
        timestamp
        + "_updated_"
        + original_filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(
        filepath
    )

    file_size = os.path.getsize(
        filepath
    )

    extension = get_extension(
        original_filename
    )

    extracted_text = extract_text(
        filepath,
        extension
    )

    features = extract_features(

        original_filename,

        extracted_text,

        file_size
    )

    try:

        prediction = predict_document(
            features
        )

    except Exception as error:

        return render_template(

            "update.html",

            error=str(error)
        )

    document_id = save_document(

        session["username"],

        original_filename,

        extension.upper(),

        file_size,

        extracted_text,

        features,

        prediction
    )

    return redirect(

        url_for(

            "result",

            document_id=document_id
        )
    )


# =========================================================
# RESULT
# =========================================================

@app.route(
    "/result/<int:document_id>"
)
@login_required
def result(document_id):

    connection = get_db()

    document = connection.execute(

        """
        SELECT *
        FROM documents
        WHERE id = ?
        AND username = ?
        """,

        (
            document_id,

            session["username"]
        )

    ).fetchone()

    connection.close()

    if document is None:

        abort(404)

    reasons = []

    if document["reasons"]:

        reasons = [

            item.strip()

            for item

            in document[
                "reasons"
            ].splitlines()

            if item.strip()
        ]

    return render_template(

        "result.html",

        document=document,

        reasons=reasons,

        original_filename=
            document["filename"],

        file_type=
            document["file_type"],

        file_size=
            document["file_size"],

        risk_score=
            document["risk_score"],

        risk_level=
            document["risk_level"].upper(),

        findings=reasons,

        result=
            document["result"],

        confidence=
            document["confidence"]
    )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
@login_required
def history():

    connection = get_db()

    documents = connection.execute(

        """
        SELECT *
        FROM documents

        WHERE username = ?

        ORDER BY id DESC
        """,

        (
            session["username"],
        )

    ).fetchall()

    connection.close()

    return render_template(

        "history.html",

        documents=documents
    )


# =========================================================
# ANALYTICS
# =========================================================

@app.route("/analytics")
@login_required
def analytics():

    username = session["username"]

    stats = get_statistics(
        username
    )

    connection = get_db()

    average_score = connection.execute(

        """
        SELECT AVG(risk_score)
        FROM documents
        WHERE username = ?
        """,

        (username,)

    ).fetchone()[0]

    average_confidence = connection.execute(

        """
        SELECT AVG(confidence)
        FROM documents
        WHERE username = ?
        """,

        (username,)

    ).fetchone()[0]

    connection.close()

    average_score = round(
        average_score or 0,
        2
    )

    average_confidence = round(
        average_confidence or 0,
        2
    )

    return render_template(

        "analytics.html",

        total=
            stats["total"],

        genuine=
            stats["genuine"],

        suspicious=
            stats["suspicious"],

        fake=
            stats["fake"],

        low=
            stats["low"],

        medium=
            stats["medium"],

        high=
            stats["high"],

        average_score=
            average_score,

        average_confidence=
            average_confidence
    )


# =========================================================
# SECURITY
# =========================================================

@app.route("/security")
@login_required
def security():

    return render_template(
        "security.html"
    )


# =========================================================
# REPORTS
# =========================================================

@app.route("/reports")
@login_required
def reports():

    connection = get_db()

    documents = connection.execute(

        """
        SELECT *
        FROM documents

        WHERE username = ?

        ORDER BY id DESC
        """,

        (
            session["username"],
        )

    ).fetchall()

    connection.close()

    return render_template(

        "reports.html",

        documents=documents
    )


# =========================================================
# ABOUT
# =========================================================

@app.route("/about")
@login_required
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# CONTACT
# =========================================================

# =========================================================
# CONTACT
# =========================================================

@app.route(
    "/contact",
    methods=["GET", "POST"]
)
@login_required
def contact():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        subject = request.form.get(
            "subject",
            ""
        ).strip()

        message = request.form.get(
            "message",
            ""
        ).strip()

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        if not name:
            return render_template(
                "contact.html",
                error="Please enter your name."
            )

        if not email:
            return render_template(
                "contact.html",
                error="Please enter your email."
            )

        if not subject:
            return render_template(
                "contact.html",
                error="Please enter the subject."
            )

        if not message:
            return render_template(
                "contact.html",
                error="Please enter your message."
            )

        # -------------------------------------------------
        # BASIC EMAIL VALIDATION
        # -------------------------------------------------

        email_pattern = (
            r"^[A-Za-z0-9._%+-]+@"
            r"[A-Za-z0-9.-]+\."
            r"[A-Za-z]{2,}$"
        )

        if not re.match(
            email_pattern,
            email
        ):
            return render_template(
                "contact.html",
                error="Please enter a valid email address."
            )

        # -------------------------------------------------
        # SAVE CONTACT MESSAGE
        # -------------------------------------------------

        connection = get_db()

        # Create table if it does not exist
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS contact_messages (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                username TEXT,

                name TEXT NOT NULL,

                email TEXT NOT NULL,

                subject TEXT NOT NULL,

                message TEXT NOT NULL,

                created_at TEXT NOT NULL

            )
            """
        )

        connection.execute(
            """
            INSERT INTO contact_messages
            (
                username,
                name,
                email,
                subject,
                message,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session["username"],
                name,
                email,
                subject,
                message,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )

        connection.commit()
        connection.close()

        # -------------------------------------------------
        # SUCCESS MESSAGE
        # -------------------------------------------------

        flash(
            "Your message has been sent successfully!"
        )

        return redirect(
            url_for("contact")
        )

    # -----------------------------------------------------
    # GET REQUEST
    # -----------------------------------------------------

    return render_template(
        "contact.html"
    )


# =========================================================
# CLEAR HISTORY
# =========================================================

@app.route(
    "/clear_history",
    methods=["POST"]
)
@login_required
def clear_history():

    connection = get_db()

    connection.execute(

        """
        DELETE FROM documents
        WHERE username = ?
        """,

        (
            session["username"],
        )
    )

    connection.commit()

    connection.close()

    flash(
        "Document history cleared successfully."
    )

    return redirect(
        url_for("history")
    )


# =========================================================
# PDF REPORT
# =========================================================

@app.route(
    "/download_report/<int:document_id>"
)
@login_required
def download_report(
    document_id
):

    if A4 is None:

        return (
            "ReportLab is not installed. "
            "Run: pip install reportlab"
        )

    connection = get_db()

    document = connection.execute(

        """
        SELECT *
        FROM documents

        WHERE id = ?

        AND username = ?
        """,

        (
            document_id,

            session["username"]
        )

    ).fetchone()

    connection.close()

    if document is None:

        abort(404)

    report_name = (

        f"DocuVerify_Report_"
        f"{document_id}.pdf"
    )

    report_path = os.path.join(

        app.config["REPORT_FOLDER"],

        report_name
    )

    doc = SimpleDocTemplate(

        report_path,

        pagesize=A4,

        rightMargin=40,

        leftMargin=40,

        topMargin=40,

        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    # =====================================================
    # TITLE
    # =====================================================

    story.append(

        Paragraph(

            "DOCUVERIFY AI",

            styles["Title"]
        )
    )

    story.append(

        Paragraph(

            "ML-Based Document Verification Report",

            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # =====================================================
    # DOCUMENT DETAILS
    # =====================================================

    data = [

        [
            "Field",
            "Value"
        ],

        [
            "Document",
            document["filename"]
        ],

        [
            "File Type",
            document["file_type"] or "-"
        ],

        [
            "File Size",
            f'{document["file_size"]} bytes'
        ],

        [
            "Prediction",
            document["result"]
        ],

        [
            "Risk Level",
            document["risk_level"]
        ],

        [
            "Risk Score",
            f'{document["risk_score"]:.2f}%'
        ],

        [
            "Confidence",
            f'{document["confidence"]:.2f}%'
        ],

        [
            "Analyzed At",
            document["analyzed_at"] or "-"
        ]
    ]

    table = Table(

        data,

        colWidths=[
            150,
            330
        ]
    )

    table.setStyle(

        TableStyle(

            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#2563eb"
                    )
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )

    story.append(
        table
    )

    story.append(
        Spacer(1, 20)
    )

    # =====================================================
    # FEATURES
    # =====================================================

    story.append(

        Paragraph(

            "Feature Analysis",

            styles["Heading2"]
        )
    )

    feature_data = [

        [
            "Feature",
            "Value"
        ],

        [
            "Text Length",
            str(
                document["text_length"]
            )
        ],

        [
            "Text Quality",
            f'{document["text_quality"]:.2f}'
        ],

        [
            "Metadata Score",
            f'{document["metadata_score"]:.2f}'
        ],

        [
            "Structure Score",
            f'{document["structure_score"]:.2f}'
        ],

        [
            "Field Score",
            f'{document["field_score"]:.2f}'
        ],

        [
            "Suspicious Keywords",
            str(
                document[
                    "suspicious_keywords"
                ]
            )
        ],

        [
            "Repeated Lines",
            str(
                document[
                    "repeated_lines"
                ]
            )
        ],

        [
            "Abnormal Symbols",
            str(
                document[
                    "abnormal_symbols"
                ]
            )
        ],

        [
            "Missing Fields",
            str(
                document[
                    "missing_fields"
                ]
            )
        ],

        [
            "Metadata Anomaly",
            str(
                document[
                    "metadata_anomaly"
                ]
            )
        ],

        [
            "Tampering Indicators",
            str(
                document[
                    "tampering_indicators"
                ]
            )
        ]
    ]

    feature_table = Table(

        feature_data,

        colWidths=[
            220,
            260
        ]
    )

    feature_table.setStyle(

        TableStyle(

            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#0f766e"
                    )
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ]
        )
    )

    story.append(
        feature_table
    )

    story.append(
        Spacer(1, 20)
    )

    # =====================================================
    # EXPLANATION
    # =====================================================

    story.append(

        Paragraph(

            "Analysis Explanation",

            styles["Heading2"]
        )
    )

    if document["reasons"]:

        for reason in document[
            "reasons"
        ].splitlines():

            story.append(

                Paragraph(

                    "• " + reason,

                    styles["BodyText"]
                )
            )

            story.append(
                Spacer(1, 5)
            )

    story.append(
        Spacer(1, 15)
    )

    # =====================================================
    # MODEL
    # =====================================================

    story.append(

        Paragraph(

            "AI / ML Analysis",

            styles["Heading2"]
        )
    )

    story.append(

        Paragraph(

            "DocuVerify AI combines machine-learning "
            "prediction with document text, structure, "
            "field and tampering indicators.",

            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(

        Paragraph(

            "Disclaimer: This automated result is intended "
            "as a supporting document-analysis tool and "
            "should not be treated as absolute legal proof "
            "of authenticity.",

            styles["BodyText"]
        )
    )

    # =====================================================
    # BUILD
    # =====================================================

    doc.build(
        story
    )

    return send_file(

        report_path,

        as_attachment=True,

        download_name=report_name
    )


# =========================================================
# ERROR HANDLER - FILE TOO LARGE
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    if "username" in session:

        return render_template(

            "upload.html",

            error=(
                "File size must be below 16 MB."
            )

        ), 413

    return redirect(
        url_for("login")
    )


# =========================================================
# ERROR HANDLER - 404
# =========================================================

@app.errorhandler(404)
def page_not_found(error):

    if "username" in session:

        return render_template(

            "base.html",

            error=(
                "Page not found."
            )

        ), 404

    return (

        """
        <h2>Page Not Found</h2>

        <a href="/login">
            Go to Login
        </a>
        """

    ), 404


# =========================================================
# GENERAL ERROR HANDLER
# =========================================================

@app.errorhandler(Exception)
def handle_exception(error):

    app.logger.exception(
        "Unhandled application error"
    )

    return (

        f"""
        <h2>
            DOCUVERIFY AI - Application Error
        </h2>

        <p>
            Something went wrong.
        </p>

        <p>
            {str(error)}
        </p>

        <a href="/dashboard">
            Back to Dashboard
        </a>
        """

    ), 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print()
    print(
        "=============================================="
    )
    print(
        "          DOCUVERIFY AI"
    )
    print(
        " Multi-Format Document Verification System"
    )
    print(
        "=============================================="
    )
    print(
        "Model Loaded :",
        model is not None
    )
    print(
        "Encoder Loaded :",
        label_encoder is not None
    )
    print(
        "Features Loaded :",
        len(feature_names)
    )
    print(
        "Database :",
        DATABASE
    )
    print(
        "Dataset :",
        get_dataset_path()
    )
    print(
        "=============================================="
    )
    print()

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )