import sqlite3
import os
from datetime import datetime


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE = os.path.join(
    BASE_DIR,
    "docuverify.db"
)


# ==========================================
# CONNECTION
# ==========================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    return connection


# ==========================================
# INITIALIZE
# ==========================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            filename TEXT NOT NULL,

            risk_score INTEGER NOT NULL,

            risk_level TEXT NOT NULL,

            findings TEXT,

            analyzed_at TEXT NOT NULL

        )
    """)

    connection.commit()

    connection.close()


# ==========================================
# SAVE ANALYSIS
# ==========================================

def save_analysis(
    filename,
    risk_score,
    risk_level,
    findings
):

    connection = get_connection()

    cursor = connection.cursor()

    findings_text = "\n".join(
        findings
    )

    analyzed_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO documents
        (
            filename,
            risk_score,
            risk_level,
            findings,
            analyzed_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        filename,
        risk_score,
        risk_level,
        findings_text,
        analyzed_at
    ))

    document_id = cursor.lastrowid

    connection.commit()

    connection.close()

    return document_id


# ==========================================
# ALL DOCUMENTS
# ==========================================

def get_all_documents():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM documents
        ORDER BY id DESC
    """)

    documents = cursor.fetchall()

    connection.close()

    return documents


# ==========================================
# STATISTICS
# ==========================================

def get_statistics():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM documents
    """)

    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM documents
        WHERE risk_level = 'LOW'
    """)

    low = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM documents
        WHERE risk_level = 'MEDIUM'
    """)

    medium = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM documents
        WHERE risk_level = 'HIGH'
    """)

    high = cursor.fetchone()[0]

    connection.close()

    return {
        "total": total,
        "low": low,
        "medium": medium,
        "high": high
    }


# ==========================================
# SINGLE DOCUMENT
# ==========================================

def get_document(
    document_id
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM documents
        WHERE id = ?
    """, (
        document_id,
    ))

    document = cursor.fetchone()

    connection.close()

    return document