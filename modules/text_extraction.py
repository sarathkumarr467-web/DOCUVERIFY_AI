import os
import fitz
from docx import Document
from PIL import Image
import pytesseract


# ==========================================
# TESSERACT PATH
# ==========================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if os.path.exists(TESSERACT_PATH):

    pytesseract.pytesseract.tesseract_cmd = (
        TESSERACT_PATH
    )


# ==========================================
# PDF EXTRACTION
# ==========================================

def extract_pdf_text(
    file_path
):

    text_parts = []

    pdf = fitz.open(
        file_path
    )

    for page in pdf:

        page_text = page.get_text(
            "text"
        )

        if page_text.strip():

            text_parts.append(
                page_text
            )

        else:

            # --------------------------
            # OCR IMAGE-BASED PDF
            # --------------------------

            pix = page.get_pixmap(
                matrix=fitz.Matrix(
                    2,
                    2
                )
            )

            image = Image.frombytes(
                "RGB",
                [
                    pix.width,
                    pix.height
                ],
                pix.samples
            )

            try:

                ocr_text = pytesseract.image_to_string(
                    image
                )

                if ocr_text.strip():

                    text_parts.append(
                        ocr_text
                    )

            except Exception as e:

                text_parts.append(
                    f"OCR Error: {str(e)}"
                )

    pdf.close()

    return "\n".join(
        text_parts
    ).strip()


# ==========================================
# DOCX EXTRACTION
# ==========================================

def extract_docx_text(
    file_path
):

    document = Document(
        file_path
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            paragraphs.append(
                paragraph.text
            )

    return "\n".join(
        paragraphs
    ).strip()


# ==========================================
# IMAGE OCR
# ==========================================

def extract_image_text(
    file_path
):

    image = Image.open(
        file_path
    )

    try:

        return pytesseract.image_to_string(
            image
        ).strip()

    except Exception as e:

        return (
            f"OCR Error: {str(e)}"
        )


# ==========================================
# MAIN EXTRACTION FUNCTION
# ==========================================

def extract_text(
    file_path
):

    extension = os.path.splitext(
        file_path
    )[1].lower()

    try:

        if extension == ".pdf":

            return extract_pdf_text(
                file_path
            )

        elif extension == ".docx":

            return extract_docx_text(
                file_path
            )

        elif extension in [
            ".png",
            ".jpg",
            ".jpeg"
        ]:

            return extract_image_text(
                file_path
            )

        else:

            return (
                "Unsupported document format."
            )

    except Exception as e:

        return (
            f"Text extraction error: {str(e)}"
        )