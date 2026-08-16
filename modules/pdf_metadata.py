import fitz
import os


def analyze_pdf_metadata(
    file_path
):

    metadata = {}

    findings = []


    if not file_path.lower().endswith(
        ".pdf"
    ):

        return {
            "metadata": {},
            "findings": []
        }


    try:

        pdf = fitz.open(
            file_path
        )

        metadata = pdf.metadata or {}

        author = metadata.get(
            "author"
        )

        creator = metadata.get(
            "creator"
        )

        producer = metadata.get(
            "producer"
        )

        creation_date = metadata.get(
            "creationDate"
        )

        modification_date = metadata.get(
            "modDate"
        )


        if not creator:

            findings.append(
                "Creator information is not available."
            )


        pdf.close()


        return {

            "metadata": {

                "author":
                    author or "Not available",

                "creator":
                    creator or "Not available",

                "producer":
                    producer or "Not available",

                "creation_date":
                    creation_date or "Not available",

                "modification_date":
                    modification_date or "Not available"
            },

            "findings": findings
        }


    except Exception as e:

        return {

            "metadata": {},

            "findings": [
                f"PDF metadata error: {str(e)}"
            ]
        }