import os
from datetime import datetime


def analyze_metadata(
    file_path
):

    findings = []

    metadata = {}


    try:

        stats = os.stat(
            file_path
        )

        metadata = {

            "file_type":
                os.path.splitext(
                    file_path
                )[1].lower(),

            "file_size":
                stats.st_size,

            "created":
                datetime.fromtimestamp(
                    stats.st_ctime
                ).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "modified":
                datetime.fromtimestamp(
                    stats.st_mtime
                ).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }


        if stats.st_mtime != stats.st_ctime:

            findings.append(
                "File has a modification timestamp."
            )


    except Exception as e:

        findings.append(
            f"Metadata analysis error: {str(e)}"
        )


    return {

        "metadata": metadata,

        "findings": findings
    }