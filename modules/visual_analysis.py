import fitz


def analyze_pdf_visuals(
    file_path
):

    findings = []

    page_results = []


    if not file_path.lower().endswith(
        ".pdf"
    ):

        return {

            "visual_score": 0,

            "findings": [
                "Visual PDF analysis is not applicable."
            ],

            "page_results": []
        }


    try:

        pdf = fitz.open(
            file_path
        )


        for page_number, page in enumerate(
            pdf,
            start=1
        ):

            rect = page.rect

            width = rect.width

            height = rect.height


            text_blocks = page.get_text(
                "blocks"
            )

            images = page.get_images(
                full=True
            )

            drawings = page.get_drawings()


            page_results.append({

                "page":
                    page_number,

                "width":
                    round(
                        width,
                        2
                    ),

                "height":
                    round(
                        height,
                        2
                    ),

                "text_blocks":
                    len(
                        text_blocks
                    ),

                "images":
                    len(
                        images
                    ),

                "drawings":
                    len(
                        drawings
                    )
            })


        pdf.close()


        # ==================================
        # PAGE SIZE CONSISTENCY
        # ==================================

        if len(page_results) > 1:

            page_sizes = set(

                (
                    page["width"],
                    page["height"]
                )

                for page in page_results
            )


            if len(page_sizes) > 1:

                findings.append(
                    "Different page dimensions detected."
                )


        # ==================================
        # IMAGE-BASED DOCUMENT
        # ==================================

        total_images = sum(

            page["images"]

            for page in page_results
        )


        total_text_blocks = sum(

            page["text_blocks"]

            for page in page_results
        )


        if (
            total_images > 0
            and total_text_blocks == 0
        ):

            findings.append(
                "Document appears to be image-based."
            )


        # ==================================
        # NO ISSUES
        # ==================================

        if not findings:

            findings.append(
                "No obvious visual/layout inconsistencies detected."
            )


        # ==================================
        # VISUAL RISK
        # ==================================

        visual_score = 0


        # Different dimensions can be a signal,
        # but not proof of tampering.

        if (
            "Different page dimensions detected."
            in findings
        ):

            visual_score += 20


        # Image-based PDFs receive ZERO risk
        # points.

        visual_score = min(
            max(
                visual_score,
                0
            ),
            100
        )


        return {

            "visual_score":
                visual_score,

            "findings":
                findings,

            "page_results":
                page_results
        }


    except Exception as e:

        return {

            "visual_score":
                0,

            "findings": [
                f"Visual analysis error: {str(e)}"
            ],

            "page_results":
                []
        }