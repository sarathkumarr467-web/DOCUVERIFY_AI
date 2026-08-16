def calculate_risk(
    text_analysis,
    metadata_result,
    pdf_metadata_result,
    visual_result
):

    score = 0

    findings = []


    # ======================================
    # TEXT SCORE
    # ======================================

    text_score = text_analysis.get(
        "risk_score",
        0
    )

    score += text_score


    for finding in text_analysis.get(
        "findings",
        []
    ):

        if finding not in findings:

            findings.append(
                finding
            )


    # ======================================
    # VISUAL SCORE
    # ======================================

    visual_score = visual_result.get(
        "visual_score",
        0
    )

    score += visual_score


    for finding in visual_result.get(
        "findings",
        []
    ):

        if (
            finding
            == "Document appears to be image-based."
        ):
            continue

        if finding not in findings:

            findings.append(
                finding
            )


    # ======================================
    # PDF METADATA
    # ======================================

    for finding in pdf_metadata_result.get(
        "findings",
        []
    ):

        # Missing creator is informational,
        # not evidence of tampering.

        if (
            "not available"
            in finding.lower()
        ):

            continue


        if finding not in findings:

            findings.append(
                finding
            )


    # ======================================
    # FILE METADATA
    # ======================================

    # Modification timestamps alone are NOT
    # treated as tampering evidence.


    # ======================================
    # LIMIT SCORE
    # ======================================

    score = min(
        max(
            score,
            0
        ),
        100
    )


    # ======================================
    # RISK LEVEL
    # ======================================

    if score >= 60:

        risk_level = "HIGH"

    elif score >= 30:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # ======================================
    # DEFAULT FINDING
    # ======================================

    if not findings:

        findings.append(
            "No major suspicious indicators detected."
        )


    return {

        "risk_score":
            score,

        "risk_level":
            risk_level,

        "findings":
            findings
    }