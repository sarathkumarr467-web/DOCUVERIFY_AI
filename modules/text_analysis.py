def analyze_text(
    text
):

    findings = []

    score = 0

    if not text or not text.strip():

        findings.append(
            "No readable text found in the document."
        )

        score += 10

        return {
            "risk_score": score,
            "findings": findings
        }


    # ======================================
    # SUSPICIOUS KEYWORDS
    # ======================================

    suspicious_keywords = [

        "edited",
        "modified",
        "altered",
        "fake",
        "forged",
        "fraud",
        "duplicate",
        "invalid",
        "tampered"
    ]


    text_lower = text.lower()

    detected = []


    for keyword in suspicious_keywords:

        if keyword in text_lower:

            detected.append(
                keyword
            )


    if detected:

        score += min(
            len(detected) * 10,
            40
        )

        findings.append(
            "Potentially suspicious terms detected: "
            + ", ".join(detected)
        )


    else:

        findings.append(
            "No major suspicious text indicators detected."
        )


    # ======================================
    # VERY SHORT TEXT
    # ======================================

    if len(text.strip()) < 20:

        findings.append(
            "Document contains very little readable text."
        )

        score += 5


    score = min(
        score,
        100
    )


    return {

        "risk_score": score,

        "findings": findings
    }