def classify_behavior(text: str):

    text = text.lower()

    if "risk" in text or "avoid" in text or "cut" in text:
        return "risk_averse"

    if "expand" in text or "scale" in text or "aggressive" in text:
        return "aggressive"

    if "tech" in text or "system" in text or "architecture" in text:
        return "technical"

    if "analyze" in text or "data" in text or "strategy" in text:
        return "analytical"

    return "neutral"