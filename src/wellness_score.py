def calculate_wellness_score(
    blink_rate,
    avg_blink_duration,
    avg_blink_interval,
    state
):
    """
    Calculate a simple LUCENT monitoring score.

    This is a project-level indicator and is not
    a medical or clinical score.
    """

    score = 100

    # Blink-rate contribution
    if blink_rate < 4:
        score -= 30
    elif blink_rate < 6:
        score -= 20
    elif blink_rate < 8:
        score -= 10

    # Blink-duration contribution
    if avg_blink_duration > 0.8:
        score -= 15
    elif avg_blink_duration > 0.6:
        score -= 10

    # Blink-interval contribution
    if avg_blink_interval > 20:
        score -= 20
    elif avg_blink_interval > 12:
        score -= 10

    # ML state contribution
    if state == "Low Blinking":
        score -= 10
    elif state == "Mild Fatigue":
        score -= 20
    elif state == "High Fatigue":
        score -= 35

    # Keep score between 0 and 100
    score = max(0, min(100, score))

    return score


def get_score_category(score):
    """
    Convert the numerical score into a simple category.
    """

    if score >= 90:
        return "Good"
    elif score >= 70:
        return "Moderate"
    elif score >= 50:
        return "Needs Attention"
    else:
        return "High Attention"