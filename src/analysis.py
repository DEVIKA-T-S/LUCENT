from src.database import get_recent_sessions
def calculate_change(before, after):
    """
    Calculate the change between before and after values.
    """

    return after - before


def calculate_percentage_change(before, after):
    """
    Calculate percentage change.
    """

    if before == 0:
        return 0

    return ((after - before) / before) * 100


def analyze_before_after(before, after):
    """
    Compare monitoring metrics before and after an intervention.
    """

    analysis = {}

    metrics = [
        "blink_rate",
        "avg_blink_duration",
        "avg_blink_interval",
        "wellness_score"
    ]

    for metric in metrics:

        before_value = before.get(metric, 0)
        after_value = after.get(metric, 0)

        analysis[metric] = {
            "before": before_value,
            "after": after_value,
            "change": calculate_change(
                before_value,
                after_value
            ),
            "percentage_change": calculate_percentage_change(
                before_value,
                after_value
            )
        }

    analysis["state_before"] = before.get(
        "state",
        "Unknown"
    )

    analysis["state_after"] = after.get(
        "state",
        "Unknown"
    )

    return analysis

def session_to_metrics(session):
    """
    Convert a database session row into
    the format used by the analysis module.
    """

    return {
        "blink_rate": session[3],
        "avg_blink_duration": session[4],
        "avg_blink_interval": session[5],
        "face_size": session[6],
        "state": session[7],
        "wellness_score": session[8]
    }    

def compare_sessions(before_session, after_session):
    """
    Compare two database sessions.
    """

    before_metrics = session_to_metrics(
        before_session
    )

    after_metrics = session_to_metrics(
        after_session
    )

    return analyze_before_after(
        before_metrics,
        after_metrics
    )    