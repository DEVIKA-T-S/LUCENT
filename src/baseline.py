import os
import json


BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "baseline.json"
)


def calculate_baseline(
    blink_rate,
    avg_blink_duration,
    avg_blink_interval,
    face_size
):
    """
    Calculate the user's personal baseline.
    """

    baseline = {
        "blink_rate": blink_rate,
        "avg_blink_duration": avg_blink_duration,
        "avg_blink_interval": avg_blink_interval,
        "face_size": face_size
    }

    return baseline


def save_baseline(baseline):
    """
    Save the personal baseline to data/baseline.json.
    """

    os.makedirs(
        os.path.dirname(BASELINE_PATH),
        exist_ok=True
    )

    with open(BASELINE_PATH, "w") as file:
        json.dump(
            baseline,
            file,
            indent=4
        )


def load_baseline():
    """
    Load the saved personal baseline.
    """

    if not os.path.exists(BASELINE_PATH):
        return None

    with open(BASELINE_PATH, "r") as file:
        baseline = json.load(file)

    return baseline