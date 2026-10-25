import time


class AlertEngine:
    """
    LUCENT Alert Engine.

    Decides when an alert should be triggered
    based on the current monitoring state.
    """

    def __init__(
        self,
        cooldown_seconds=60,
        persistence_count=3
    ):
        self.cooldown_seconds = cooldown_seconds
        self.persistence_count = persistence_count

        self.condition_count = 0
        self.last_alert_time = 0

    def check_alert(self, current_state):
        """
        Check whether an alert should be triggered.

        Returns:
            True  -> trigger alert
            False -> no alert
        """

        alert_states = [
            "Low Blinking",
            "Mild Fatigue",
            "High Fatigue"
        ]

        if current_state in alert_states:
            self.condition_count += 1
        else:
            self.condition_count = 0

        current_time = time.time()

        enough_persistence = (
            self.condition_count >= self.persistence_count
        )

        cooldown_finished = (
            current_time - self.last_alert_time
            >= self.cooldown_seconds
        )

        if enough_persistence and cooldown_finished:
            self.last_alert_time = current_time
            self.condition_count = 0

            return True

        return False