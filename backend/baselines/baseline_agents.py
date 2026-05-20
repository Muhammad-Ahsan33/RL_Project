# baseline_agents.py

class FixedScheduleBaseline:
    """
    Irrigates every 3 days with 150L regardless of anything.
    Represents what most small farms in Pakistan do today.
    """
    def __init__(self, irrigate_every_days=3, volume_action=2):
        self.irrigate_every_days = irrigate_every_days
        self.volume_action       = volume_action   # action 2 = 150L

    def predict(self, obs, step):
        day = step / 12   # 12 steps per day
        if int(day) % self.irrigate_every_days == 0:
            return self.volume_action
        return 0   # skip


class ThresholdBaseline:
    def __init__(self, threshold=0.28, volume_action=2):
        self.threshold     = threshold
        self.volume_action = volume_action

    def predict(self, obs, step=None):
        root_moisture = obs[1]
        if root_moisture < self.threshold:
            return self.volume_action
        return 0


print("Baseline agents loaded ✓")