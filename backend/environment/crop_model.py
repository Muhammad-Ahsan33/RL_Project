# crop_model.py
import numpy as np

class CropModel:
    """
    Wheat crop growth model over a ~120-day season.
    Timestep: 2 hours → 1440 steps per season.
    Tracks growth stage, stress accumulation, and estimated yield.
    """

    # Growth stages and their duration in days
    STAGES = {
        0: ("germination",  10),
        1: ("vegetative",   30),
        2: ("tillering",    20),
        3: ("flowering",    15),   # most water-sensitive stage
        4: ("grain_fill",   30),
        5: ("maturity",     15),
    }

    # Water stress sensitivity multiplier per stage (flowering is 3x base)
    STRESS_WEIGHTS = {0: 1.0, 1: 1.5, 2: 1.5, 3: 3.0, 4: 2.0, 5: 0.8}

    def __init__(self):
        self.reset()

    def reset(self):
        self.day           = 0
        self.stage         = 0
        self.stress_days   = 0.0      # accumulated stress (weighted)
        self.waterlog_days = 0.0
        self.base_yield    = 100.0    # max possible yield score
        return self._get_obs()

    def step(self, root_moisture, wilting_point, field_capacity):
        """
        Advance crop by one 2-hour timestep.

        Args:
            root_moisture : current root zone moisture fraction
            wilting_point : soil wilting point
            field_capacity: soil field capacity

        Returns:
            crop_obs  : dict with stage, health, yield_estimate
            stress_penalty : float penalty for this step
        """
        # Advance time (12 steps = 1 day)
        self.day += 1 / 12.0

        # Update growth stage
        self._update_stage()

        # Calculate stress this step
        drought_stress   = self._drought_stress(root_moisture, wilting_point)
        waterlog_stress  = self._waterlog_stress(root_moisture, field_capacity)

        weight = self.STRESS_WEIGHTS[self.stage]
        stress_this_step = (drought_stress + waterlog_stress) * weight / 12.0

        self.stress_days   += drought_stress * weight / 12.0
        self.waterlog_days += waterlog_stress / 12.0

        penalty = stress_this_step * 2.0

        return self._get_obs(), penalty

    def _drought_stress(self, moisture, wilting_point):
        """0 = no stress, 1 = maximum drought stress"""
        optimal = 0.25   # optimal moisture fraction
        if moisture >= optimal:
            return 0.0
        elif moisture <= wilting_point:
            return 1.0
        else:
            return (optimal - moisture) / (optimal - wilting_point)

    def _waterlog_stress(self, moisture, field_capacity):
        """0 = no stress, 1 = maximum waterlog stress"""
        if moisture <= field_capacity:
            return 0.0
        overwater = moisture - field_capacity
        return min(1.0, overwater / 0.10)

    def _update_stage(self):
        cumulative = 0
        for stage_id, (name, duration) in self.STAGES.items():
            cumulative += duration
            if self.day <= cumulative:
                self.stage = stage_id
                break
            self.stage = 5  # maturity

    def get_yield_score(self):
        """Returns final yield score 0-100 based on accumulated stress"""
        stress_factor = min(1.0, self.stress_days / 30.0)
        yield_score   = self.base_yield * (1.0 - 0.7 * stress_factor)
        return max(0.0, yield_score)

    def _get_obs(self):
        return {
            "stage"         : self.stage,
            "stage_name"    : self.STAGES[self.stage][0],
            "day"           : self.day,
            "stress_days"   : self.stress_days,
            "yield_estimate": self.get_yield_score(),
        }

    @property
    def is_done(self):
        return self.day >= 120

print("CropModel loaded ✓")