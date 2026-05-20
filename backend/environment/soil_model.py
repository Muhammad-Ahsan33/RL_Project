# soil_model.py
import numpy as np

class SoilModel:
    """
    3-layer bucket model of soil water dynamics.
    Layers: surface (0-10cm), root zone (10-30cm), deep (30-60cm)
    All moisture values in fraction: 0.0 = bone dry, 1.0 = fully saturated
    """

    # Soil constants (loamy soil — typical for Pakistani Punjab agriculture)
    WILTING_POINT    = 0.12   # crop dies below this
    FIELD_CAPACITY   = 0.40   # drainage kicks in above this
    SATURATION       = 0.45   # fully waterlogged

    # Fraction transferred between layers per time step (2h)
    # SURFACE_TO_ROOT  = 0.18
    # ROOT_TO_DEEP     = 0.08
    # DEEP_DRAINAGE    = 0.04

    SURFACE_TO_ROOT  = 0.35
    ROOT_TO_DEEP     = 0.25
    DEEP_DRAINAGE    = 0.15

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Start season with realistic mid-range moisture"""
        self.moisture = np.array([
            self.rng.uniform(0.20, 0.30),   # surface
            self.rng.uniform(0.22, 0.32),   # root zone
            self.rng.uniform(0.25, 0.33),   # deep
        ], dtype=np.float32)
        return self.moisture.copy()

    def step(self, irrigation_mm, rainfall_mm, temperature_c, humidity_pct):
        """
        Advance soil moisture by one 2-hour timestep.

        Args:
            irrigation_mm  : water applied in mm (0 if no irrigation)
            rainfall_mm    : rainfall in mm for this 2h window
            temperature_c  : ambient temperature in Celsius
            humidity_pct   : relative humidity 0-100

        Returns:
            moisture  : np.array shape (3,), updated moisture fractions
            runoff_mm : water lost as runoff (surface already at capacity)
        """
        m = self.moisture.copy()

        # Convert mm to volumetric fraction (assuming 100mm effective depth per layer)
        mm_per_unit = 100.0

        # 1. Add water to surface layer
        surface_input = (irrigation_mm + rainfall_mm) / mm_per_unit
        m[0] += surface_input

        # 2. Runoff: surface can't hold more than saturation
        runoff = max(0.0, m[0] - self.SATURATION)
        m[0] = min(m[0], self.SATURATION)

        # 3. Evapotranspiration from surface and root zone
        # et = self._calc_et(temperature_c, humidity_pct)
        # et_surface = et * 0.4   # surface bears 40% of ET
        # et_root    = et * 0.6   # root zone bears 60%

        et = self._calc_et(temperature_c, humidity_pct) * 12.0  # daily ET
        et_surface = et * 0.4
        et_root    = et * 0.6

        m[0] = max(0.0, m[0] - et_surface)
        m[1] = max(0.0, m[1] - et_root)

        # 4. Percolation: surface → root zone (only above field capacity)
        if m[0] > self.FIELD_CAPACITY:
            perc = (m[0] - self.FIELD_CAPACITY) * self.SURFACE_TO_ROOT
            m[0] -= perc
            m[1] += perc

        # 5. Percolation: root zone → deep
        if m[1] > self.FIELD_CAPACITY:
            perc = (m[1] - self.FIELD_CAPACITY) * self.ROOT_TO_DEEP
            m[1] -= perc
            m[2] += perc

        # 6. Deep drainage (water lost from system)
        if m[2] > self.FIELD_CAPACITY:
            m[2] -= (m[2] - self.FIELD_CAPACITY) * self.DEEP_DRAINAGE

        # Clamp all layers
        m = np.clip(m, 0.0, self.SATURATION)

        self.moisture = m
        return self.moisture.copy(), runoff

    # def _calc_et(self, temp_c, humidity_pct):
    #     """
    #     Simplified Hargreaves-Samani evapotranspiration.
    #     Returns ET in volumetric fraction for a 2-hour window.
    #     """
    #     if temp_c <= 0:
    #         return 0.0
    #     # Base ET driven by temperature and vapour pressure deficit
    #     vpd = max(0.0, 1.0 - humidity_pct / 100.0)
    #     et_daily = 0.0023 * (temp_c + 17.8) * np.sqrt(max(temp_c, 0)) * vpd
    #     et_2h = et_daily / 12.0   # daily → 2-hour window
    #     return float(np.clip(et_2h / 100.0, 0.0, 0.015))  # convert mm→fraction


    def _calc_et(self, temp_c, humidity_pct):
      if temp_c <= 0:
          return 0.0
      vpd      = max(0.0, 1.0 - humidity_pct / 100.0)
      et_daily = 0.0023 * (temp_c + 17.8) * np.sqrt(max(temp_c, 0)) * vpd
      et_2h    = et_daily / 12.0 * 4.5
      return float(np.clip(et_2h / 50.0, 0.0, 0.025))  # was /10.0 now /30.0
    


    @property
    def is_stressed(self):
        """True if root zone moisture is below wilting point"""
        return self.moisture[1] < self.WILTING_POINT

    @property
    def is_waterlogged(self):
        """True if root zone is above field capacity significantly"""
        return self.moisture[1] > self.FIELD_CAPACITY + 0.05

print("SoilModel loaded ✓")