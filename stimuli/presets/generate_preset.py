import numpy as np


class Preset:
    def __init__(self, glo, lam_I):
        self.B = np.array([
            [1, 1, 1, 0, 0, ],   # MOTION STRUCTURE COMPONENT MATRIX
            [1, 1, 0, 1, 0, ],   # each row describes one dot (N dots)
            [1, 0, 0, 0, 1, ],   # each column is a motion source (M sources)
            ], dtype=np.float64)
        volatility_factor = 2  # Makes the stimulus change more rapidly without affecting the covariance matrix
        speed_factor = 1.0
        lam_T = 2.                                                        # total
        lam_G = np.sqrt(glo) * np.sqrt(max(0., lam_T ** 2 - lam_I ** 2))  # global
        lam_C = np.sqrt(max(0., lam_T ** 2 - lam_G ** 2 - lam_I ** 2))    # cluster
        lam_M = np.sqrt(max(0., lam_T ** 2 - lam_G ** 2))                 # maverick
        self.lam = speed_factor * np.sqrt(volatility_factor) * np.array([lam_G, lam_C, lam_I, lam_I, lam_M])
        # THE FULL MOTION STRUCTURE MATRIX WILL BE: L = B @ diag(lam)
        # TIME CONSTANT for significant changes in angular velocities (in seconds)
        self.tau_vphi = 2. / volatility_factor
        self.L = self.B @ np.diag(self.lam)

    def getL(self):
        return self.B @ np.diag(self.lam)
