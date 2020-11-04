import numpy as np
from numpy.linalg import inv, det

from config import SimulationConfig as SimConfig, ExperimentConfig as ExpConfig
from stimuli.motion_structure import MotionStructure


class MotionStructureKalmanFilter:
    def __init__(self, structure, z_0, σ_obs=0.):
        """
        Kalman filter for motion structure identification.
        :param structure: Structure of the motion.
        :type  structure: MotionStructure
        :param z_0: Initial state of the motion.
        :type  z_0: numpy.ndarray
        :param σ_obs: Observation noise.
        :type  σ_obs: float
        """
        n, n_v = structure.n, structure.n_v()  # n locations, m velocities
        n_z = n + n_v
        self.F = structure.build_F_φ()      # State-transition model F_t = I + F * dt
        D = structure.build_D_φ()
        self.Q = D @ D.T                     # Covariance of the process noise Q_t = Q * dt
        self.H = np.zeros((n, n_z))          # Observation model H_t = H
        self.H[:n, :n] = np.eye(n)           # Only locations are observable
        self.R = σ_obs ** 2 * np.eye(n)        # Covariance of the observation noise R_t = R
        assert len(z_0) == n_z, f"Error: Expect initial state z_0 (length: {len(z_0)}) to have length {n_z}!"
        self.z = z_0                         # Initialize the hidden state with true initial state
        self.P = structure.τ_ω / 2 * self.Q  # The true initial state is sampled from the stationary distribution
        self.I = np.eye(n_z)
        self.L_const = n * np.log(2 * np.pi)
        self.n = n

    def predict(self, dt):
        """
        Prediction step of the Kalman filter
        :param dt: time increment length
        :type dt: float
        """
        Ft = self.I + self.F * dt         # State-transition model F_t = I + F * dt
        Qt = self.Q * dt                  # covariance of the process noise Q_t = Q * dt
        self.z = Ft @ self.z              # Predicted (a priori) state estimate z_{t|t-1}
        self.z[:self.n] %= 2 * np.pi
        # print(dt)
        self.P = Ft @ self.P @ Ft.T + Qt  # Predicted (a priori) error covariance P_{t|t-1}

    def update(self, x):
        """
        Update step of the Kalman filter
        :param x: observed state
        :return: log likelihood of the observed state given the Kalman filter
        """
        y = x - self.H @ self.z                  # Pre-fit residual y_{t|t-1}
        y = (y + np.pi) % (2 * np.pi) - np.pi
        S = self.H @ self.P @ self.H.T + self.R  # Pre-fit residual covariance S_t
        S_inv = inv(S)
        K = self.P @ self.H.T @ S_inv            # Optimal Kalman gain K_t
        self.z = self.z + K @ y                  # Updated (a posteriori) state estimate z_{t|t}
        self.z[:self.n] %= 2 * np.pi
        self.P = (self.I - K @ self.H) @ self.P  # Updated (a posteriori) estimate covariance P_{t|t}
        dL = -(y.T @ S_inv @ y + np.log(det(S)) + self.L_const) / 2  # log-likelihood for the current observation
        return dL


def apply_kalman_filter(x: np.ndarray, t: np.ndarray, structure: MotionStructure, σ_obs: float = 0.):
    n = ExpConfig.n_dots
    L = 0
    f = MotionStructureKalmanFilter(structure, x[0] if SimConfig.whiten else x[0][np.r_[:n, -n:0]], σ_obs)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        f.predict(dt)
        L += f.update(x[i][:n])
    return L
