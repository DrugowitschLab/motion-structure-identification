from config import SimulationConfig as sim
from utils.data import Logger, load_data
from stimuli.motion_structure import MotionStructure
from filterpy.kalman import KalmanFilter
from johannes_kalman_filter_code.classKalman import PhiKalmanFilterPermutation
import numpy as np
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal as mvn
from time import time
import csv


class MotionStructureKalmanFilter:
    def __init__(self, structure, z_0, σ_R=0.):
        """

        :param structure:
        :type  structure: MotionStructure
        :param z_0:
        :type  z_0: numpy.ndarray
        :param σ_R:
        :type  σ_R: float
        :param whiten:
        :type  whiten: bool
        """
        n, n_v = structure.n, structure.n_v()  # n locations, m velocities
        n_z = n + n_v
        self.F = structure.build_F_φ()      # State-transition model F_t = I + F * dt
        D = structure.build_D_φ()
        self.Q = D @ D.T                     # Covariance of the process noise Q_t = Q * dt
        self.H = np.zeros((n, n_z))          # Observation model H_t = H
        self.H[:n, :n] = np.eye(n)           # Only locations are observable
        self.R = σ_R ** 2 * np.eye(n)        # Covariance of the observation noise R_t = R
        assert len(z_0) == n_z, f"Error: Expect initial state z_0 (length: {len(z_0)}) to have length {n_z}!"
        self.z = z_0                         # Initialize the hidden state with true initial state
        self.P = structure.τ_ω / 2 * self.Q  # The true initial state is sampled from the stationary distribution
        self.I = np.eye(n_z)
        self.ℓ_const = n * np.log(2 * np.pi)
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
        dℓ = -(y.T @ S_inv @ y + np.log(det(S)) + self.ℓ_const) / 2  # log-likelihood for the current observation
        # dℓ = np.log(mvn(mean=np.zeros(self.n), cov=S).pdf(y))  # inv(S) is already calculated
        return dℓ


def apply_Kalman_filter_on_trial(x, t, structure, σ_R, permutation=None, implementation='sichao'):
    n, n_z = 3, 8 if sim.whiten else 6
    if permutation:
        permutation_full = np.arange(n_z)
        permutation_full[:n] = permutation
        permutation_full[-n:] = permutation_full[-n:][permutation]
        x = list(np.array(x)[:, permutation_full])
    x0 = x[0] if sim.whiten else x[0][np.r_[:n, -n:0]]
    ℓ = 0
    if implementation == 'johannes':
        f = PhiKalmanFilterPermutation(structure.L, structure.τ_ω, σ_R, whitespace=sim.whiten)
        f.init_filter(x0)
        for i in range(1, len(t)):
            f.propagate_to_time(t[i])
            # print(f.propagated_variables["mu_hat"][:n], x[i][:n])
            f.integrate_observation(x[i])
            ℓ += np.log(f.observation_likelihood(x[i]))
    elif implementation == 'sichao':
        f = MotionStructureKalmanFilter(structure, x0, σ_R)
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            f.predict(dt)
            # print(f.H @ f.z, x[i][:n])
            ℓ += f.update(x[i][:n])
    elif implementation == 'filterpy':
        f_ = MotionStructureKalmanFilter(structure, x0, σ_R)
        f = KalmanFilter(n_z, n)
        f.x = f_.z
        f.P = f_.P
        f.H = f_.H
        f.R = f_.R
        F = f_.F
        Q = f_.Q
        I = f_.I
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            f.predict(F=I + F * dt, Q=Q * dt)
            f.x[:n] %= 2 * np.pi
            # print(f.H @ f.x, x[i][:n])
            f.update(x[i][:n])
            f.x[:n] %= 2 * np.pi
            ℓ += f.log_likelihood
    return ℓ


def apply_filters_on_trial(x, t, σ_R, glo_SDH=2/3):
    return [
        apply_Kalman_filter_on_trial(x, t, MotionStructure(0, 2), σ_R),
        apply_Kalman_filter_on_trial(x, t, MotionStructure(1, 1/4), σ_R),
        apply_Kalman_filter_on_trial(x, t, MotionStructure(0, 1/4), σ_R),  # , permutation=[0, 1, 2]),
        apply_Kalman_filter_on_trial(x, t, MotionStructure(0, 1/4), σ_R, permutation=[1, 2, 0]),
        apply_Kalman_filter_on_trial(x, t, MotionStructure(0, 1/4), σ_R, permutation=[2, 0, 1]),
        apply_Kalman_filter_on_trial(x, t, MotionStructure(glo_SDH, 1/4), σ_R),  # , permutation=[0, 1, 2]),
        apply_Kalman_filter_on_trial(x, t, MotionStructure(glo_SDH, 1/4), σ_R, permutation=[1, 2, 0]),
        apply_Kalman_filter_on_trial(x, t, MotionStructure(glo_SDH, 1/4), σ_R, permutation=[2, 0, 1]),
    ]


def apply_Kalman_filter_on_experiment(file, σ_R, glo_SDH=2/3):
    data = load_data(file)
    filename = file[:-4]
    print(filename)
    # logger = Logger(filename + '.llv')
    with open('../data/pilot.sim.csv', 'w') as csvfile:
        keys = ['IND', 'GLO', 'CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201']
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for trial in data:
            x, t = trial['phi'], trial['t']
            vals = apply_filters_on_trial(x, t, σ_R, glo_SDH)
            # logger.log({keys[i]: vals[i] for i in range(len(keys))})
            # logger.dump()
            writer.writerow({keys[i]: vals[i] for i in range(len(keys))})
    # logger.close()


if __name__ == '__main__':
    file = '../data/sichao/pilot_sichao.dat'
    s = time()
    # for σ_R in []
    apply_Kalman_filter_on_experiment(file, σ_R=1, glo_SDH=0.75)
    print(time() - s)

