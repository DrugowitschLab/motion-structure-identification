from config import SimulationConfig as sim
from utils.data import load_data
from stimuli.motion_structure import MotionStructure
import numpy as np
from numpy.linalg import inv, det

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
        # from scipy.stats import multivariate_normal as mvn
        # dL = np.log(mvn(mean=np.zeros(self.n), cov=S).pdf(y))  # inv(S) is already calculated
        return dL


def apply_Kalman_filter(x, t, structure, σ_R=0., reture_sum=True, implementation='sichao'):
    n = 3
    z0 = x[0] if sim.whiten else x[0][np.r_[:n, -n:0]]
    L = []
    if implementation == 'johannes':
        from analysis.johannes_kalman_filter_code.classKalman import PhiKalmanFilterPermutation
        f = PhiKalmanFilterPermutation(structure.L, structure.τ_ω, σ_R, whitespace=sim.whiten)
        f.init_filter(z0)
        for i in range(1, len(t)):
            f.propagate_to_time(t[i])
            # print(f.propagated_variables["mu_hat"][:n], x[i][:n])
            f.integrate_observation(x[i])
            L.append(np.log(f.observation_likelihood(x[i])))
    elif implementation == 'sichao':
        f = MotionStructureKalmanFilter(structure, z0, σ_R)
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            f.predict(dt)
            # print(f.H @ f.z, x[i][:n])
            L.append(f.update(x[i][:n]))
    if reture_sum:
        return sum(L)
    else:
        return L


def apply_Kalman_filters(x, t, structures, σ_R=0.):
    L = np.array([apply_Kalman_filter(x, t, structures[s], σ_R) for s in structures])
    return L.sum(axis=1)


def apply_Kalman_filter_on_experiment(file, structures, σ_R=0., σ_x=0., repeats=1):
    data = load_data(file)
    filename = file[:-4]
    print(filename)
    # logger = Logger(filename + '.llv')
    with open(f'../data/{filename}_σ={σ_R}.csv', 'w') as csvfile:
        keys = ['IND', 'GLO', 'CLU_012', 'CLU_120', 'CLU_201', 'SDH_012', 'SDH_120', 'SDH_201', 'ground_truth', 'choice', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()

        for trial in data:
            for _ in range(repeats):
                x, t = trial['φ'][3:], trial['t'][3:]
                x = np.random.normal(loc=x, scale=σ_x)
                vals = list(apply_filters_on_trial(x, t, σ_R, structures))
                vals.append(trial['ground_truth'])
                vals.append(trial['choice'])
                vals.append(trial['confidence'])
                # logger.log({keys[i]: vals[i] for i in range(len(keys))})
                # logger.dump()
                writer.writerow({keys[i]: vals[i] for i in range(len(keys))})
    # logger.close()


if __name__ == '__main__':
    np.set_printoptions(linewidth=150)
    # file = '../data/sichao/pilot_sichao.dat'
    file = '../data/exp1/sichao_0806/sichao_0806.dat'
    t_s = time()
    # for σ_R in []
    glo = 3/4
    λ_I = 1/4
    apply_Kalman_filter_on_experiment(file, σ_R=0, σ_x=0, repeats=1, structures={
        'IND': MotionStructure(1, 2),
        'GLO': MotionStructure(1, λ_I),
        'CLU_012': MotionStructure(0, λ_I, permutation=[0, 1, 2]),
        'CLU_120': MotionStructure(0, λ_I, permutation=[1, 2, 0]),
        'CLU_201': MotionStructure(0, λ_I, permutation=[2, 0, 1]),
        'SDH_012': MotionStructure(glo, λ_I, permutation=[0, 1, 2]),
        'SDH_120': MotionStructure(glo, λ_I, permutation=[1, 2, 0]),
        'SDH_201': MotionStructure(glo, λ_I, permutation=[2, 0, 1]),
    })
    print(time() - t_s)

