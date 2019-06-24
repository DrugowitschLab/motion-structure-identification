from datetime import datetime
from time import time
import numpy as np
__authors__ = "Johannes Bill & Sichao Yang"
__contact__ = "sichao@cs.wisc.edu"
__date__ = datetime(2019, 6, 10)


class StructuredMotionStimulus(object):
    def __init__(self, config, preset, seed=None, f_dW=None, phi0=None, is_dev=False):
        self.L = preset.B @ np.diag(preset.lam)
        self.N, self.M = self.L.shape                              # N dots, M motion components
        tau_vphi = preset.tau_vphi
        if isinstance(tau_vphi, float):
            tau_vphi = np.array([tau_vphi] * self.M)
        self.tau_vphi = tau_vphi
        self.__dict__.update(config['sim'])
        # self.tau_r = config['sim']['tau_r']
        # self.tau_vr = config['sim']['tau_vr']
        # self.radial_sigma = config['sim']['radial_sigma']
        # self.radial_mean = config['sim']['radial_mean']
        # self.dt = config['sim']['dt']
        self.t_in_trial = 0
        self.fps = config['display']['fps']
        self.f_dW = f_dW
        if phi0 is not None:
            assert len(phi0) == self.N, "Error: If not None, len(phi0) must equal num dots."
        self.phi0 = phi0
        self.rng = np.random.RandomState(seed=seed)
        self.dt_per_frame = 1. / self.fps / self.dt                          # num integration time steps between frames
        assert np.isclose(self.dt_per_frame, int(round(self.dt_per_frame)))  # should be integer valued
        self.dt_per_frame = int(round(self.dt_per_frame))
        # # #  Create angular matrices  # # #
        self.Fphi = SDE.build_F_angular_white(tau_vphi, self.L)
        self.Dphi = SDE.build_D_angular_white(self.L)
        # # #  Create radial matrices  # # #
        self.Fr = SDE.build_F_radial(self.tau_r, self.tau_vr, self.N)
        self.br = SDE.build_bias_radial(self.radial_mean, self.tau_r, self.N)
        self.Dr = SDE.build_D_radial(self.radial_sigma, self.N)
        # # #  HERE are the dynamics  # # #
        self.sqrtdt = np.sqrt(self.dt)
        if f_dW is None:
            self.dWphi = lambda: self.rng.normal(loc=0.0, scale=self.sqrtdt, size=self.N + self.M)
        else:
            self.dWphi = f_dW(self.dt).__next__
        self.dphi = lambda x, F, D: F @ x * self.dt + D @ self.dWphi()
        self.dWr = lambda: self.rng.normal(loc=0.0, scale=self.sqrtdt, size=self.N + self.N)
        self.dr = lambda x, F, b, D: (F @ x + b) * self.dt + D @ self.dWr()
        # # # Initialize states  # # #
        self.Phi = np.zeros(self.N + self.M)
        self.R = np.zeros(self.N + self.N)
        self.reset(is_dev)

    def reset(self, is_dev=False, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        N, M = self.N, self.M
        # # #  phi: draw stationary (uniform) locations  # # #
        self.Phi[:N] = 2 * np.pi * self.rng.rand(N) if self.phi0 is None else self.phi0
        # # #  phi: draw stationary distribution velocities  # # #
        # Sigma = self.tau_vphi * self.L@self.L.T / 2
        if self.f_dW is None:
            Sigma = self.tau_vphi * np.eye(self.M) / 2
            if is_dev:
                print(Sigma)
            vphi = self.rng.multivariate_normal(mean=np.zeros(M), cov=Sigma)
            self.Phi[N:] = vphi
        else:
            vphi = np.zeros(M)
            self.Phi[N:] = vphi
            nSteps = int(round(5 * min(10, np.max(self.tau_vphi)) / self.dt))
            self.advance(n_steps=nSteps)
        # # #  r: draw approx. stationary location and velocity  # # #
        self.R[N:] = self.rng.normal(loc=0., scale=np.sqrt(self.tau_vr/2)*self.radial_sigma, size=N)  # velocities follow OU process
        self.R[:N] = self.radial_mean + self.R[N:] * self.tau_r          # This is an approximation: integrated OU under exp decay.
        self.t_in_trial = 0.

    # # #  Euler integration # # #
    def advance(self, n_steps=None):
        if n_steps is None:
            n_steps = self.dt_per_frame
        # # #  Call above dynamics and add them to current value  # # #
        for tn in range(n_steps):
            self.Phi += self.dphi(x=self.Phi, F=self.Fphi, D=self.Dphi)
            self.Phi[:self.N] = self.Phi[:self.N] % (2 * np.pi)     # We are on a circle --> wrap locations to [0, 2*pi]
            self.R += self.dr(x=self.R, F=self.Fr, b=self.br, D=self.Dr)
            self.R[:self.N] = np.maximum(0., self.R[:self.N])       # No negative radii allowed
        self.t_in_trial += self.dt * n_steps
        return self.t_in_trial, self.Phi.copy(), self.R.copy()

    def __str__(self):
        """  Print a preview of the motion structure """
        return (" > The motion structure matrix L looks as follows:\n"
                f"{asciiL(self.L, 3)}\n"
                " > This leads to the following velocity covariance matrix:\n"
                f"{asciiL(1 / 2. * self.L @ np.diag(self.tau_vphi) @ self.L.T, 3)}")


class SDE:
    @staticmethod
    def build_F_angular_white(tau, L):
        """ Angular drift """
        N, M = L.shape
        return np.block([
            [np.zeros((N, N)), L],
            [np.zeros((M, N)), -np.diag(1 / tau)]
        ])

    @staticmethod
    def build_D_angular_white(L):
        """ Angular diffusion """
        N, M = L.shape
        # d = np.zeros((N + M, N + M))
        # d[N:, N:] = eye(M)
        return np.block([
            [np.zeros((N, N)), np.zeros((N, M))],
            [np.zeros((M, N)), np.eye(M)]
        ])

    # # #  Radial components can be added to make the dot orbits non-overlapping  # # #
    @staticmethod
    def build_F_radial(tau_r, tau_vr, N):
        """ Radial drift """
        eye_n = np.eye(N)
        return np.block([
            [-eye_n / tau_r, eye_n],    # We add an active location decay for stable orbits
            [np.zeros((N, N)), -eye_n / tau_vr]  # Velocities follow an OU process
        ])

    @staticmethod
    def build_D_radial(radial_sigma, N):
        """ Radial diffusion """
        zeros_n = np.zeros((N, N))
        return np.block([
            [zeros_n, zeros_n],
            [zeros_n, radial_sigma * np.eye(N)]  # Velocities follow an OU process
        ])

    # # #  Radii don't decay to zero but radial_mean  # # #
    @staticmethod
    def build_bias_radial(radial_mean, tau_r, N):
        b = np.zeros(2 * N)
        b[:N] = radial_mean / tau_r  # only locations, not velocities
        return b


def asciiL(L, indent=0):
    """ String representation of motion structure matrix """
    indent = " " * indent
    theta = np.array([0.05, 0.500, 0.999, 1.001])
    chars = {
        'zero': " ",
        3: "█",
        2: "▓",
        1: "▒",
        0: "░"
    }
    vmax, vmin = L.max(), L.min()
    char = lambda val: chars['zero'] if val == 0. else chars[((val - vmin) / (vmax - vmin) < theta).argmax()]
    s = indent + "┌" + "─" * (2 * L.shape[1]) + "┐\n"
    for line in L:
        s += indent + "│" + "".join([char(v) * 2 for v in line]) + "│\n"
    s += indent + "└" + "─" * (2 * L.shape[1]) + "┘"
    return s
