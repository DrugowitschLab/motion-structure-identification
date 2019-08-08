from config import fps, SimulationConfig as sim
from datetime import datetime
import numpy as np
__authors__ = "Johannes Bill & Sichao Yang"
__contact__ = "sichao@cs.wisc.edu"
__date__ = datetime(2019, 6, 10)


class StructuredMotionSimulation(object):
    def __init__(self, structure, seed=None, φ_0=None):
        """
        Simulate motion under the given structure.
        :param structure: the motion structure to be simulated
        :type structure: stimuli.motion_structure.MotionStructure
        :param seed: seed for the random state generator
        :type None | int
        :param φ_0: initial locations of the dots
        :type None | np.
        :param whiten: True: Wiener processes as white noises on m velocities of the motion sources;
                       False: L transforms Wiener processes into noises on n velocities of the dots.
        :type whiten: bool
        """
        n, m, n_v = structure.n, structure.m, structure.n_v()  # n locations, m velocities
        # # #  Create angular matrices  # # #
        self.F_φ = structure.build_F_φ()
        self.D_φ = structure.build_D_φ()
        # # #  Create radial matrices  # # #
        self.F_r = structure.build_F_r()
        self.b_r = structure.build_b_r()
        self.D_r = structure.build_D_r()
        # # #  Dynamics  # # #
        self.rng = np.random.RandomState(seed=seed)
        dW = lambda sz: self.rng.normal(loc=0., scale=np.sqrt(sim.dt), size=sz)
        self.dφ = lambda x, F, D: (F @ x) * sim.dt + D @ dW(n + m)
        self.dr = lambda x, F, D, b: (F @ x + b) * sim.dt + D @ dW(n + n)

        # # # Sample the initial states from stationary distribution # # #
        self.φ = np.zeros(n + n_v)
        self.r = np.zeros(n + n)
        self.φ[n:] = self.rng.multivariate_normal(mean=np.zeros(n_v), cov=structure.τ_ω / 2 * np.eye(n_v))
        if φ_0:
            assert len(φ_0) == n, "Error: len(φ_0) != dot number!"
            self.φ[:n] = φ_0
        else:
            self.φ[:n] = 2 * np.pi * self.rng.rand(n)
        self.r[n:] = self.rng.normal(loc=0., scale=np.sqrt(sim.τ_v / 2) * sim.σ_r, size=n)  # v follows OU process
        self.r[:n] = sim.μ_r + self.r[:n] * sim.τ_r  # approximation: integrated OU process under exponential decay
        self.n = n

        # # #  simulation time  # # #
        self.dt = 1. / sim.dt / fps  # number of integration steps per frame
        assert np.isclose(self.dt, round(self.dt))  # should be integer valued
        self.dt = round(self.dt)

    # # #  Euler integration # # #
    def advance(self):
        for tn in range(self.dt):
            self.φ += self.dφ(x=self.φ, F=self.F_φ, D=self.D_φ)
            self.φ[:self.n] = self.φ[:self.n] % (2 * np.pi)    # circular motion --> wrap locations to [0, 2*pi)
            self.r += self.dr(x=self.r, F=self.F_r, D=self.D_r, b=self.b_r)
            self.r[:self.n] = np.maximum(0., self.r[:self.n])  # No negative radii allowed
        return self.φ, self.r
