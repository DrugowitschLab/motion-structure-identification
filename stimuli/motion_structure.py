from config import SimulationConfig as sim
import numpy as np


class MotionStructure:
    def __init__(self, glo, λ_I, volatility_factor=4/3):
        """
        Build motion structure matrix L specified by glo and λ_I
        :param glo: λ_G**2 / (λ_G**2 + λ_C**2)
        :type glo: float
        :param λ_I: the strength of the independent motion source
        :type λ_I: float
        :param volatility_factor: a coefficient that makes stimulus changes faster without affecting the covariance
        :type volatility_factor: float
        """
        B = np.array([
            [1, 1, 1, 0, 0, ],  # MOTION STRUCTURE COMPONENT MATRIX
            [1, 1, 0, 1, 0, ],  # each row describes one dot (N dots)
            [1, 0, 0, 0, 1, ],  # each column is a motion source (M sources)
        ], dtype=np.float64)
        λ_T = 2.                                                # total
        assert 0 <= λ_I <= λ_T, 'Make sure 0 <= λ_I <= λ_T'
        λ_G = np.sqrt(glo) * np.sqrt(max(0., λ_T**2 - λ_I**2))  # global
        λ_C = np.sqrt(max(0., λ_T**2 - λ_G**2 - λ_I**2))        # cluster
        λ_M = np.sqrt(max(0., λ_T**2 - λ_G**2))                 # maverick
        λ = np.sqrt(volatility_factor) * np.array([λ_G, λ_C, λ_I, λ_I, λ_M])
        self.L = B @ np.diag(λ)
        self.n, self.m = self.L.shape
        self.τ_ω = 2. / volatility_factor

    def build_F_φ(self):
        """ Angular drift
        [ 0   I
          0 I/τ_ω ]
        = whiten =>
        [ 0   L
          0 I/τ_ω ]
        """
        F_φ = np.zeros((self.n + self.n_v(), self.n + self.n_v()))
        F_φ[:self.n, self.n:] = self.L if sim.whiten else np.eye(self.n)
        F_φ[self.n:, self.n:] = -np.eye(self.n_v()) / self.τ_ω  # Velocities follow an OU process
        return F_φ

    def build_D_φ(self):
        """ Angular diffusion
        [ 0 0
          0 L ]
        = whiten =>
        [ 0 0
          0 I ]
        """
        D_φ = np.zeros((self.n + self.n_v(), self.n + self.m))
        D_φ[self.n:, self.n:] = np.eye(self.m) if sim.whiten else self.L
        return D_φ

    def build_F_r(self):
        """ Radial drift
        [ -I/τ_r    I
            0    -I/τ_v ]
        """
        D_r = np.zeros((self.n + self.n, self.n + self.n))
        I_n = np.eye(self.n)
        D_r[:self.n, :self.n] = -I_n / sim.τ_r  # Active location decay for stable orbits
        D_r[self.n:, self.n:] = -I_n / sim.τ_v  # Velocities follow an OU process
        D_r[:self.n, self.n:] = I_n
        return D_r

    def build_D_r(self):
        """ Radial diffusion
        [ 0,   0
          0, σ_r*I ]
        """
        D = np.zeros((self.n + self.n, self.n + self.n))
        D[self.n:, self.n:] = sim.σ_r * np.eye(self.n)  # Velocities follow an OU process
        return D

    def build_b_r(self):
        """ Radial bias
        [ (μ_r/τ_r)_n 0_n ]
        """
        b = np.zeros(self.n + self.n)
        b[:self.n] = sim.μ_r / sim.τ_r  # only locations, not velocities
        return b

    def n_v(self):
        return self.m if sim.whiten else self.n

    def __str__(self):
        """  Print a preview of the motion structure """
        Σ = self.τ_ω / 2 * self.L @ self.L.T
        print(Σ)
        return (" > Motion structure matrix L:\n"
                f"{asciiL(self.L, 3)}\n"
                " > Velocity covariance matrix:\n"
                f"{asciiL(Σ, 3)}")


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
