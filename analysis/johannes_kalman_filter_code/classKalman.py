import numpy as np
from scipy.stats import multivariate_normal as multivariate_normal_pdf

# Replace slow scipy permutation by faster own one
#from sympy.combinatorics import Permutation
from analysis.johannes_kalman_filter_code.classPermutation import Permutation
Permutation.secure = False               # Perform type checks on cost of execution speed
Permutation.print_cyclic = True          # If True, uses the slower sympy implementation for printing

pi = np.pi
numerical_infty = 1.e10                   # covariance values to mimic infinity (for velocity observations)

# # # Angular matrices # # #
# Drift normal
def build_F_angular(tau_vphi, L):
    N = L.shape[0]
    F = np.zeros((2*N,2*N))
    zero = np.zeros((N,N))
    oneN = np.eye(N)
    F[:] = np.vstack([ np.hstack([zero, oneN]),
                    np.hstack([zero, -oneN/tau_vphi])
                    ])
    return F

# Drift whitened
def build_F_angular_white(tau_vphi, L):
    N, M = L.shape
    F = np.zeros((N+M,N+M))
    zeroNN = np.zeros((N,N))
    zeroMN = np.zeros((M,N))
    zeroNM = np.zeros((N,M))
    F[:] = np.vstack([ np.hstack([zeroNN,      L         ]),
                       np.hstack([zeroMN, -np.diag(1/tau_vphi)])
                     ])
    return F

# Diffusion normal
def build_D_angular(L):
    N, M = L.shape
    D = np.zeros((2*N,N+M))
    zeroNN = np.zeros((N,N))
    zeroMN = np.zeros((M,N))
    zeroNM = np.zeros((N,M))
    oneM = np.eye(M)
    D[:] = np.vstack([ np.hstack([zeroNN, zeroNM]),
                    np.hstack([zeroNN, L   ]),
                    ])
    return D

# Diffusion whitened
def build_D_angular_white(L):
    N,M = L.shape
    D = np.zeros((N+M,N+M))
    zeroNN = np.zeros((N,N))
    zeroMN = np.zeros((M,N))
    zeroNM = np.zeros((N,M))
    oneM = np.eye(M)
    D[:] = np.vstack([ np.hstack([zeroNN, zeroNM]),
                       np.hstack([zeroMN, oneM   ]),
                     ])
    return D



# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #   K A L M A N   F I L T E R   # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PhiKalmanFilterPermutation(object):
    def __init__(self, L, tau_vphi, sigma_obs_phi, init_certain=False, gam=None, whitespace=False):
        if whitespace is True:
            if isinstance(tau_vphi, float):
                tau_vphi = tau_vphi * np.ones(L.shape[1])
            assert isinstance(tau_vphi, np.ndarray), " > ERROR: For whitespace, tau_vphi must be ndarray or float."
        elif whitespace is False:
            assert isinstance(tau_vphi, float), " > ERROR: Without whitespace, tau_vphi must be float."
        else:
            raise Exception(" > ERROR: whitened must be True or False")
        self.whitespace = whitespace
        self.N, self.M = L.shape
        self.numVelo = self.M if whitespace else self.N
        self.L = L
        self.tau_vphi = tau_vphi
        self.sigma_obs_phi = sigma_obs_phi
        self.init_certain = init_certain
        if gam is None:
            gam = Permutation(size=self.N)      # Identity
        self.set_permutation(gam)
        # current state estimates
        self.mu = None
        self.Sig = None
        self.t_last = None
        self._is_propagated = False                             # measures where we are in propagation/integration
        # Helper matrices for Kalman updates
        if whitespace is True:
            self.F_base = build_F_angular_white(self.tau_vphi, self.L)    # without inter-observation interval
            D_base = build_D_angular_white(self.L)
        elif whitespace is False:
            self.F_base = build_F_angular(self.tau_vphi, self.L)    # without inter-observation interval
            D_base = build_D_angular(self.L)
        self.Q_base = D_base @ D_base.T                         # without inter-observation interval
        self.H = np.zeros((2*self.N, self.N+self.numVelo))
        self.H[:self.N,:self.N] = np.eye(self.N)                # observation matrix
        if whitespace is True:
            self.H[self.N:,self.N:] = L
        elif whitespace is False:
            self.H[self.N:,self.N:] = np.eye(self.N)
        self.R = np.zeros((2*self.N, 2*self.N))
        self.R[:self.N,:self.N] = self.sigma_obs_phi**2 * np.eye(self.N)    # assumed observation noise in phi (Has to be extended if occlusions could occur!)
        self.R[self.N:,self.N:] = numerical_infty * np.eye(self.N)          # assumed observation noise in vphi

    def set_permutation(self, gam):
        assert isinstance(gam, Permutation), " > ERROR: Permutations must be type <Permutation> from sympy.combinatorics!"
        self.gam = gam
        self.gam_full = self.extend_permutation_to_velocities(gam)

    def extend_permutation_to_velocities(self, gam):
        N = self.N
        return Permutation(list(gam(np.arange(self.N))) + list(gam(np.arange(self.N, 2*self.N))))

    def init_filter(self, s):
        """Init with true state"""
        latentDim = self.N+self.numVelo     # 2N if not whitened; N+M if whitened
        assert len(s) == latentDim, "ERROR: Init state s does not match latentDim!"
        self.mu = s
        self.Sig = np.zeros((latentDim, latentDim))
        self.Sig[:self.N,:self.N] = self.sigma_obs_phi**2 * np.eye(self.N)
        if self.whitespace is True:
            self.Sig[self.N:,self.N:] = np.diag(self.tau_vphi) / 2.
        elif self.whitespace is False:
            self.Sig[self.N:,self.N:] = (self.tau_vphi / 2.) * self.L @ self.L.T
        if self.init_certain:
            self.Sig /= 1000.
        # apply permutation
        #self.mu = self.Gam @ self.mu
        #self.Sig = self.Gam @ self.Sig @ self.Gam.T
        # time is zero
        self.t_last = 0.
        self._is_propagated = False
        self.archive = dict(t=[self.t_last], mu=[self.mu], Sig=[self.Sig], gam=[self.gam.perm.copy()])

    def propagate_and_integrate_observation(self, x, t):
        """For the non-particle filter version, we can go straight forward."""
        self.propagate_to_time(t)
        self.integrate_observation(x)
        return self.mu, self.Sig

    def propagate_to_time(self, t):
        assert self.t_last is not None, " > ERROR: Initialize filter state first!"
        assert self._is_propagated is False, " > ERROR: Trying to propagate already propagated filter." # This could (in principle) be allowed.
        # observation time update
        dt = t - self.t_last
        self.t_last = t
        assert dt >= 0, " > ERROR: Negative interval since last observation!"
        # Let's hack in the Kalman update equations...
        N = self.N
        F = np.eye(N+self.numVelo) + dt * self.F_base
        Q = dt * self.Q_base
        H = self.H
        R = self.R
        # Prior moments
        mu_hat = F @ self.mu
        mu_hat[:N] %= (2*pi)

        Sig_hat = F @ self.Sig @ F.T + Q
        # Kalman gain
        S_res = R + H @ Sig_hat @ H.T
        K = Sig_hat @ H.T @ np.linalg.inv(S_res)
        self.propagated_variables = dict(   # All we need for updates:
            t = t,                          # time
            mu_hat = mu_hat,                # mean
            Sig_hat = Sig_hat,              # cov
            K = K                           # Kalman gain
            )
        self._is_propagated = True

    def integrate_observation(self, x):
        assert self._is_propagated is True, " > ERROR: Propagate filter before integrating new observation!"
        assert self.propagated_variables["t"] == self.t_last, " > ERROR: Inconsistent internal time information."
        # retrieve pre-computed values and required variables
        mu_hat = self.propagated_variables["mu_hat"]
        Sig_hat = self.propagated_variables["Sig_hat"]
        K = self.propagated_variables["K"]
        N = self.N
        R = self.R
        H = self.H
        # calc residual
        mu_res = self.calculate_residual_mean(x)
        # integrate observation: means
        self.mu = mu_hat + K @ mu_res
        self.mu[:N] %= 2*pi
        # integrate observation: covariance
        M = ( np.eye(N+self.numVelo) - K @ H )
        self.Sig = M @ Sig_hat @ M.T + K @ R @ K.T          # <<-- WARNING: IF data points could have varying precision, R must be permutated/adapted.
        # switch state
        self._is_propagated = False
        # store results
        self.archive["t"].append(self.propagated_variables["t"])
        self.archive["mu"].append(self.mu)
        self.archive["Sig"].append(self.Sig)
        self.archive["gam"].append(self.gam.perm.copy())

    def calculate_residual_mean(self, x, gam=None):
        """mu_hat under current propagation assuming permutation gam (defaults to self.gam)."""
        assert self.propagated_variables["t"] == self.t_last, " > ERROR: Inconsistent internal time information."
        mu_hat = self.propagated_variables["mu_hat"]
        if gam is None:
            gam_full = self.gam_full
        else:
            gam_full = self.extend_permutation_to_velocities(gam)
        N = self.N
        H = self.H
        mu_res = (gam_full(x) - H @ mu_hat)             # <<--- THIS IS WHERE THE PERMUTATION ENTERS!
        mu_res[:N] += pi
        mu_res[:N] %= (2*pi)
        mu_res[:N] -= pi
        return mu_res

    def observation_likelihood(self, x, perm=None, differential=True, location_only=True):
        """Calculate data likelihood under a hypothetical data assignment.
             differential: # whether perm is relative to current internal gam, or absolute.
               differential ==  True: used permutation is (self.gam * perm)
               differential == False: used permutation is perm
             If perm==None, either case will use the current internal gam.
             If location_only: Do not evaluate Gaussian on velocities (e.g. due to inf-variance)
        """
        if perm is None:
            gam = None
        else:
            if differential is True:
                gam = self.gam * perm
            elif differential is False:
                gam = perm
            else:
                raise Exception(" > ERROR: 'differential' must be True or False.")
        # Calc residual under permutation
        mu_res = self.calculate_residual_mean(x, gam=gam)
        # calculate pdf (This could be done once, and then be reused for efficiency)
        N = self.N
        H = self.H
        Sig_tot = H @ self.propagated_variables["Sig_hat"] @ H.T + self.R              # IF data points could have varying precision, R must be adapted.
        if location_only is True:
            pdf = multivariate_normal_pdf(mean=np.zeros(N), cov=Sig_tot[:N,:N]).pdf
            mu_res = mu_res[:N]
        else:
            pdf = multivariate_normal_pdf(mean=np.zeros(2*N), cov=Sig_tot).pdf
        # Here comes the value
        px = pdf(mu_res)
        return px

    def clone(self, perm=None, differential=True):
        """Clone the Kalman filter (incl history), but with a changed permutation.
             See 'observation_likelihood' for definition of 'perm' and 'differential'.
        """
        if perm is None:
            gam = self.gam
        else:
            if differential is True:
                gam = self.gam * perm
            elif differential is False:
                gam = perm
            else:
                raise Exception(" > ERROR: 'differential' must be True or False.")
        from copy import deepcopy
        kwargs = dict(L = self.L,
                  tau_vphi = self.tau_vphi,
                  sigma_obs_phi = self.sigma_obs_phi,
                  gam = gam,
                  whitespace=self.whitespace
                )
        c = type(self)(**kwargs)
        # copy states
        c.mu = np.copy(self.mu)
        c.Sig = np.copy(self.Sig)
        c.t_last = self.t_last
        c._is_propagated = self._is_propagated
        c.archive = deepcopy(self.archive)
        c.propagated_variables = deepcopy(self.propagated_variables)
        return c















