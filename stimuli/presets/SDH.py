r"""
    |
    o
   / \
  o   \
 / \   \
o   o   o
"""
import numpy as np

human_readable_dsl = None       # None = DRYRUN

# MOTION STRUCTURE COMPONENT MATRIX
# each row describes one dot (N dots)
# each column is a motion source (M sources)
B = np.array([
    [1, 1, 1,0,0,],
    [1, 1, 0,1,0,],
    [1, 0, 0,0,1,],
    ], dtype=np.float64)

volatility_factor = 4/3             # Makes the stimulus change more rapidly without affecting the covariance matrix
speed_factor = 1.0
glo = 2/3

# MOTION STRENGTHS
# strength of the components (columns of B)
lam_T = 2.
lam_I = 1/4
lam_G = np.sqrt(glo) * np.sqrt(lam_T**2 - lam_I**2)
lam_C = np.sqrt( max(0., lam_T**2 - lam_G**2 - lam_I**2) )
lam_M = np.sqrt( max(0., lam_T**2 - lam_G**2) )
print([lam_G, lam_C] + [lam_I]*2 + [lam_M]*1)

lam = np.sqrt(volatility_factor) * np.array([lam_G, lam_C] + [lam_I]*2 + [lam_M]*1)
lam *= speed_factor

# THE FULL MOTION STRUCTURE MATRIX WILL BE: L = B @ diag(lam)
# TIME CONSTANT for significant changes in angular velocities (in seconds)
tau_vphi = 2. / volatility_factor

# INITIAL POSITIONS
N = B.shape[0]
#phi0 = np.linspace(0, 2*np.pi, N+1)[:-1]         # Equidistant full circle
#phi0 = np.linspace(0, np.pi, N+1)[:-1]          # Equidistant half circle

# Target dots to fade out
targets = [2]
disc_color = ( np.array([1,3,5], dtype='float') + 0.5 ) / 12
