import pylab as pl
import numpy as np
from pathlib import Path

root = Path(__file__).parent
fps = 50.
verbose = True


def seconds2frames(seconds: float):
    return int(round(seconds * fps))


class DisplayConfig:
    fps = fps  # Target frames per second (for live preview and video)
    backend_interactive = "Qt5Agg"  # matplotlib backend for preview
    # (Use "Qt4Agg", "macosx" or "Qt5Agg" if you encounter wrong frame timing with "TkAgg")
    backend_noninteractive = "Agg"  # matplotlib backend for video rendering
    axes_radius = 2.0  # Range of the display
    monitor_dpi = 109.  # Monitor resolution (dots per inch), required for correct figure and font size
    figsize = (6 * 16 / 9, 6)  # Figure size (width, height) in inches
    bg_color = "w"  # Background color
    disc_color = (np.array([1, 3, 5], dtype='float') + 0.5) / 12  # Dot color
    disc_radius = np.array([20] * 3)  # Dot size (same units as fontsize)
    label_visible = False  # Identify the dots with numbers?
    label_color = "w"  # Label color
    label_fontsize = 8  # Label fontsize
    show_grid = True  # Show a grid in polar coordinates?
    dots_kwargs = dict(
        animated=False,
        s=disc_radius ** 2,
        c=disc_color,
        marker='o',
        cmap=pl.cm.Paired,  # https://matplotlib.org/examples/color/colormaps_reference.html
        vmin=0., vmax=1.,
        linewidths=0.,
        zorder=2
    )
    label_kwargs = dict(
        fontsize=label_fontsize,
        color=label_color,
        weight='bold', ha='center', va='center',
        visible=label_visible
    )

    @staticmethod
    def config_ax(ax):
        ax.set_thetagrids(np.arange(0, 360, 45))
        ax.set_rmax(DisplayConfig.axes_radius)
        ax.set_xticks([])
        ax.grid(DisplayConfig.show_grid)
        ax.spines['polar'].set_visible(False)
        ax.set_yticks(np.array([0, 1.0, ]) * SimulationConfig.μ_r)
        ax.set_yticklabels([])


class ExperimentConfig:
    n_trials_exp1 = 200
    n_trials_exp2 = 100
    n_dots = 3
    permutations = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]    # 3 permutations of 3 dots, assuming within-cluster symmetry.
    # 3 dots can have 6 permutations, but it is unnecessary to distinguish between the two clustered dots.
    delay = seconds2frames(0.5)
    duration = seconds2frames(4.5)  # 4.5s -> ~25+15 min / 200 trials
    λ_T = 2.
    λ_I = 1 / 4
    glo_H = 3 / 4
    glo_exp2 = [0.00, 0.20, 0.35, 0.55, 0.75]


class SimulationConfig:
    volatility_factor = 4 / 3
    dt = 0.001  # Integration time step (Euler integrator)
    # # #  The next four values control stochastic radii (optional feature)  # # #
    τ_v = 0.001  # OU time constant for radial velocities
    τ_r = 0.001  # Time constant for stabilizing orbits' radii
    σ_r = 0.001  # Diffusion of radii
    μ_r = 1.     # np.array([1.]*61 + [0.8]*0 + [0.9]*0)  # Avg radius of dot orbits
    whiten = True


class VideoConfig:
    dpi = 150  # Resolution of rendered frames for video (high values: high quality, slow rendering)
    bitrate = 1024  # Target video size per second (our simple video may not use all of it)
    renderer = 'ffmpeg'
    # External library for rendering (see: https://matplotlib.org/api/animation_api.html#writer-classes)
    codec = 'libx264'  # Video codec used for rendering (name may be renderer dependent)
