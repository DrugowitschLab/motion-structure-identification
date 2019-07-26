import numpy as np

fps = 50.
dev = False


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


class ExperimentConfig:
    seconds2frames = lambda seconds: int(round(seconds * fps))
    delay = seconds2frames(0.5)
    duration = seconds2frames(4.5)


class SimulationConfig:
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

# config = dict(
#     # # #  Parameters for the simulation of dot motion  # # #
#     sim=dict(
#         dt=0.001,  # Integration time step (Euler integrator)
#         # # #  The next four values control stochastic radii (optional feature)  # # #
#         tau_vr=0.001,  # OU time constant for radial velocities
#         tau_r=0.001,  # Time constant for stabilizing orbits' radii
#         radial_sigma=0.001,  # Diffusion of radii
#         radial_mean=1.,  # np.array([1.]*61 + [0.8]*0 + [0.9]*0)                        # Avg radius of dot orbits
#     ),
#     # # #  Parameters for the plotting of dots on screen or video  # # #
#     display=dict(
#         fps=fps,  # Target frames per second (for live preview and video)
#         backend_interactive="Qt5Agg",  # matplotlib backend for preview
#         # (Use "Qt4Agg", "macosx" or "Qt5Agg" if you encounter wrong frame timing with "TkAgg")
#         backend_noninteractive="Agg",  # matplotlib backend for video rendering
#         axes_radius=2.0,  # Range of the display
#         monitor_dpi=109.,  # Monitor resolution (dots per inch), required for correct figure and font size
#         figsize=(6 * 16 / 9, 6),  # Figure size (width, height) in inches
#         bg_color="w",  # Background color
#         disc_color=(np.array([1, 3, 5], dtype='float') + 0.5) / 12,  # Dot color
#         disc_radius=np.array([20] * 3),  # Dot size (same units as fontsize)
#         show_labels=False,  # Identify the dots with numbers?
#         label_color="w",  # Label color
#         label_fontsize=8,  # Label fontsize
#         show_grid=True,  # Show a grid in polar coordinates?
#     ),
#     # trial phases
#     experiment=dict(
#         pre_present=s2f(0.5),
#         present=s2f(4.),
#         post_choice=s2f(1.),  # deprecated
#     ),
#     # # #  Parameters for video rendering   # # #
#     video=dict(
#         dpi=150,  # Resolution of rendered frames for video (high values: high quality, slow rendering)
#         bitrate=1024,  # Target video size per second (our simple video may not use all of it)
#         renderer='ffmpeg',
#         # External library for rendering (see: https://matplotlib.org/api/animation_api.html#writer-classes)
#         codec='libx264',  # Video codec used for rendering (name may be renderer dependent)
#     ),
#     DEV=False,  # Developer mode (show ticks, additional output,...)
# )
