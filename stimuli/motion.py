from config import fps, dev, SimulationConfig, DisplayConfig, ExperimentConfig
from utils.time import Timer
from utils.lock import Lock
from utils.device_input import Devices
from utils.data import SimLogger
import numpy as np
import pylab as pl
from datetime import datetime
__authors__ = "Johannes Bill & Sichao Yang"
__contact__ = "sichao@cs.wisc.edu"
__date__ = datetime(2019, 6, 10)


class Motion:
    from matplotlib.backend_bases import MouseButton
    trigger_keys = [' ', MouseButton.LEFT]

    def __init__(self, ax, n, is_dev=False):
        self.is_dev = is_dev
        self.n = n
        self.frame = ExperimentConfig.duration + 1
        self.frame_times = []  # wall clock times of rendered frames
        self.timer = Timer()  # wall clock time
        self.logger = SimLogger()
        self.sim_lock = Lock()  # A "lock" for security
        self.ax = ax
        self.sim, self.plotted_dots, self.plotted_labels, self.plotted_text = [None] * 4
        self.onstart, self.onstop = lambda: None, lambda data: None
        self.draw()

    def draw(self):
        """ Initializes plot """
        x, y = np.zeros(self.n), np.ones(self.n)  # TODO: n
        # # #  Plot the dots  # # #
        self.plotted_dots = self.ax.scatter(
            x, y, animated=False,
            s=DisplayConfig.disc_radius ** 2,
            c=DisplayConfig.disc_color,
            marker='o',
            cmap=pl.cm.Paired,  # https://matplotlib.org/examples/color/colormaps_reference.html
            vmin=0., vmax=1.,
            linewidths=0.,
            zorder=2)
        self.plotted_dots.set_visible(True)  # Initially dots are invisible

        # # #  Plot the labels  # # #
        label_kwargs = dict(fontsize=DisplayConfig.label_fontsize,
                            color=DisplayConfig.label_color,
                            weight='bold', ha='center', va='center',
                            visible=DisplayConfig.label_visible)
        self.plotted_labels = [self.ax.text(xn, yn, str(n + 1), **label_kwargs) for n, (xn, yn) in enumerate(zip(x, y))]
        for label in self.plotted_labels:
            label.set_visible(DisplayConfig.label_visible)

        # # #  Plot the text instructions  # # #
        self.plotted_text = self.ax.text(0, 0, '', weight='bold', size='14', ha='center')

        # # #  Axes range and decoration  # # #
        self.ax.set_thetagrids(np.arange(0, 360, 45))
        self.ax.set_rmax(DisplayConfig.axes_radius)
        self.ax.set_xticks([])
        self.ax.grid(DisplayConfig.show_grid)
        self.ax.spines['polar'].set_visible(False)
        self.ax.set_yticks(np.array([0, 1.0, ]) * SimulationConfig.μ_r)
        self.ax.set_yticklabels([])

    def reset(self, preset, seed=None, onstart=lambda: None, onstop=lambda data: None, color_permutation=[0, 1, 2]):
        """
        :param preset: {
            B:
            lam:
            tau_vphi:
        }
        :param seed:        seed for the numpy random number generator
        :param onstart:
        :param onstop:
        :param color_permutation:
        :type  color_permutation: list
        """
        self.plotted_dots.set_color(self.plotted_dots.to_rgba(DisplayConfig.disc_color[color_permutation]))
        self.onstart = onstart
        self.onstop = onstop
        self.plotted_text.set_text('Click <left mouse button> or \npress <space> to continue.')
        for key in self.trigger_keys:
            Devices.enable(key, self.start)
        self.frame = ExperimentConfig.duration + 1
        self.frame_times = []
        self.timer.reset()
        from stimuli.simulation import StructuredMotionSimulation as Sim
        self.sim = Sim(preset, seed=seed)
        self.logger.reset(self.sim.φ.copy(), self.sim.r.copy())
        self.advance(0)

    def advance(self, t):
        self.sim_lock.lock("Error: Plotting update called before sim was ready. Too high fps?")
        # # #  Update the figure with latest data  # # #
        x, y = self.logger.get(self.n)
        self.plotted_dots.set_offsets(np.vstack([x, y]).T)
        for n, (xn, yn) in enumerate(zip(x, y)):
            self.plotted_labels[n].set_position((xn, yn))
        # # #  Integrate the stimulus until the next frame  # # #
        self.logger.log(*self.sim.advance(t))  # See class StructuredMotionStimulus in functions.py for dynamics
        self.sim_lock.unlock()

    def update(self):
        if ExperimentConfig.delay <= self.frame <= ExperimentConfig.duration:
            t_timer = self.timer.get_seconds()
            self.frame_times.append(self.timer.get_seconds())  # Store the time of frame drawing
            f = self.frame - ExperimentConfig.delay
            if f % int(fps) == 0:
                print(f"   > Wall-clock time: {t_timer:7.3f}s, "
                      f"simulation time: {f / fps:7.3f}s, "
                      f"frame number: {f:5d}")
            self.advance(t_timer)
            if self.frame == ExperimentConfig.duration:
                self.finish()
        self.frame += 1
        # # #  Return the list of variable figure elements (required for blitting)  # # #
        return [self.plotted_dots] + self.plotted_labels + [self.plotted_text]

    def start(self, x=None, y=None):
        print('@start')
        self.plotted_text.set_text('')
        for key in self.trigger_keys:
            Devices.disable(key)
        self.frame = 0
        self.onstart()

    def finish(self):
        dt = np.array(self.frame_times[1:]) - np.array(self.frame_times[:-1])
        print(f" > Avg frame interval was {dt.mean():.4f}s "
              f"with std deviation ±{dt.std():.4f}s "
              f"(target was {1 / fps:.4f}s).")
        self.onstop(self.logger.data)
