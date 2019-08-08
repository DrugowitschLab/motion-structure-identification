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
    first_trial = True
    show_structure = False

    def __init__(self, ax, n_dots):
        self.n_dots = n_dots
        self.frame = ExperimentConfig.duration + 1
        self.timer = Timer()  # wall clock time
        self.logger = SimLogger()
        self.sim_lock = Lock()  # A "lock" for security
        self.sim, self.plotted_dots, self.plotted_labels, self.plotted_text = [None] * 4
        self.onstart, self.onstop, self.onfinish = lambda: None, lambda data: None, lambda: None
        self.draw(ax)

    def draw(self, ax):
        """ Initializes plot """
        x, y = np.zeros(self.n_dots), np.ones(self.n_dots)
        # # #  Plot the dots  # # #
        self.plotted_dots = ax.scatter(x, y, **DisplayConfig.dots_kwargs)
        self.plotted_dots.set_visible(True)  # Initially dots are invisible
        # # #  Plot the labels  # # #
        self.plotted_labels = [ax.text(xn, yn, str(n + 1), **DisplayConfig.label_kwargs)
                               for n, (xn, yn) in enumerate(zip(x, y))]
        # # #  Plot the text instructions  # # #
        self.plotted_text = ax.text(0, 0, '', weight='bold', size='14', ha='center')
        # # #  Axes range and decoration  # # #
        DisplayConfig.config_ax(ax)

    def reset(self, preset, seed=None, color_permutation=[0, 1, 2],
              onstart=lambda: None, onstop=lambda data: None, onfinish=lambda: None):
        self.onstart = onstart
        self.onstop = onstop
        self.onfinish = onfinish
        self.plotted_dots.set_color(self.plotted_dots.to_rgba(DisplayConfig.disc_color[color_permutation]))
        self.timer.reset()
        self.logger.reset()
        from stimuli.simulation import StructuredMotionSimulation as Sim
        self.sim = Sim(preset, seed=seed)
        self.advance()
        if self.first_trial:
            self.wait()
        else:
            self.frame = 0

    def advance(self):
        # # #  Update the figure with latest data  # # #
        x, y = self.sim.φ[:self.n_dots], self.sim.r[:self.n_dots]
        self.plotted_dots.set_offsets(np.vstack([x, y]).T)
        for n, (xn, yn) in enumerate(zip(x, y)):
            self.plotted_labels[n].set_position((xn, yn))
        # # #  Integrate the stimulus until the next frame  # # #
        self.sim.advance()  # See class StructuredMotionStimulus in functions.py for dynamics

    def update(self):
        if ExperimentConfig.delay <= self.frame <= ExperimentConfig.duration:
            self.sim_lock.lock("Error: Plotting update called before sim was ready. Too high fps?")
            t_timer = self.timer.get_seconds()
            self.logger.log(t_timer, self.sim.φ.copy(), self.sim.r.copy())
            f = self.frame - ExperimentConfig.delay
            if f % int(fps) == 0:
                print(f"   > Wall-clock time: {t_timer:7.3f}s, "
                      f"simulation time: {f / fps:7.3f}s, "
                      f"frame number: {f:5d}")
            self.advance()
            if self.frame == ExperimentConfig.delay:
                Devices.enable('s', self.skip)
            if self.frame == ExperimentConfig.duration:
                self.logger.data['n_frame'] = t_timer
                Devices.disable('s')
                self.onstop(self.logger.data)
                self.wait()
            self.sim_lock.unlock()
        self.frame += 1
        # # #  Return the list of variable figure elements (required for blitting)  # # #
        return [self.plotted_dots] + self.plotted_labels + [self.plotted_text]

    def prompt(self, text='Click <left mouse button> or \npress <space> to continue.'):
        self.plotted_text.set_text(text)

    def skip(self):
        self.frame = ExperimentConfig.duration

    def wait(self):
        if self.first_trial:
            self.prompt()
        for key in self.trigger_keys:
            Devices.enable(key, self.start)

    def start(self, x=None, y=None):
        self.plotted_text.set_text('')
        for key in self.trigger_keys:
            Devices.disable(key)
        if self.first_trial:
            self.first_trial = False
            self.frame = 0
        else:
            self.finish()

    def finish(self):
        ts = np.array(self.logger.data['t'])
        dt = ts[1:] - ts[:-1]
        print(f" > Avg frame interval was {dt.mean():.4f}s "
              f"with std deviation ±{dt.std():.4f}s "
              f"(target was {1 / fps:.4f}s).")
        self.onfinish()
