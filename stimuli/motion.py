from utils.time import Timer
from utils.lock import Lock
import numpy as np
import pylab as pl
from datetime import datetime
__authors__ = "Johannes Bill & Sichao Yang"
__contact__ = "sichao@cs.wisc.edu"
__date__ = datetime(2019, 6, 10)


class Motion:
    from config import config
    """ :var config: see config.py """

    def __init__(self, ax, n, is_dev=False):
        self.is_dev = is_dev
        self.n = n
        self.fps = self.config['display']['fps']
        self.show_labels = self.config['display']['show_labels']
        self.t_start = self.config['experiment']['pre_present']
        self.t_stop = self.config['experiment']['present'] + self.t_start
        self.frame = 0  # [0, t_start): still; [t_start, t_stop]: play; (t_stop, choose]: still; (choose, +oo): wait
        self.times = []  # wall clock times of rendered frames
        self.timer = Timer()  # wall clock time
        self.logger = Logger()
        self.sim_lock = Lock()  # A "lock" for security
        self.ax = ax
        self.stimulus, self.plotted_dots, self.plotted_labels, self.plotted_text = [None] * 4
        self.cid = {}
        self.callback = lambda data: None
        self.colors = self.config['display']['disc_color'].copy()
        self.draw()

    def draw(self):
        """ Initializes plot """
        x, y = np.zeros(self.n), np.ones(self.n)  # TODO: n
        # # #  Plot the dots  # # #
        self.plotted_dots = self.ax.scatter(
            x, y, animated=False,
            s=self.config['display']['disc_radius'] ** 2,
            c=self.colors,
            marker='o',
            cmap=pl.cm.Paired,  # https://matplotlib.org/examples/color/colormaps_reference.html
            vmin=0., vmax=1.,
            linewidths=0.,
            zorder=2)
        self.plotted_dots.set_visible(True)  # Initially dots are invisible

        # # #  Plot the labels  # # #
        label_kwargs = dict(fontsize=self.config['display']['label_fontsize'],
                            color=self.config['display']['label_color'],
                            weight='bold', ha='center', va='center',
                            visible=self.show_labels)
        self.plotted_labels = [self.ax.text(xn, yn, str(n + 1), **label_kwargs) for n, (xn, yn) in enumerate(zip(x, y))]
        for label in self.plotted_labels:
            label.set_visible(self.show_labels)

        # # #  Plot the text instructions  # # #
        self.plotted_text = self.ax.text(0, 0, '', weight='bold', size='14', ha='center')

        # # #  Axes range and decoration  # # #
        self.ax.set_thetagrids(np.arange(0, 360, 45))
        self.ax.set_rmax(self.config['display']['axes_radius'])
        self.ax.set_xticks([])
        self.ax.grid(self.config['display']['show_grid'])
        self.ax.spines['polar'].set_visible(False)
        self.ax.set_yticks(np.array([0, 1.0, ]) * self.config['sim']['radial_mean'])
        self.ax.set_yticklabels([])

    def reset(self, preset, seed=None, callback=lambda data: None):
        """
        :param preset: {
            B:
            lam:
            tau_vphi:
        }
        :param seed:        seed for the numpy random number generator
        :param is_dev:      if DEVELOPER mode is on
        """
        np.random.shuffle(self.colors)
        self.plotted_dots.set_color(self.plotted_dots.to_rgba(self.colors))
        self.callback = callback
        self.plotted_text.set_text('')
        self.frame = 0
        self.times = []
        self.timer.reset()
        from stimuli.stimulus import StructuredMotionStimulus as Stimulus
        self.stimulus = Stimulus(self.config, preset, seed=seed, f_dW=None, phi0=None, is_dev=self.is_dev)
        self.logger.reset(self.stimulus.Phi.copy(), self.stimulus.R.copy())

    def update(self):
        if 0 < self.frame < self.t_start or self.frame > self.t_stop:
            self.frame += 1
            return [self.plotted_dots] + self.plotted_labels + [self.plotted_text]
        t_timer = self.timer.get_seconds()
        self.times.append(self.timer.get_seconds())  # Store the time of frame drawing
        if self.frame % int(self.fps) == 0:
            print(f"   > Wall-clock time: {t_timer:7.3f}s, "
                  f"simulation time: {self.frame / self.fps:7.3f}s, "
                  f"frame number: {self.frame:5d}")

        self.sim_lock.lock("Error: Plotting update called before sim was ready. Too high fps?")
        # # #  Update the figure with latest data  # # #
        x, y = self.logger.get(self.stimulus.N)
        self.plotted_dots.set_offsets(np.vstack([x, y]).T)
        for n, (xn, yn) in enumerate(zip(x, y)):
            self.plotted_labels[n].set_position((xn, yn))
        # # #  Integrate the stimulus until the next frame  # # #
        self.logger.log(*self.stimulus.advance())  # See class StructuredMotionStimulus in functions.py for dynamics
        self.sim_lock.unlock()

        if self.frame == self.t_stop:
            self.finish()
        self.frame += 1
        # # #  Return the list of variable figure elements (required for blitting)  # # #
        return [self.plotted_dots] + self.plotted_labels + [self.plotted_text]

    def finish(self):
        print(self.timer.get_seconds())
        self.callback(self.logger.data)

    def wait(self, callback=lambda: None):
        self.plotted_text.set_text('Click <left mouse button> or \npress <space> to continue.')
        self.cid['mouse'] = self.ax.get_figure().canvas.mpl_connect('button_press_event',
                                                                    lambda event: self.mousedown(event, callback))
        self.cid['key'] = self.ax.get_figure().canvas.mpl_connect('key_press_event',
                                                                  lambda event: self.keydown(event, callback))

    def mousedown(self, event, callback=lambda: None):
        if self.is_dev:
            print(f"<{event.button}> is pressed!")
        from matplotlib.backend_bases import MouseButton
        if event.button == MouseButton.LEFT:
            self.exit(callback)

    def keydown(self, event, callback=lambda: None):
        if self.is_dev:
            print(f"<{event.key}> is pressed!")
        if event.key == ' ':
            self.exit(callback)

    def exit(self, callback):
        for c in self.cid.values():
            self.ax.get_figure().canvas.mpl_disconnect(c)
        callback()


class Logger:
    def __init__(self):
        self.data = {}

    def reset(self, phi, r):
        self.data = dict(  # Store stimulus history (unless INFRUN)
            t=[0],         # time points of frames (in sim time)
            phi=[phi],     # Angular locations and velocities for all N dots
            r=[r],         # Radial locations and velocities for all N dots
        )

    def log(self, t, phi, r):
        self.data['t'].append(t)
        self.data['phi'].append(phi)
        self.data['r'].append(r)

    def get(self, n):
        return self.data['phi'][-1][:n], self.data['r'][-1][:n]
