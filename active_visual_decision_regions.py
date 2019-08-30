import numpy as np
import pylab as pl
import matplotlib as mpl
from utils.time import timestamp
from utils.data import Logger, load_data
from utils.device_input import Cursor, Devices
from stimuli.presets.generate_preset import Preset


class Experiment:
    from config import config
    structures = ('IND', 'GLO', 'CLU', 'SDH')
    p_structures = [0.25, 0.25, 0.25, 0.25]
    presets = {'IND': Preset(0, 2),
               'GLO': Preset(1, 1/4),
               'CLU': Preset(0, 1/4),
               'SDH': Preset(2/3, 1/4)}
    confidence = ('low', 'high')
    conf_score = {'low': {True: 1, False: 0}, 'high': {True: 2, False: -1}}

    def __init__(self, file, n_trials, n_repetition, is_fullscreen=False, is_dev=False, seed_file=None):
        self.n_trials, self.n_repetition, self.DEV = n_trials, n_repetition, is_dev
        self.fig, self.ax, self.motion, self.choice, self.scores, self.seeds, self.structure = [None] * 7
        self.tasks = []
        self.idx = 0

        self.create_figure(is_fullscreen)
        self.create_axes()
        self.create_counterbalance(n_repetition, seed_file)

        self.cursor = Cursor(self.fig)
        self.event = Devices(self.fig, self.DEV)
        self.event.enable('escape', self.finish)

        self.init()

        import matplotlib.animation as animation
        interval = 1000 / self.config['display']['fps']
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=interval, blit=True, repeat=False)
        pl.show()

    def create_figure(self, is_fullscreen):
        mpl.use(self.config['display']['backend_interactive'])
        print(" > Used backend:", mpl.get_backend())
        pl.ioff()
        mpl.rcParams['toolbar'] = 'None'
        mpl.rc("figure", dpi=self.config['display']['monitor_dpi'])  # set monitor dpi
        self.fig = pl.figure(figsize=self.config['display']['figsize'])
        self.fig.canvas.set_window_title("Motion Structure Identification Task")
        self.fig.set_facecolor(self.config['display']['bg_color'])
        if is_fullscreen:
            manager = pl.get_current_fig_manager()
            if mpl.get_backend() == "TkAgg":
                manager.full_screen_toggle()
            elif mpl.get_backend() in ("Qt4Agg", "Qt5Agg"):
                manager.window.showFullScreen()

        mpl.rcParams['toolbar'] = 'toolbar2'
        self.fig2 = pl.figure(figsize=self.config['display']['figsize'])
        self.fig2.canvas.set_window_title("Decision Boundary Visualization")

    def create_axes(self):
        choice_coords = [(0.64, 0.01, 0.35, 0.48), (0.01, 0.01, 0.35, 0.48)]
        self.ax = {
            'motion': self.fig.add_axes((0.01, 0.01, 0.98, 0.98), projection='polar'),
            'choice': self.fig.add_axes(choice_coords[0]),
            'scores': self.fig.add_axes((0.64, 0.71, 0.35, 0.28))
        }
        self.ax2 = self.fig2.add_axes((0.05, 0.05, 0.9, 0.9))
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 2)
        # from matplotlib.widgets import Cursor as Cursor2
        # Cursor2(self.ax2, useblit=False, color='red', linewidth=2)
        from analysis.decision_regions import DecisionRegions
        self.dr = DecisionRegions(self.ax2, [[0, 2], [1, 1 / 4], [0, 1 / 4], [2 / 3, 1 / 4]], [0, 1, 2, 3])
        self.fig2.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        self.glo, self.lam_I = event.xdata, event.ydata
        if 0 <= self.glo <= 1 and 0 <= self.lam_I <= 2:
            preset = Preset(self.glo, self.lam_I)
            self.next_trial(preset)



    def create_counterbalance(self, n_rep, seeds_file=None):
        if seeds_file:
            self.seeds = load_data(seeds_file, 'seed')
        else:
            n_unique = self.n_trials // n_rep
            max_seed = np.iinfo(np.uint32).max
            self.seeds = np.tile(np.random.randint(0, high=max_seed, size=(n_unique, 1), dtype=np.uint32), (n_rep, 1))
            np.random.shuffle(self.seeds)

    def init(self):
        from stimuli.motion import Motion
        from response.choice import Choice
        from stimuli.text import Scores
        self.motion = Motion(self.ax['motion'], 3)
        self.choice = Choice(self.ax['choice'], self.structures, self.confidence, visual=False)
        self.scores = Scores(self.ax['scores'])

        self.tasks.append(self.motion)
        self.tasks.append(self.choice)
        self.tasks.append(self.scores)

        self.ax['motion'].set_visible(True)
        self.ax['choice'].set_visible(False)
        self.ax['scores'].set_visible(False)
        self.cursor.set_visible(False)

        # self.next_trial()

    def next_trial(self, preset):
        print(f'Trial #{self.idx + 1}')
        if self.idx >= self.n_trials:
            self.finish()
        seed = self.seeds[self.idx]
        self.motion.reset(preset=preset,
                          seed=seed,
                          onstart=lambda: self.motion_start(),
                          onstop=lambda data: self.motion_stop(data))
        self.idx += 1

    def motion_start(self):
        self.ax['choice'].set_visible(False)
        self.ax['scores'].set_visible(False)
        self.tasks.remove(self.choice)
        self.scores.clear()

    def motion_stop(self, data):
        self.cursor.reset_mouse_position()
        self.cursor.set_visible(True)
        self.choice.reset(self.structure, lambda data: self.choice_made(data))
        self.tasks.append(self.choice)

    def choice_made(self, data):
        if self.DEV:
            print(data['rt'])
        self.cursor.set_visible(False)
        self.scores.increase(self.conf_score[data['confidence']][data['result']])
        self.scores.reset(self.idx, self.n_trials)
        self.dr.fit([self.glo, self.lam_I], self.structures.index(data['choice']))

    def update(self, frame):
        return [objects for task in self.tasks for objects in task.update()]

    def finish(self):
        if mpl.get_backend() == "TkAgg":  # TkAgg bug: https://github.com/matplotlib/matplotlib/issues/9856/
            print(" > Done. Please close the figure window.")
        else:
            print(" > Done. Figure window will be closed.")
            pl.close(self.fig)
        exit(0)


if __name__ == '__main__':
    pid = 'sichao'
    file = f'data/{pid}_{timestamp()}'
    import os
    assert not os.path.exists(os.path.join('data', file)), f'{file} already exists'
    exp = Experiment(file, 200, 2, False, True)
