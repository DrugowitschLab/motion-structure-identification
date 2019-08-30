from config import fps, DisplayConfig
import numpy as np
import pylab as pl
import matplotlib as mpl
from utils.time import timestamp
from utils.data import Logger, load_data
from utils.device_input import Cursor, Devices


class Experiment:
    directory = 'data/'
    structures = ()
    p_structures = []
    presets = {}
    confidence = ()
    confidence_score = {'low': {True: 0, False: 0}, 'high': {True: 0, False: 0}}

    def __init__(self, pid, n_trials, n_repetition=1, is_fullscreen=True, seed_file=None):
        file = f'{self.directory}/{pid}_{timestamp()}.dat'
        self.n_trials, self.n_repetition = n_trials, n_repetition
        self.fig, self.ax, self.motion, self.choice, self.scores, self.seeds, self.structure, self.truth = [None] * 8
        self.tasks = []
        self.idx = 0
        self.logger = Logger(file)

        self.create_figure(is_fullscreen)
        self.create_axes()
        self.create_counterbalance(n_repetition, seed_file)

        self.init()

        import matplotlib.animation as animation
        interval = 1000 / fps
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=interval, blit=True, repeat=False)
        pl.show()

    def create_figure(self, is_fullscreen):
        print(DisplayConfig.backend_interactive)
        mpl.use(DisplayConfig.backend_interactive)
        print(" > Used backend:", mpl.get_backend())
        pl.ioff()
        mpl.rcParams['toolbar'] = 'None'
        mpl.rc("figure", dpi=DisplayConfig.monitor_dpi)  # set monitor dpi
        self.fig = pl.figure(figsize=DisplayConfig.figsize)
        self.fig.canvas.set_window_title("Motion Structure Identification Task")
        self.fig.canvas.window().statusBar().setVisible(False)
        self.fig.set_facecolor(DisplayConfig.bg_color)
        if is_fullscreen:
            manager = pl.get_current_fig_manager()
            if mpl.get_backend() == "TkAgg":
                manager.full_screen_toggle()
            elif mpl.get_backend() in ("Qt4Agg", "Qt5Agg"):
                manager.window.showFullScreen()
        Cursor.init(self.fig)
        Devices.init(self.fig)
        Devices.enable('escape', self.finish)

    def create_axes(self):
        self.ax = {
            'motion': self.fig.add_axes((0.01, 0.01, 0.98, 0.98), projection='polar'),
            'choice': self.fig.add_axes((0.64, 0.01, 0.35, 0.48)),
            'scores': self.fig.add_axes((0.64, 0.71, 0.35, 0.28))
        }

    def create_counterbalance(self, n_rep, seeds_file=None):
        if seeds_file:
            self.seeds = load_data(seeds_file, 'seed')
            self.truth = load_data(seeds_file, 'answer')
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
        self.choice = Choice(self.ax['choice'], self.structures, self.confidence)
        self.scores = Scores(self.ax['scores'])

        self.tasks.append(self.motion)
        self.tasks.append(self.choice)
        self.tasks.append(self.scores)

        self.ax['motion'].set_visible(True)
        self.ax['choice'].set_visible(False)
        self.ax['scores'].set_visible(False)
        Cursor.set_visible(False)

        self.next_trial()

    def next_trial(self):
        print(f'Trial #{self.idx + 1}')
        if self.idx >= self.n_trials:
            self.finish()
        seed = self.seeds[self.idx]
        self.logger.log({'seed': seed})
        rng = np.random.RandomState(seed=seed)
        color_permutation = [[0, 1, 2], [1, 2, 0], [2, 0, 1]][rng.randint(3)]
        self.logger.log({'permutation': color_permutation})
        if self.truth is not None:
            self.structure = self.truth[self.idx]
        else:
            self.structure = rng.choice(self.structures, p=self.p_structures)
        print(self.structure)
        structure_obj = self.presets[self.structure]
        print(structure_obj)

        self.motion.reset(preset=structure_obj,
                          seed=seed,
                          onstop=self.motion_stop,
                          onfinish=self.next_trial,
                          color_permutation=color_permutation)
        self.ax['choice'].set_visible(False)
        self.ax['scores'].set_visible(False)
        self.tasks.remove(self.choice)
        self.scores.clear()

        self.idx += 1

    def motion_stop(self, data):
        Cursor.reset_mouse_position()
        Cursor.set_visible(True)
        self.logger.log(data)
        self.choice.reset(self.structure, self.choice_made)
        self.tasks.append(self.choice)

    def choice_made(self, data):
        Cursor.set_visible(False)
        self.motion.prompt()
        self.logger.log(data)
        self.logger.dump()
        self.scores.increase(self.confidence_score[data['confidence']][data['result']])
        self.scores.reset(self.idx, self.n_trials)

    def update(self, frame):
        return [objects for task in self.tasks for objects in task.update()]

    def finish(self):
        self.logger.close()
        if mpl.get_backend() == "TkAgg":  # TkAgg bug: https://github.com/matplotlib/matplotlib/issues/9856/
            print(" > Done. Please close the figure window.")
        else:
            print(" > Done. Figure window will be closed.")
            pl.close(self.fig)
        exit(0)
