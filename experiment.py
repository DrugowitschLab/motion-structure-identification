from config import fps, dev, DisplayConfig
from stimuli.motion_structure import MotionStructure
import numpy as np
import pylab as pl
import matplotlib as mpl
from utils.time import timestamp
from utils.data import Logger, load_data
from utils.device_input import Cursor, Devices


class Experiment:
    structures = ('IND', 'GLO', 'CLU', 'SDH')
    # p_structures = [0.25, 0.25, 0.25, 0.25]
    p_structures = [0, 1/3, 1/3, 1/3]
    glo_SDH = 0.80  # 2/3
    presets = {'IND': MotionStructure(0, 2),
               'GLO': MotionStructure(1, 1 / 4),
               'CLU': MotionStructure(0, 1 / 4),
               'SDH': MotionStructure(glo_SDH, 1 / 4)}
    confidence = ('low', 'high')
    conf_score = {'low': {True: 1, False: 0}, 'high': {True: 2, False: -1}}

    def __init__(self, file, n_trials, n_repetition, is_fullscreen=False, seed_file=None):
        self.n_trials, self.n_repetition = n_trials, n_repetition
        self.fig, self.ax, self.motion, self.choice, self.scores, self.seeds, self.structure = [None] * 7
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
        choice_coords = [(0.64, 0.01, 0.35, 0.48), (0.01, 0.01, 0.35, 0.48)]
        self.ax = {
            'motion': self.fig.add_axes((0.01, 0.01, 0.98, 0.98), projection='polar'),
            'choice': self.fig.add_axes(choice_coords[0]),
            'scores': self.fig.add_axes((0.64, 0.71, 0.35, 0.28))
        }

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
        self.structure = rng.choice(self.structures, p=self.p_structures)
        color_permutation = [[0, 1, 2], [1, 2, 0], [2, 0, 1]][rng.randint(3)]
        self.motion.reset(preset=self.presets[self.structure],
                          seed=seed,
                          onstart=lambda: self.motion_start(),
                          onstop=lambda data: self.motion_stop(data),
                          color_permutation=color_permutation)
        self.logger.log({'permutation': color_permutation})
        self.idx += 1

    def motion_start(self):
        self.ax['choice'].set_visible(False)
        self.ax['scores'].set_visible(False)
        self.tasks.remove(self.choice)
        self.scores.clear()

    def motion_stop(self, data):
        Cursor.reset_mouse_position()
        Cursor.set_visible(True)
        self.logger.log(data)
        self.choice.reset(self.structure, lambda data: self.choice_made(data))
        self.tasks.append(self.choice)

    def choice_made(self, data):
        if dev:
            print(data['rt'])
        Cursor.set_visible(False)
        self.logger.log(data)
        self.logger.dump()
        self.scores.increase(self.conf_score[data['confidence']][data['result']])
        self.scores.reset(self.idx, self.n_trials)
        self.next_trial()

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


if __name__ == '__main__':
    pid = 'sichao'
    file = f'data/{pid}_{timestamp()}.dat'
    import os
    assert not os.path.exists(os.path.join('data', file)), f'{file} already exists'
    exp = Experiment(file, 200, 2, True)
