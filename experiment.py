import numpy as np
import pylab as pl
import matplotlib as mpl
from utils.log import Logger
from utils.cursor import Cursor


class Experiment:
    from config import config
    import stimuli.presets.IND as IND
    import stimuli.presets.GLO as GLO
    import stimuli.presets.CLU as CLU
    import stimuli.presets.SDH as SDH
    structures = ('IND', 'GLO', 'CLU', 'SDH')
    presets = {'IND': IND, 'GLO': GLO, 'CLU': CLU, 'SDH': SDH}
    confidence = ('low', 'high')
    conf_score = {'low': {True: 1, False: 0}, 'high': {True: 2, False: -1}}

    def __init__(self, pid, n_trials, n_repetition, is_fullscreen=False, is_dev=False, seed_file=None):
        self.n_trials, self.n_repetition = n_trials, n_repetition
        self.fig, self.ax, self.motion, self.choice, self.scores, self.seeds, self.structure = [None] * 7
        self.tasks = []
        self.idx = 0
        self.logger = Logger(f'data/{pid}')

        self.create_figure(is_fullscreen)
        self.create_axes()
        self.create_tasks()
        self.create_counterbalance(n_repetition)
        if seed_file:
            self.load_seeds(seed_file)
        self.next_trial()

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

    def create_axes(self):
        choice_coords = [(0.64, 0.01, 0.35, 0.48), (0.01, 0.01, 0.35, 0.48)]
        self.ax = {
            'motion': self.fig.add_axes((0.01, 0.01, 0.98, 0.98), projection='polar'),
            'choice': self.fig.add_axes(choice_coords[0]),
            'scores': self.fig.add_axes((0.64, 0.71, 0.35, 0.28))
        }

    def create_tasks(self):
        from stimuli.motion import Motion
        from response.choice import Choice
        from stimuli.text import Scores
        self.motion = Motion(self.ax['motion'], 3)
        self.choice = Choice(self.ax['choice'], self.structures, self.confidence)
        self.scores = Scores(self.ax['scores'])
        self.tasks.append(self.motion)
        self.tasks.append(self.scores)

    def create_counterbalance(self, n_rep):
        n_unique = self.n_trials // n_rep
        max_seed = np.iinfo(np.uint32).max
        self.seeds = np.tile(np.random.randint(0, high=max_seed, size=(n_unique, 2), dtype=np.uint32), (n_rep, 1))
        np.random.shuffle(self.seeds)

    def next_trial(self):
        # Cursor.set_visible(False)
        print(self.idx)
        if self.idx >= self.n_trials:
            self.finish()
        self.ax['choice'].set_visible(False)
        self.ax['scores'].set_visible(False)
        pl.draw()
        seed = self.seeds[self.idx, 0]
        self.logger.log({'seed': seed})
        rng = np.random.RandomState(seed=self.seeds[self.idx, 1])
        self.structure = rng.choice(self.structures, p=[0.25, 0.25, 0.25, 0.25])
        # self.structure = self.structs[self.idx]  # if seed_file
        self.motion.reset(self.presets[self.structure], seed, lambda data: self.motion_callback(data))
        self.idx += 1

    def motion_callback(self, data):
        # Cursor.reset_mouse_position(self.ax['motion'])
        # Cursor.set_visible(True)
        self.logger.log(data)
        self.choice.reset(self.structure, lambda data: self.choice_callback(data))
        self.tasks.append(self.choice)

    def choice_callback(self, data):
        # Cursor.set_visible(False)
        self.logger.log(data)
        self.logger.dump()
        self.scores.increase(self.conf_score[data['confidence']][data['result']])
        self.scores.reset(self.idx, self.n_trials)
        self.motion.wait(lambda: self.scores_callback())

    def scores_callback(self):
        self.tasks.remove(self.choice)
        self.scores.clear()
        self.next_trial()

    def update(self, frame):
        return [objects for task in self.tasks for objects in task.update()]

    def finish(self):
        if mpl.get_backend() == "TkAgg":  # TkAgg bug: https://github.com/matplotlib/matplotlib/issues/9856/
            print(" > Done. Please close the figure window.")
        else:
            print(" > Done. Figure window will be closed.")
            pl.close(self.fig)
        exit(0)

    def load_seeds(self, data_file):
        import pickle
        data = []
        with open(data_file, 'rb') as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
        self.structs = []
        for i in range(len(data)):
            self.seeds[i, 0] = data[i]['seed']
            self.structs.append(data[i]['answer'])


if __name__ == '__main__':
    data_file = 'sichao_rep'
    import os
    assert not os.path.exists(os.path.join('data', data_file)), f'{data_file} already exists'
    exp = Experiment(data_file, 200, 2, True)
