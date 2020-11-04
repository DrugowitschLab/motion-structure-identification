from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pylab as pl
import matplotlib as mpl
from matplotlib import animation

from config import fps, DisplayConfig, ExperimentConfig
from task import Task
from utils.time import timestamp
from utils.data import Logger, load_data
from utils.device_input import Cursor, Devices
from stimuli.motion_structure import MotionStructure
from stimuli.motion import Motion
from response.choice import Choice
from stimuli.text import Scores


class Experiment(ABC):
    directory: str = 'data'                         # data storage path
    structures: List[str]                           # motion structure names
    p_structures: List[float]                       # generation probabilities of motion structures, must sum up to 1
    presets: Dict[str, MotionStructure]             # motion structure objects
    confidence: List[str] = ['low', 'high']         # confidence level names
    confidence_score: Dict[Tuple[bool, str], int]   # (correct/incorrect, confidence) -> score mapping
    seeds: Union[List[int], np.ndarray]             # ``n_trials`` seeds for stochastic trial generation
    truth: Optional[Union[List[int], np.ndarray]]   # ``n_trials`` ground truth for deterministic trial generation
    fig: pl.Figure                                  # figure
    ax: Dict[str, pl.Axes]                          # axes for motion, choice, score
    motion: Motion                                  # motion simulator
    choice: Choice                                  # choice responsor
    scores: Scores                                  # score visualizer
    tasks: List[Task] = []                          # pipeline of rendering tasks
    idx: int = 0                                    # current trial index

    def __init__(self, pid: str, n_trials: int, n_repetitions: int = 1,
                 is_fullscreen: bool = True, seed_file: Optional[str] = None):
        """ Initializes the experiment animation.

        :param pid: participant id.
        :param n_trials: number of total trials.
        :param n_repetitions: number of repetitions of unique trials.
        :param is_fullscreen: fullscreen toggle.
        :param seed_file: path to the file storing seeds and ground truth names, for repetition.
        """
        path = os.path.join(self.directory, pid)
        if not os.path.exists(path):
            os.mkdir(path)
        self.n_trials = n_trials
        self.logger = Logger(os.path.join(path, f'{pid}_{timestamp()}.dat'))
        self.create_figure(is_fullscreen)
        self.create_axes()
        self.create_counterbalance(n_repetitions, seed_file)
        self.init()
        self.animation = animation.FuncAnimation(self.fig, self.update, interval=1000 / fps, blit=True, repeat=False)
        pl.show()

    def create_figure(self, is_fullscreen: bool):
        """ Sets up animation window.

        :param is_fullscreen: fullscreen toggle.
        """
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
        if is_fullscreen:   # Show fullscreen.
            manager = pl.get_current_fig_manager()
            if mpl.get_backend() == "TkAgg":
                manager.full_screen_toggle()
            elif mpl.get_backend() in ("Qt4Agg", "Qt5Agg"):
                manager.window.showFullScreen()
        Cursor.init(self.fig)   # Initializes cursor controller.
        Devices.init(self.fig)  # Initializes mouse/keyboard event controller.
        Devices.enable('escape', self.finish)   # Press 'ESC' to quit.

    def create_axes(self):
        """ Creates axes for motion animation, choice response, and score visualization. """
        self.ax = {
            'motion': self.fig.add_axes((0.01, 0.01, 0.98, 0.98), projection='polar'),
            'choice': self.fig.add_axes((0.64, 0.01, 0.35, 0.48)),
            'scores': self.fig.add_axes((0.64, 0.71, 0.35, 0.28))
        }

    def create_counterbalance(self, n_rep, seeds_file=None):
        """ Generates seeds of all the trials """
        if seeds_file:
            self.seeds = load_data(seeds_file, 'seed')
            self.truth = load_data(seeds_file, 'answer')
        else:
            n_unique = self.n_trials // n_rep
            max_seed = np.iinfo(np.uint32).max
            self.seeds = np.tile(np.random.randint(0, high=max_seed, size=(n_unique, 1), dtype=np.uint32), (n_rep, 1))
            np.random.shuffle(self.seeds)
            self.truth = None

    def init(self):
        """ Initializes animation. """
        self.motion = Motion(self.ax['motion'])
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
        color_permutation = ExperimentConfig.permutations[rng.choice(len(ExperimentConfig.permutations))]
        self.logger.log({'permutation': color_permutation})
        if self.truth is not None:
            structure = self.truth[self.idx]
        else:
            structure = rng.choice(self.structures, p=self.p_structures)
        print(structure)
        preset = self.presets[structure]
        print(preset)

        self.motion.reset(preset=preset,
                          seed=seed,
                          onstop=lambda data: self.motion_stop(data, structure),
                          onfinish=self.next_trial,
                          color_permutation=color_permutation)
        self.ax['choice'].set_visible(False)
        self.ax['scores'].set_visible(False)
        self.tasks.remove(self.choice)
        self.scores.clear()

        self.idx += 1

    def motion_stop(self, data, structure):
        Cursor.reset_mouse_position()
        Cursor.set_visible(True)
        self.logger.log(data)
        self.choice.reset(structure, self.choice_made)
        self.tasks.append(self.choice)

    def choice_made(self, data):
        Cursor.set_visible(False)
        self.motion.prompt()
        self.logger.log(data)
        self.logger.dump()
        self.scores.increase(self.confidence_score[data['result'], data['confidence']])
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
