# # def parse_args():
# #     from argparse import ArgumentParser, RawTextHelpFormatter
# #     parser = ArgumentParser(formatter_class=RawTextHelpFormatter,
# #                             description="Structured Motion Stimuli for Chicken experiments",
# #                             epilog="If using ipython3, indicate end of ipython arg parser via '--':\n"
# #                                    "   $ ipython3 play.py -- <args>")
# #     parser.add_argument(dest="preset_path", metavar="stimulus_file.py", type=str,
# #                         help="python file defining the motion structure (current working directory)")
# #     parser.add_argument("-s", dest="rngseed", metavar="rngseed", default=None, type=int,
# #                         help="Seed for numpy's random number generator (default: None)")
# #     parser.add_argument("-v", dest="vidfile", metavar="video_file.mp4", default=None, type=str,
# #                         help="Save video of stimulus to disk (default: None)")
# #     parser.add_argument("-t", dest="duration", metavar="seconds", default=None, type=float,
# #                         help="Stimulus duration in seconds, required for -v  (default: infinity)")
# #     parser.add_argument("-f", dest="isFullscreen", action='store_true',
# #                         help="Run in full screen (press ESC to close; default: false)")
# #     parser.add_argument("-T", dest="maxTrials", metavar="num trials", default=None, type=int,
# #                         help="Maximum number of trials (default: infinity)")
# #     parser.add_argument("-R", dest="repTrials", metavar="num reps", default=None, type=int,
# #                         help="Trial repetitions (requires -T; leads to T/R unique trials; default: 1)")
# #     parser.add_argument("-g", dest="greeter", metavar="string", default=None, type=str,
# #                         help="Greeter displayed before first trial")
# #     parser.add_argument("-u", dest="userID", metavar="ID", default=None, type=int,
# #                         help="Integer-valued ID of the participant")
# #     return parser.parse_args()
# #
# #
# # import pylab as pl
# # import stimuli.presets.GLO as preset
# # from config import config
# # from stimuli.animation import Animation
# # fig = pl.figure(figsize=config['display']['figsize'])
# # fig.canvas.set_window_title("Structured Motion Stimulus")
# # fig.set_facecolor(config['display']['bg_color'])
# # ax = fig.add_axes((0.01, 0.01, 0.98, 0.98), projection='polar')
# # a = Animation(ax, preset, seed=None, duration=10, DEV=False)
# # a.run()
# # pl.show()
# #
#
# from experiment import Experiment
# experiment = Experiment()
