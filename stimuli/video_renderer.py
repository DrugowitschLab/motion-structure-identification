from stimuli.motion import Motion
from stimuli.motion_structure import MotionStructure
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from config import fps, ExperimentConfig, DisplayConfig


presets = {
    'IND': MotionStructure(1, ExperimentConfig.λ_T),
    'GLO': MotionStructure(1, ExperimentConfig.λ_I),
    'CLU': MotionStructure(0, ExperimentConfig.λ_I),
    'SDH': MotionStructure(ExperimentConfig.glo_SDH, ExperimentConfig.λ_I)
}


print(DisplayConfig.backend_interactive)
mpl.use(DisplayConfig.backend_interactive)
print(" > Used backend:", mpl.get_backend())
plt.ioff()
mpl.rcParams['toolbar'] = 'None'
mpl.rc("figure", dpi=DisplayConfig.monitor_dpi)  # set monitor dpi
fig = plt.figure(figsize=DisplayConfig.figsize)
fig.canvas.set_window_title("Motion Structure Identification Task")
fig.canvas.window().statusBar().setVisible(False)
fig.set_facecolor(DisplayConfig.bg_color)
ax = fig.add_axes((0.01, 0.01, 0.98, 0.98), projection='polar')
motion = Motion(ax, 3)
s = 'SDH'
motion.reset(preset=presets[s])
motion.start()
motion.plotted_text.set_text(s)
ani = animation.FuncAnimation(fig, lambda frame: motion.update(),
                              frames=ExperimentConfig.duration - ExperimentConfig.delay,
                              interval=1000 / fps, blit=True)
# plt.show()
ani.save(f'{s}.gif', writer='imagemagick', fps=fps)
