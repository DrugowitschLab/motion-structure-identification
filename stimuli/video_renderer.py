from stimuli.motion import Motion
import matplotlib.pyplot as plt
from stimuli.presets import generate_preset

self.motion.reset(preset=generate_preset(),
                  onstart=lambda: self.motion_start(),
                  onstop=lambda data: self.motion_stop(data))
