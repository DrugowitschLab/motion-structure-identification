from utils.time import Timer
from utils.device_input import Devices
from response.visual import draw_structure
from matplotlib.patches import FancyBboxPatch
import pylab as pl


class Choice:
    def __init__(self, ax, structures, confidence, padding=0.02, visual=True):
        self.ax = ax
        self.timer = Timer()
        self.idx2button = {structure: [] for structure in structures}
        self.button2idx = {}
        self.draw(structures, confidence, padding)
        self.ground_truth, self.cid, self.callback = None, None, lambda data: None
        self.visual = visual

    def draw(self, structures, confidence, padding):
        _, _, w_win, h_win = self.ax.get_window_extent().bounds
        wh_ratio = w_win / h_win
        self.ax.axis('off')
        self.ax.axis('equal')
        self.ax.axis([0, wh_ratio, 0, 1])
        n_col, n_row = len(structures), len(confidence)
        w = (wh_ratio - padding * (n_col + 1)) / n_col
        h = (1 - padding * (n_row + 1)) / n_row

        label_kwargs = dict(fontsize=20)
        y = padding
        for r in range(n_row):
            self.ax.text(0, y + h / 2, confidence[r], ha='right', va='center', rotation=90, **label_kwargs)
            x = padding
            for c in range(n_col):
                if r == 0:
                    self.ax.text(x + w / 2, 1, structures[c], ha='center', va='bottom', **label_kwargs)
                draw_structure(self.ax, (x + padding, y + padding, w - padding * 2, h - padding * 2), structures[c])
                button = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0', alpha=0.2, color='gray', picker=True)
                self.button2idx[button] = (structures[c], confidence[r])
                self.idx2button[structures[c]].append(button)
                self.ax.add_patch(button)
                x += w + padding
            y += h + padding

    def reset(self, ground_truth, callback=lambda data: None):
        self.ground_truth = ground_truth
        self.callback = callback
        for button in self.button2idx:
            button.set_color('gray')
        self.ax.set_visible(True)
        pl.draw()
        Devices.disable('mouse')
        self.cid = self.ax.get_figure().canvas.mpl_connect('pick_event', lambda event: self.onclick(event))
        self.timer.restart()

    def update(self):
        return [self.ax]

    def onclick(self, event):
        structure_chosen, confidence_chosen = self.button2idx[event.artist]
        if self.ground_truth in self.idx2button:
            if structure_chosen == self.ground_truth:
                result = True
            else:
                for button in self.idx2button[structure_chosen]:
                    button.set_color('red')
                result = False
            for button in self.idx2button[self.ground_truth]:
                button.set_color('green')
            pl.draw()
        else:  # experiment 2
            result = True
            for button in self.idx2button[structure_chosen]:
                button.set_color('blue')
            pl.draw()
        self.ax.get_figure().canvas.mpl_disconnect(self.cid)  # ignore further mouse clicks
        self.callback({
            'ground_truth': self.ground_truth,
            'choice': structure_chosen,
            'confidence': confidence_chosen,
            'result': result,
            'rt': self.timer.get_seconds()
        })
        Devices.enable('mouse')
        # post callback behaviors:


if __name__ == '__main__':
    # unit test
    fig = pl.figure(figsize=(6 * 16 / 9, 6))
    Devices.init(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    c = Choice(ax, ('IND', 'GLO', 'CLU', 'SDH'), ('low', 'high'), 0.02)
    c.reset('GLO', callback=lambda data: print(data))
    pl.show()
