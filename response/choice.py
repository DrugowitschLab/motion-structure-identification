from utils.time import Timer
from response.visual import draw_structure
from matplotlib.patches import FancyBboxPatch
import pylab as pl

class Choice:
    def __init__(self, ax, structures, confidence, padding=0.02):
        self.ax = ax
        self.timer = Timer()
        self.idx2button = {structure: [] for structure in structures}
        self.button2idx = {}
        self.draw(structures, confidence, padding)
        self.answer, self.cid, self.callback = None, None, lambda data: None

    def draw(self, structures, confidence, padding):
        _, _, w_win, h_win = self.ax.get_window_extent().bounds
        wh_ratio = w_win / h_win
        self.ax.axis('off')
        self.ax.axis('equal')
        self.ax.axis([0, wh_ratio, 0, 1])
        n_col, n_row = len(structures), len(confidence)
        w = (wh_ratio - padding * (n_col + 1)) / n_col
        h = (1 - padding * (n_row + 1)) / n_row

        x = padding
        for c in range(n_col):
            y = padding
            for r in range(n_row):
                draw_structure(self.ax, (x + padding, y + padding, w - padding * 2, h - padding * 2), structures[c])
                button = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0', alpha=0.2, color='gray', picker=True)
                self.button2idx[button] = (structures[c], confidence[r])
                self.idx2button[structures[c]].append(button)
                self.ax.add_patch(button)
                y += h + padding
            x += w + padding

    def reset(self, answer, callback=lambda data: None):
        self.answer = answer
        self.callback = callback
        for button in self.button2idx:
            button.set_color('gray')
        self.ax.set_visible(True)
        pl.draw()
        self.cid = self.ax.get_figure().canvas.mpl_connect('pick_event', lambda event: self.onclick(event))
        self.timer.restart()

    def update(self):
        return [self.ax]

    def onclick(self, event):
        structure_chosen, confidence_chosen = self.button2idx[event.artist]
        if structure_chosen == self.answer:
            result = True
        else:
            for button in self.idx2button[structure_chosen]:
                button.set_color('red')
            result = False
        self.callback({
            'answer': self.answer,
            'choice': structure_chosen,
            'confidence': confidence_chosen,
            'result': result,
            'rt': self.timer.get_seconds()
        })
        for button in self.idx2button[self.answer]:
            button.set_color('green')
        pl.draw()
        self.ax.get_figure().canvas.mpl_disconnect(self.cid)  # ignore further mouse clicks
        # post callback behaviors:


if __name__ == '__main__':
    # unit test
    fig = pl.figure(figsize=(6 * 16 / 9, 6))
    ax = fig.add_axes((0, 0, 1, 1))
    c = Choice(ax, ('IND', 'GLO', 'CLU', 'SDH'), ('low', 'high'), 0.02)
    c.reset('GLO', callback=lambda data: print(data))
    pl.show()
