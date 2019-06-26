import pylab as pl


class Scores:
    from config import config

    arrows = ['(↓1)', '(0)', '(↑1)', '(⇈2)']
    colors = ['red', 'gray', 'green', 'green']

    def __init__(self, ax):
        self.ax = ax
        self.score = 0
        self.text = {}
        self.draw()
        self.callback = lambda: None
        self.frame = 0

    def draw(self):
        self.ax.axis('off')
        self.ax.axis('equal')
        kwargs = dict(weight='bold', family='monospace', size='14', ha='right', va='top', transform=self.ax.transAxes)
        self.text['score'] = self.ax.text(1, 1, '', **kwargs)
        self.text['change'] = self.ax.text(1, 1, '', **kwargs)

    def reset(self, idx, n_trial, callback=lambda: None):
        self.callback = callback
        self.text['score'].set_text(f"score: {self.score:3d}     \ntrials: {idx:3d}/{n_trial:3d} ")
        self.ax.set_visible(True)
        pl.draw()
        self.frame = 0

    def increase(self, change):
        self.text['change'].set_text(f" {self.arrows[change + 1]:4}")
        self.text['change'].set_color(self.colors[change + 1])
        self.score += change

    def clear(self):
        self.text['score'].set_text('')
        self.text['change'].set_text('')
        pl.draw()

    def update(self):
        if self.frame == self.config['experiment']['post_choice']:
            self.finish()
        self.frame += 1
        return []

    def finish(self):
        self.callback()


if __name__ == '__main__':
    fig = pl.figure(figsize=(6 * 16 / 9, 6))
    ax = fig.add_axes((0, 0, 1, 1))
    s = Scores(ax)
    pl.show()
