from config import dev
from matplotlib import get_backend


class Cursor:
    fig = None

    @staticmethod
    def init(fig):
        Cursor.fig = fig

    @staticmethod
    def set_visible(is_visible):
        if get_backend() == "Qt5Agg":
            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QCursor
            cursor_shape = Qt.ArrowCursor if is_visible else Qt.BlankCursor
            QApplication.setOverrideCursor(QCursor(cursor_shape))

    @staticmethod
    def reset_mouse_position():
        if get_backend() == "Qt5Agg":
            from PyQt5.QtGui import QCursor
            # https://stackoverflow.com/questions/29702424/how-to-get-matplotlib-figure-size
            w_win, h_win = Cursor.fig.get_size_inches() * Cursor.fig.dpi
            QCursor.setPos(w_win // 2, h_win // 2)


class Devices:
    fig = None
    callback = {}
    """ mapping from buttons/keys to callback functions """
    mid = None
    """ button_press_event callback id """
    kid = None
    """ key_press_event callback id """

    @staticmethod
    def init(fig):
        Devices.fig = fig
        Devices.mid = fig.canvas.mpl_connect('button_press_event', lambda event: Devices.mousedown(event))
        Devices.kid = fig.canvas.mpl_connect('key_press_event', lambda event: Devices.keydown(event))

    @staticmethod
    def enable(button, callback=None):
        if button == 'mouse':
            Devices.mid = Devices.fig.canvas.mpl_connect('button_press_event', lambda event: Devices.mousedown(event))
        elif button == 'key':
            Devices.kid = Devices.fig.canvas.mpl_connect('key_press_event', lambda event: Devices.keydown(event))
        Devices.callback[button] = callback

    @staticmethod
    def disable(button):
        if button == 'mouse':
            Devices.fig.canvas.mpl_disconnect(Devices.mid)
        elif button == 'key':
            Devices.fig.canvas.mpl_disconnect(Devices.kid)
        Devices.callback.pop(button, None)

    @staticmethod
    def mousedown(event):
        if dev:
            print(f'Pressed <{event.button}> @ ({event.xdata:.4f},{event.ydata:.4f})')
        if event.button in Devices.callback:
            Devices.callback[event.button](event.xdata, event.ydata)

    @staticmethod
    def keydown(event):
        if dev:
            print(f'Pressed <{event.key}>')
        if event.key in Devices.callback:
            Devices.callback[event.key]()
