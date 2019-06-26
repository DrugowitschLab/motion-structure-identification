import matplotlib as mpl


class Cursor(object):
    @staticmethod
    def set_visible(is_visible):
        if mpl.get_backend() == "Qt5Agg":
            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QCursor
            cursor_shape = Qt.ArrowCursor if is_visible else Qt.BlankCursor
            QApplication.setOverrideCursor(QCursor(cursor_shape))

    @staticmethod
    def reset_mouse_position(fig):
        if mpl.get_backend() == "Qt5Agg":
            from PyQt5.QtGui import QCursor
            # https://stackoverflow.com/questions/29702424/how-to-get-matplotlib-figure-size
            w_win, h_win = fig.get_size_inches() * fig.dpi
            QCursor.setPos(w_win // 2, h_win // 2)
