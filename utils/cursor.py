class Cursor(object):
    @staticmethod
    def set_visible(is_visible):
        import matplotlib as mpl
        if mpl.get_backend() in ("Qt4Agg", "TkAgg"):
            pass
        elif mpl.get_backend() == "Qt5Agg":
            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QCursor
            cursor_shape = Qt.ArrowCursor if is_visible else Qt.BlankCursor
            QApplication.setOverrideCursor(QCursor(cursor_shape))

    @staticmethod
    def reset_mouse_position(ax):
        import matplotlib as mpl
        if mpl.get_backend() in ("Qt4Agg", "TkAgg"):
            pass
        elif mpl.get_backend() == "Qt5Agg":
            from PyQt5.QtGui import QCursor
            import pylab as pl
            _, _, w_win, h_win = ax.get_window_extent().bounds
            # screen_coords = pl.gca().transAxes.transform((0, 0)).astype(int)
            # screen_coords += pl.get_current_fig_manager().window.geometry().getRect()[:2]
            # print(screen_coords)
            QCursor.setPos(w_win // 2, h_win // 2)