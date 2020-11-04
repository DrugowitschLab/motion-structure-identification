import matplotlib as mpl

medium = 8
small = 6

config = {
    'figure': {'dpi': 300},
    'axes': {'titlesize': medium, 'titlepad': 2,
             'labelsize': medium, 'labelpad': 1,
             'axisbelow': True, 'linewidth': 0.5},
    'xtick': {'labelsize': medium, 'direction': 'in'},
    'xtick.major': {'size': 3, 'pad': 1, 'width': 0.5},
    'ytick': {'labelsize': medium, 'direction': 'in'},
    'ytick.major': {'size': 3, 'pad': 1, 'width': 0.5},
    'legend': {'fontsize': medium, 'borderpad': 0.1, 'labelspacing': 0.1, 'columnspacing': 0.1,
               'handlelength': 1.0, 'handleheight': 1, 'handletextpad': 0.1, 'borderaxespad': 0.1},
    'font': {'family': 'sans-serif', 'size': medium, 'weight': 'normal', 'stretch': 'condensed'},
    'lines': {'linewidth': 0.5, 'markersize': 2, 'color': 'k'},
    'image': {'aspect': 'equal'},
    'savefig': {'dpi': 300, 'pad_inches': 0},
    'mathtext': {'fontset': 'custom',
                 'it': 'STIXGeneral:italic', 'bf': 'STIXGeneral:italic:bold', 'cal': 'STIXNonUnicode'}
}

for key, kwargs in config.items():
    mpl.rc(key, **kwargs)
