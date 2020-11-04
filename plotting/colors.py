import colorsys
import matplotlib.colors


def sns_edge_color(face_color: str):
    # seaborn.categorical.py: lines 311-317
    lum = colorsys.rgb_to_hls(*matplotlib.colors.to_rgba(face_color)[:3])[1] * .6
    return matplotlib.colors.rgb2hex((lum, lum, lum))


colors = {
    'decision_human': 'darkslateblue',
    'decision_model': 'slateblue',
    'decision_transfer': 'dodgerblue',
    'consistency_human': 'gray',
    'consistency_model': 'white',
    'confidence_human': 'darkgreen',
    'regression_line': 'gray',
}