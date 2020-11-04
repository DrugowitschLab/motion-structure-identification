#!/usr/bin/env python
__author__ = 'Sichao Yang, Computational Principles of Intelligence Lab, MPI for Biological Cybernetics: cpilab.org'
__contact__ = 'sichao@cs.wisc.edu'
__date__ = 'July 24, 2020'
__version__ = '1.0'


from typing import Callable, Dict, List, Optional
from string import ascii_uppercase
from os.path import join
import matplotlib.pyplot as plt
from lxml import etree

px_per_unit = {
    'px': 1,                # pixel
    'in': 96,               # inch
    'cm': 37.795275591,     # centimeter
    'pc': 16,               # pica
    'pt': 4/3,              # point
}
working_dir: str    # the working directory, which all which all SVG file paths are relative to
unit = 'cm'         # the unit of all input lengths (width, height, x, and y, but not fontsize)


class Panel:
    """ A panel in a figure. """
    def __init__(self, path: str, x: float, y: float, w: float, h: float,
                 plotting_function: Optional[Callable[[plt.Axes], None]] = None, keep_aspect_ratio: bool = True):
        """
        :param path: relative path from working directory to the SVG image for this panel.
        :param x: desired distance between the left of the panel and the left of the figure.
        :param y: desired distance between the top of the panel and the top of the figure.
        :param w: desired width of the panel.
        :param h: desired height of the panel.
        :param plotting_function: a function that generates the figure on a pyplot Axes.
        :param keep_aspect_ratio: whether to keep the aspect ratio while resizing, default to False.
        """
        self.path = join(working_dir, path)
        px = px_per_unit[unit]                              # converts lengths from custom unit to pixel
        self.x, self.y, self.w, self.h, self.plotting_function = x * px, y * px, w * px, h * px, plotting_function
        inch = px_per_unit[unit] / px_per_unit['in']        # converts lengths from custom unit to inch
        self.w_in, self.h_in = w * inch, h * inch           # to specify pyplot figure size, in inches
        self.keep_aspect_ratio = keep_aspect_ratio
        # self.keep_aspect_ratio = plotting_function is None  # always keep aspect ratio if the svg is pre-generated

    def plot(self) -> etree.Element:
        """ Loads an SVG into a <g> element, rescales it to the desired size, and positions it at the desired location.
        The SVG will be generated/regenerated using the custom pyplot plotting function if provided.
        :return: the <g> SVG element containing all sub-elements in the panel.
        """
        if self.plotting_function is not None:                         # regenerates the figure
            _, ax = plt.subplots(figsize=(self.w_in, self.h_in))       # pyplot figure size is in inches
            self.plotting_function(ax)                                 # the function should plot onto the provided axis
            plt.savefig(self.path, transparent=True)                   # save the plot as an SVG to be loaded back later
        with open(self.path) as f:
            svg = etree.parse(f).getroot()                             # parses the SVG file into an XML element tree
        _x1, _y1, _x2, _y2 = [float(f) for f in svg.get('viewBox').split()]         # the original size
        scale_x, scale_y = self.w / (_x2 - _x1), self.h / (_y2 - _y1)               # desired scale ratio
        if self.keep_aspect_ratio:
            scale_x = scale_y = min(scale_x, scale_y)
        svg.tag = 'g'               # converts <svg> into <g>, where all sub-elements can be transformed together
        svg.attrib.clear()          # clear the <svg> attributes for clarity, although they are not used by <g>
        svg.set('transform', f'translate({self.x} {self.y}) scale({scale_x} {scale_y})')
        return svg

    def preview(self):
        """ Plots and displays the panel using matplotlib.pyplot.show. """
        assert self.plotting_function is not None
        _, ax = plt.subplots(figsize=(self.w_in, self.h_in))
        self.plotting_function(ax)
        plt.show()


class PanelLabel:
    """ The label of a panel. """
    def __init__(self, label: str, dx: float, dy: float, style: Optional[Dict[str, any]] = None):
        """
        :param label: text label of this panel.
        :param dx: desired offset from the left of the panel to the anchor of the text.
        :param dy: desired offset from the top of the panel to the anchor of the text.
        :param style: styling attributes of this SVG element. https://www.w3.org/TR/SVG11/styling.html#StyleAttribute.
        """
        px = px_per_unit[unit]  # converts lengths from custom unit to pixel
        self.label, self.dx, self.dy, self.style = label, dx * px, dy * px, style

    def plot(self, x, y) -> etree.ElementTree:
        """ Creates a <text> element as the label of a panel and moves it to a relative location from the panel.
        :param x: desired distance between the left of the panel and the left of the figure.
        :param y: desired distance between the top of the panel and the top of the figure.
        :return: the <text> SVG element as the index label of the panel.
        """
        text = etree.Element('text', {'x': str(x + self.dx), 'y': str(y + self.dy), **self.style})  # sets offset
        text.text = self.label
        return text

    @staticmethod
    def generate_labels(n: int, dx: float, dy: float, style: Optional[Dict[str, any]] = None,
                        labels: Optional[List[str]] = None):
        """ Creates the index labels for all the panels in a batch.
        :param n: number of panels.
        :param dx: desired offset from the left of the panel to the anchor of the text, for all panels.
        :param dy: desired offset from the top of the panel to the anchor of the text, for all panels.
        :param style: styling attributes of all labels. https://www.w3.org/TR/SVG11/styling.html#StyleAttribute.
        :param labels: labels to use, default to be uppercase letters: 'A', 'B', 'C', ...
        :return: a list of PanelLabel objects as index labels for all panels.
        """
        labels = list(ascii_uppercase) if labels is None else labels
        assert n <= len(labels), f'{n} panels are expected, but only {len(labels)} labels are provided.'
        return [PanelLabel(labels[i], dx, dy, style) for i in range(n)]


class Figure:
    """ A figure, container of all panels and panel labels. """
    def __init__(self, w, h, panels: List[Panel], labels: List[PanelLabel]):
        """
        :param w: desired width of the figure.
        :param h: desired height of the figure.
        :param panels: panels to be combined into the figure.
        :param labels: index labels of the panels.
        """
        assert len(panels) == len(labels)
        self.panels, self.labels = panels, labels
        px = px_per_unit[unit]  # converts lengths from custom unit to pixel
        w, h = w * px, h * px
        self.fig = etree.Element('svg',     # creates an empty <svg> tag, as the container of all <g> and <text> tags
                                 {'version': '1.1', 'width': str(w), 'height': str(h), 'viewBox': f'0 0 {w} {h}'},
                                 nsmap={None: 'http://www.w3.org/2000/svg', 'xlink': 'http://www.w3.org/1999/xlink'})

    def plot(self, path: str):
        """ Sets all <g> panels and <text> labels as sub-elements of the <svg> figure, and saves it to the disk.
        :param path: relative path from working directory to the desired output SVG file.
        """
        for panel, label in zip(self.panels, self.labels):
            self.fig.append(panel.plot())
            self.fig.append(label.plot(panel.x, panel.y))
            with open(join(working_dir, path), 'wb') as f:
                f.write(etree.tostring(self.fig, xml_declaration=True, pretty_print=True, standalone=True))
