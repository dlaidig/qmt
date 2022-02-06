# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import atexit
import os
from pathlib import Path

_config = {
    'mode': 'show',
    'filename': None,
    'autoclose': True,
    'i': 0,
    'kwargs': dict(figsize=(9, 7)),
    'pdfpages': None,
    'title': None,
}


NO_VALUE = object()  # make sure we can detect that None is passed


def setupDebugPlots(mode=NO_VALUE, filename=NO_VALUE, autoclose=NO_VALUE, figsize=NO_VALUE, figsize_cm=NO_VALUE,
                    dpi=NO_VALUE, title=NO_VALUE):
    """
    Sets up how debug plots are created when functions are called with ``plot=True``.

    :param mode: Sets the mode in which debug plots work. Possible modes:

        - show (default): figure is created and plt.show() is called automatically
        - create: figure is only created but not saved or shown
        - save: figure is automatically saved to file using savefig
        - pdfpages: figures are written to a multipages PDF file
    :param filename: filename of the output file, can contain a format string like ``{i:03d}`` for an auto-increasing
        counter
    :param autoclose: if set to True (default), figures will automatically be closed in save and pdfpages mode
    :param figsize: figure size in inches
    :param figsize_cm: figure size in cm
    :param dpi: dpi of the figure
    :param title: custom title that some plot functions may add to the plot (e.g. to identify different data sets)
    :return: None
    """
    outputChanged = False

    if mode is not NO_VALUE:
        assert mode in ('show', 'create', 'save', 'pdfpages')
        assert mode in ('show', 'create', 'save', 'pdfpages')
        _config['mode'] = mode
        outputChanged = True

    if filename is not NO_VALUE:
        assert filename is None or isinstance(filename, (str, Path))
        _config['filename'] = str(filename) if isinstance(filename, Path) else filename
        outputChanged = True

    if autoclose is not NO_VALUE:
        _config['autoclose'] = bool(autoclose)

    if figsize is not NO_VALUE:
        if figsize is None:
            _config['kwargs'].pop('figsize', None)
        else:
            assert isinstance(figsize, (tuple, list))
            assert len(figsize) == 2
            _config['kwargs']['figsize'] = figsize

    if figsize_cm is not NO_VALUE:
        if figsize_cm is None:
            _config['kwargs'].pop('figsize', None)
        else:
            assert isinstance(figsize_cm, (tuple, list))
            assert len(figsize_cm) == 2
            _config['kwargs']['figsize'] = tuple(v/2.54 for v in figsize_cm)

    if dpi is not NO_VALUE:
        if dpi is None:
            _config['kwargs'].pop('dpi', None)
        else:
            _config['kwargs']['dpi'] = float(dpi)

    if title is not NO_VALUE:
        assert title is None or isinstance(title, str)
        _config['title'] = title

    if _config['mode'] == 'save' or _config['mode'] == 'pdfpages':
        assert _config['filename'] is not None

    if outputChanged:
        _config['i'] = 0
        if _config['pdfpages'] is not None:
            _config['pdfpages'].close()
            _config['pdfpages'] = None

        if _config['mode'] == 'pdfpages':
            from matplotlib.backends.backend_pdf import PdfPages
            os.makedirs(os.path.dirname(os.path.abspath(_config['filename'])), exist_ok=True)
            _config['pdfpages'] = PdfPages(filename=_config['filename'])


atexit.register(lambda: setupDebugPlots(mode='show'))  # ensures PDF is properly closed


class AutoFigure:
    """
    Context manager for handling automatic figure creation in plot functions.

    Use :meth:`qmt.setupDebugPlots` to control how debug plots are created.
    """
    def __init__(self, fig):
        if fig is None or fig is True:
            self.fig = self._createFigure()
            self.enabled = True
        else:
            self.fig = fig
            self.enabled = False

    def __enter__(self):
        return self.fig

    def __exit__(self, *args):
        if not self.enabled:
            return

        import matplotlib.pyplot as plt
        if _config['mode'] == 'show':
            plt.show()
        elif _config['mode'] == 'save':
            filename = _config['filename'].format(i=_config['i'])
            _config['i'] += 1
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            self.fig.savefig(filename)
            if _config['autoclose']:
                plt.close(self.fig)
        elif _config['mode'] == 'pdfpages':
            _config['pdfpages'].savefig(self.fig)
            if _config['autoclose']:
                plt.close(self.fig)

    @staticmethod
    def _createFigure():
        import matplotlib.pyplot as plt
        fig = plt.figure(**_config['kwargs'])
        return fig

    @staticmethod
    def title(functionName=None):
        title = _config['title']
        if functionName is None:
            return '' if title is None else title
        else:
            return f'{functionName} debug plot' if title is None else f'{functionName} debug plot: {title}'


def extendXlim(ax, left=None, right=None):
    """
    Extends xlim of matplotlib axes.

    This function is useful for safely setting axes limits without the risk of accidentally cutting off data.
    Make sure to plot all data before calling this function so that the automatic xlim range is set.

    :param ax: matplotlib axes
    :param left: lower x-axis limit (if set to None, this limit will not be adjusted)
    :param right: upper x-axis limit (if set to None, this limit will not be adjusted)
    :return: None
    """
    if left is not None and ax.get_xlim()[0] > left:
        ax.set_xlim(left=left)
    if right is not None and ax.get_xlim()[1] < right:
        ax.set_xlim(right=right)


def extendYlim(ax, bottom=None, top=None):
    """
    Extends ylim of matplotlib axes.

    This function is useful for safely setting axes limits without the risk of accidentally cutting off data.
    Make sure to plot all data before calling this function so that the automatic ylim range is set.

    :param ax: matplotlib axes
    :param bottom: lower y-axis limit (if set to None, this limit will not be adjusted)
    :param top: upper y-axis limit (if set to None, this limit will not be adjusted)
    :return: None
    """
    if bottom is not None and ax.get_ylim()[0] > bottom:
        ax.set_ylim(bottom=bottom)
    if top is not None and ax.get_ylim()[1] < top:
        ax.set_ylim(top=top)
