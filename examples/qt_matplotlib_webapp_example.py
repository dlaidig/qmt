#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
"""
This example shows how to combine Qt (with PySide2), Matplotlib, and the qmt Webapp framework in one application.

Note that the qasync event loop is used, which makes it possible to integrate async Python code with a Qt application.
"""
# for more information, see:
# https://github.com/CabbageDevelopment/qasync/blob/master/examples/aiohttp_fetch.py
# https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html

import asyncio
import signal

import numpy as np
from PySide2 import QtCore
from PySide2 import QtWidgets
import qasync
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import qmt

CONFIG = {  # configuration for /view/boxmodel to show a single joint
    'base': 'common',
    'defaults': {
        'scale': 3,
    },
    'segments': {
        'first': {
            'signal': 'quat1',
            'parent': 'root',
            'dimensions': [3, 12, 3],
            'color': 'red',
            'face': 2,
            'q_segment2sensor': [1, 0, -1, 0],
        },
        'second': {
            'signal': 'quat2',
            'parent': 'first',
            'dimensions': [3, 12, 3],
            'origin_rel': [0, 0.5, 0],
            'origin_abs': [0, 0.5 * 3, 0],
            'position_rel': [0, -0.5, -0.5],
            'position_abs': [0, -0.5 * 3, 0.5 * 3],
            'color': 'blue',
            'face': 2,
            'q_segment2sensor': [1, 0, 0, 0],
        },
    },
}


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Qt+Matplotlib+Webapp example')
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel(__doc__, self)
        layout.addWidget(label)

        groupBoxLayout = QtWidgets.QHBoxLayout()
        layout.addLayout(groupBoxLayout)

        plotGroupBox = QtWidgets.QGroupBox('Matplotlib', self)
        plotLayout = QtWidgets.QVBoxLayout()
        plotGroupBox.setLayout(plotLayout)
        groupBoxLayout.addWidget(plotGroupBox)

        self.canvas = FigureCanvas(Figure(figsize=(6, 5)))
        plotLayout.addWidget(NavigationToolbar(self.canvas, self))
        plotLayout.addWidget(self.canvas)

        webGroupBox = QtWidgets.QGroupBox('Webapp', self)
        webLayout = QtWidgets.QVBoxLayout()
        webGroupBox.setLayout(webLayout)
        groupBoxLayout.addWidget(webGroupBox)

        self.webLayout = webLayout

        self.webapp = qmt.Webapp('/view/boxmodel', show='widget', config=CONFIG)
        self.webapp.on('connected', self.addWebappWidget)  # this is called after the window is created
        self.webapp.run()

        self.button = QtWidgets.QPushButton('Generate New Random Data', self)
        layout.addWidget(self.button)

        self.axes = self.canvas.figure.subplots(2, 1, sharex=True)

        self.button.clicked.connect(self.generate)
        self.generate()

    def generate(self):
        # create interpolation between two random quaternions
        t = qmt.timeVec(T=3, Ts=0.04)
        qFirst = qmt.slerp(qmt.randomQuat(), qmt.randomQuat(), t/t[-1])
        qSecond = qmt.slerp(qmt.randomQuat(), qmt.randomQuat(), t/t[-1])
        qJoint = qmt.qrel(qFirst, qSecond)

        # plot quaternion and Euler angles with matplotlib
        self.axes[0].cla()
        self.axes[0].plot(t, qJoint)
        self.axes[0].grid()
        self.axes[0].set_ylim(-1, 1)
        self.axes[0].legend('wxyz')
        self.axes[0].set_title('joint orientation quaternion')
        self.axes[1].cla()
        self.axes[1].plot(t, np.rad2deg(qmt.eulerAngles(qJoint, 'zyx')))
        self.axes[1].grid()
        self.axes[1].set_ylim(-180, 180)
        self.axes[1].legend('zyx')
        self.axes[1].set_title('Euler angles of joint orientation [°]')
        self.axes[1].set_xlabel('time [s]')
        self.canvas.draw()

        # update the data displayed by the webapp
        self.webapp.data = dict(t=t, quat1=qFirst, quat2=qSecond)

    def closeEvent(self, event):
        super().closeEvent(event)
        self.webapp.viewer.prepareShutdown()

    async def addWebappWidget(self, *_):
        self.webLayout.addWidget(self.webapp.viewer.mainWindow)
        self.webapp.viewer.mainWindow.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.show()


async def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    future = asyncio.Future()
    qasync.QApplication.instance().aboutToQuit.connect(lambda: future.cancel())

    mainWindow = MainWindow()
    mainWindow.resize(QtCore.QSize(1200, 800))

    await future


if __name__ == "__main__":
    try:
        qmt.Webapp.initialize()
        qasync.run(main())
    except asyncio.CancelledError:
        pass
