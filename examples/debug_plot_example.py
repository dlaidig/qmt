#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import qmt
import numpy as np
import matplotlib.pyplot as plt

q1 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    ], float)

q2 = np.array([
    [0.5, 0.5, -0.5, -0.5],
    ], float)

# do not create debug plots
qmt.qmult(q1, q2)

# default: show plot right after creation
qmt.qmult(q1, q2, plot=True)

# create figures, but do not call plt.show() automatically
# also, make them a bit bigger (using units that are at least a little bit more reasonable than inches)
qmt.setupDebugPlots(mode='create', figsize_cm=(30, 20))
qmt.qmult(q1, q2, plot=True)
qmt.qmult(q1, q2, plot=True)
qmt.setupDebugPlots(figsize_cm=None)  # reset figsize to default
qmt.qmult(q1, q2, plot=True)
plt.show()

# save to PNG files with auto increasing number in file
qmt.setupDebugPlots(mode='save', filename='example_output/debug_plot_{i:03d}.png', figsize_cm=(29.7, 21))
qmt.qmult(q1, q2, plot=True)
qmt.qmult(q1, q2, plot=True)
qmt.qmult(q1, q2, plot=True)
qmt.qmult(q1, q2, plot=True)
qmt.qmult(q1, q2, plot=True)

# save to multipage PDF
qmt.setupDebugPlots(mode='pdfpages', filename='example_output/debug_plots.pdf', figsize_cm=(29.7, 21))
qmt.qmult(q1, q2, plot=True)
qmt.qmult(q1, q2, plot=True)
qmt.qmult(q1, q2, plot=True)
qmt.qmult(q1, q2, plot=True)
qmt.qmult(q1, q2, plot=True)
