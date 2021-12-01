#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import qmt
import numpy as np

# Perform quaternion multiplication using Python, using Matlab and create a debug plot using Python:

q1 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    ], float)

q2 = np.array([
    [0.5, 0.5, -0.5, -0.5],
    ], float)

q3 = qmt.qmult(q1, q2)
print('python:\n', q3)

# if matlab is not found in path:
# qmt.matlab.init(executable='/usr/local/MATLAB/R2017b/bin/matlab')

q3 = qmt.matlab.qmult(q1, q2, nargout=1)
print('matlab:\n', q3)

print('creating debug plot')
q3 = qmt.qmult(q1, q2, plot=True)

# Use the Matlab interface for other purposes:

m = qmt.matlab.instance
print('matlab version:\n', m.version())
print('the (probably) most inefficent way to generate an identity matrix:\n', m.eye(5))

# Create a .mat file using Struct()

data = qmt.Struct()
data['foo'] = 42
data['bar.baz'] = np.ones((100, 3))

print('data:', data)
print('keys:', data.keys())
print('leaf keys:', data.leafKeys())
print('all keys:', data.allKeys())

data.save('example_output/basic_example_script_output.mat', makedirs=True)

fileData = qmt.Struct.load('example_output/basic_example_script_output.mat')
print('data read from file:', fileData)
