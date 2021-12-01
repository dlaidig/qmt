.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

Guidelines for developers
#########################

Code guidelines for data processing functions/algorithms
========================================================

Data processing functions take input data, calculate something and return output data. This can range from simple
quaternion multiplication to complex analysis frameworks that take inertial sensor data from a full kinematic chain,
perform different calculations, and return a large number of output signals.

In general, there are three different kinds of implementations of data processing algorithms:

- online: processes data sample-by-sample, immediately returns output for the current sample
- vectorized: processes all data at once but could be implemented as an online algorithm
- offline: processes all data at once and is impossible to implement as an online algorithm

Function naming and parameters
------------------------------

Functions are named using camelCase, starting with a small letter. Names should be descriptive and be similar to
names used in existing functions. The same holds for names of input arguments and returned outputs.

The arguments of each function are first the inputs (i.e. data to be processed), then parameters (i.e. things that
influence how the data is processed), and then ``debug`` and ``plot`` flags which default to ``False``. Parameters can
have default values.

Example::

    def oriEstIMU(gyr, acc, mag, rate, tauAcc=1.0, tauMag=3.0, accRating=1.0, zeta=0.0, debug=False, plot=False):
        # calculates quat based on inputs gyr, acc and mag with parameters rate, tauAcc, tauMag, accRating and zeta
        if debug or plot:
            debugData = dict(diagreement=disagreement, bias=bias)
            if plot:
                oriEstIMU_debugPlot(debugData, plot)
            if debug:
                return quat, debugData
        return quat


If there are too many inputs, parameters, and/or outputs, they can be replaced by dicts called ``inputs``, ``params``
and ``outputs``. If an ``outputs`` dict exists, the ``debug`` dict is returned as part of it. See the following example
that also shows how to use default parameters::

    def oriEstIMU(inputs, params, debug=False):
        defaults = dict(tauAcc=1.0, tauMag=3.0, accRating=1.0, zeta=0.0)
        params.update((k, v) for k, v in defaults.items() if k not in params)
        params = setDefaults(params, default, ['rate'])  # rate is a non-optional parameter without a default value
        outputs = {}
        outputs['quat'] = # ...
        if debug or plot:
            debugData = dict(diagreement=disagreement, bias=bias)
            if plot:
                oriEstIMU_debugPlot(debugData, plot)
            if debug:
                outputs['debug'] = debugData
        return outputs

.. note:: The usage of dicts is independent for inputs, parameters, and outputs. For this example function, only
    ``params`` should probably be passed using a dict.

.. note:: In functions without an ``outputs`` dict, the debug flag changes the number of outputs! This makes simple
    functions much more convenient to use, but can lead to unexpected behavior.


Plot functions
--------------

Each data processing function must be accompanied by a corresponding plot function called ``$functionname_debugPlot``.
The function takes the ``debug`` dict of the processing function and an optional ``fig`` parameter containing a
matplotlib figure object (or a ``figs`` list if the function plots into multiple figures).

The plot function must use ``qmt.utils.plot.AutoFigure`` to create and show/save plots automatically as configured
with :func:`qmt.setupDebugPlots`. See the implementation of .:ref:qmult_plotFunction: for an example.


Online algorithms
-----------------

For online data processing, the :class:`qmt.Block` class should be used. An online data processing block typically has
a state and, given one input sample, calculates one output sample. Furthermore, processing can be adjusted with
parameters.

Other conventions
-----------------

- For time series, each row contains one sample. For example, a series of N quaternions is stored as an array with shape
  (N, 4) and not (4, N).
- Scalar time series (e.g. time vectors, single angles, ...) should be stored with shape (N,) and not (N, 1).
- Functions that can be applied to a single sample or a time series (e.g. :func:`qmt.qmult`) should apply both
  as input and return outputs of the same shape, e.g. ``qmult((4,), (4,))`` returns an output of shape (4,) while
  ``qmult((1, 4), (1, 4))`` returns an output of shape (1, 4). If possible, numpy broadcasting should be used for input
  data.
