.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
.. SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

Application Examples
####################

This page shows examples of how the ``qmt`` toolbox can be used to analyze inertial sensor data. You can find
the code files and the example data in the
`examples/ <https://github.com/dlaidig/qmt/tree/main/examples>`__ folder of the repository.

Full Body Motion Tracking -- Basic Example
==========================================

This example shows how the qmt framework can be used to perform offline full body 6D motion tracking and animate a
3D avatar with just a few lines of code. Sensor-to-segment orientations and an initial heading offset are determined
by manual tuning -- which can easily be done in the webapp.

:download:`examples/full_body_tracking_basic_example.py <../examples/full_body_tracking_basic_example.py>`

`(view on Github) <https://github.com/dlaidig/qmt/blob/main/examples/full_body_tracking_basic_example.py>`__

.. literalinclude:: ../examples/full_body_tracking_basic_example.py
    :language: python

Full Body Motion Tracking -- Advanced Example
=============================================

This example builds upon the previous example and replaces the manual tuning with constraint-based methods for
magnetometer-free motion tracking.

:download:`examples/full_body_tracking_advanced_example.py <../examples/full_body_tracking_advanced_example.py>`

`(view on Github) <https://github.com/dlaidig/qmt/blob/main/examples/full_body_tracking_advanced_example.py>`__

.. literalinclude:: ../examples/full_body_tracking_advanced_example.py
    :language: python

Full Body Motion Tracking Demo
==============================

The full body tracking demo shows how a custom webapp can be used to realize an interactive demo application. The
Python code performs full body 6D motion tracking (as in the two previous examples). The custom webapp allows the
user to change parameters. Those parameters are then sent to the Python code, which re-processes the data and sends
the resulting orientation back to the webapp for 3D visualization.

:download:`examples/full_body_tracking_demo.py <../examples/full_body_tracking_demo.py>`

`(view on Github) <https://github.com/dlaidig/qmt/blob/main/examples/full_body_tracking_demo.py>`__
