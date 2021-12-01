.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

Python utilities
################

**Data file handling**

.. autosummary::
    qmt.Struct
    qmt.toJson

**Plotting**

.. autosummary::
    qmt.setupDebugPlots
    qmt.extendXlim
    qmt.extendYlim

**Webapp framework**

.. autosummary::
    qmt.Webapp
    qmt.AbstractDataSource
    qmt.ProcessDataSource
    qmt.ProcessDataSourceConnection
    qmt.WebappProcessConnection
    qmt.ClockDataSource
    qmt.PlaybackDataSource
    qmt.DummyImuDataSource
    qmt.dataSourceFromJson

**Matlab interface**

.. create this table manually since autosummary:: does not work for qmt.matlab

===========================================  =============================================
:data:`qmt.matlab`                           Global object that provides access to Matlab.
:class:`qmt.utils.transplant.MatlabWrapper`  Provides access to the qmt Matlab functions.
===========================================  =============================================

**Data synchronization**

.. autosummary::
    qmt.SyncMapper

**Other utilities**

.. autosummary::
    qmt.setDefaults
    qmt.startStopInd

Data file handling
==================

.. autoclass:: qmt.Struct
   :members:

.. autofunction:: qmt.toJson

Plotting
========

.. autofunction:: qmt.setupDebugPlots

.. autofunction:: qmt.extendXlim

.. autofunction:: qmt.extendYlim

Webapp framework
================

.. autoclass:: qmt.Webapp
   :members:

.. autoclass:: qmt.AbstractDataSource
   :members:

.. autoclass:: qmt.ProcessDataSource
   :members:

.. autoclass:: qmt.ProcessDataSourceConnection
   :members:

.. autoclass:: qmt.WebappProcessConnection
   :members:

.. autoclass:: qmt.ClockDataSource
   :members:

.. autoclass:: qmt.PlaybackDataSource
   :members:

.. autoclass:: qmt.DummyImuDataSource
   :members:

.. autofunction:: qmt.dataSourceFromJson

Matlab interface
================

.. data:: qmt.matlab
   :annotation: = <qmt.utils.transplant.MatlabWrapper object>

    Global object that provides access to Matlab.

.. autoclass:: qmt.utils.transplant.MatlabWrapper
   :members:

Data synchronization
====================

.. autoclass:: qmt.SyncMapper
   :members:


Other utilities
===============

.. autofunction:: qmt.setDefaults

.. autofunction:: qmt.startStopInd
