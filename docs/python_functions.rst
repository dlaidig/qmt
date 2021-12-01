.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

.. _python_functions:

Python data processing functions
################################

**Basic quaternion functions**

*Basic operations*

.. autosummary::
    qmt.qmult
    qmt.qinv
    qmt.qrel
    qmt.rotate
    qmt.quatTransform
    qmt.quatProject

*Conversion from/to other representations*

.. autosummary::
    qmt.eulerAngles
    qmt.quatFromEulerAngles
    qmt.quatAngle
    qmt.quatAxis
    qmt.quatFromAngleAxis
    qmt.quatToRotMat
    qmt.quatFromRotMat
    qmt.quatToRotVec
    qmt.quatFromRotVec
    qmt.quatToGyrStrapdown
    qmt.quatFromGyrStrapdown
    qmt.quatFrom2Axes
    qmt.quatFromVectorObservations
    qmt.headingInclinationAngle

*Interpolation*

.. autosummary::
    qmt.slerp
    qmt.quatInterp

*Utilities*

.. autosummary::
    qmt.randomQuat
    qmt.averageQuat
    qmt.quatUnwrap
    qmt.posScalar
    qmt.deltaQuat
    qmt.addHeading


**Orientation estimation functions**

.. autosummary::
    qmt.madgwickAHRS
    qmt.mahonyAHRS
    qmt.oriEstIMU


**Reset alignment + heading reset**

.. autosummary::
    qmt.resetAlignment
    qmt.resetHeading


**Heading correction functions**

.. autosummary::
    qmt.headingCorrection
    qmt.removeHeadingDriftForStraightWalk

**Joint estimation functions**

.. autosummary::
    qmt.jointAxisEstHingeOlsson

**Optical mocap functions**

.. autosummary::
    qmt.syncOptImu
    qmt.alignOptImu
    qmt.alignOptImuByMinimizingRmse

**Utility functions**

*Angle utilities*

.. autosummary::
    qmt.wrapToPi
    qmt.wrapTo2Pi
    qmt.nanUnwrap
    qmt.angleBetween2Vecs

*Vector utilities*

.. autosummary::
    qmt.timeVec
    qmt.vecnorm
    qmt.normalized
    qmt.allUnitNorm
    qmt.randomUnitVec
    qmt.vecInterp
    qmt.nanInterp

*Other utilities*

.. autosummary::
    qmt.rms

Basic quaternion functions
==========================

Basic operations
----------------

.. autofunction:: qmt.qmult

.. autofunction:: qmt.qinv

.. autofunction:: qmt.qrel

.. autofunction:: qmt.rotate

.. autofunction:: qmt.quatTransform

.. autofunction:: qmt.quatProject

Conversion from/to other representations
----------------------------------------

.. autofunction:: qmt.eulerAngles

.. autofunction:: qmt.quatFromEulerAngles

.. autofunction:: qmt.quatAngle

.. autofunction:: qmt.quatAxis

.. autofunction:: qmt.quatFromAngleAxis

.. autofunction:: qmt.quatToRotMat

.. autofunction:: qmt.quatFromRotMat

.. autofunction:: qmt.quatToRotVec

.. autofunction:: qmt.quatFromRotVec

.. autofunction:: qmt.quatToGyrStrapdown

.. autofunction:: qmt.quatFromGyrStrapdown

.. autofunction:: qmt.quatFrom2Axes

.. autofunction:: qmt.quatFromVectorObservations

.. autofunction:: qmt.headingInclinationAngle

Interpolation
-------------

.. autofunction:: qmt.slerp

.. autofunction:: qmt.quatInterp

Utilities
---------

.. autofunction:: qmt.randomQuat

.. autofunction:: qmt.averageQuat

.. autofunction:: qmt.quatUnwrap

.. autofunction:: qmt.posScalar

.. autofunction:: qmt.deltaQuat

.. autofunction:: qmt.addHeading


Orientation estimation functions
================================

.. autofunction:: qmt.madgwickAHRS

.. autofunction:: qmt.mahonyAHRS

.. autofunction:: qmt.oriEstIMU


Reset alignment + heading reset
===============================

.. autofunction:: qmt.resetAlignment

.. autofunction:: qmt.resetHeading


Heading correction functions
============================

.. autofunction:: qmt.headingCorrection

.. autofunction:: qmt.removeHeadingDriftForStraightWalk


Joint estimation functions
==========================

.. autofunction:: qmt.jointAxisEstHingeOlsson


Optical mocap functions
=======================

.. autofunction:: qmt.syncOptImu

.. autofunction:: qmt.alignOptImu

.. autofunction:: qmt.alignOptImuByMinimizingRmse


Utility functions
=================

Angle utilities
---------------
.. autofunction:: qmt.wrapToPi

.. autofunction:: qmt.wrapTo2Pi

.. autofunction:: qmt.nanUnwrap

.. autofunction:: qmt.angleBetween2Vecs

Vector utilities
----------------

.. autofunction:: qmt.timeVec

.. autofunction:: qmt.vecnorm

.. autofunction:: qmt.normalized

.. autofunction:: qmt.allUnitNorm

.. autofunction:: qmt.randomUnitVec

.. autofunction:: qmt.vecInterp

.. autofunction:: qmt.nanInterp

Other utilities
---------------

.. autofunction:: qmt.rms

