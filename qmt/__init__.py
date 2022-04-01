# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

from .utils.struct import Struct
from .utils.transplant import matlab
from .utils.plot import setupDebugPlots, extendXlim, extendYlim
from .utils.webapp import Webapp
from .utils.webapp_process import WebappProcessConnection
from .utils.datasource import AbstractDataSource, ProcessDataSource, ProcessDataSourceConnection, ClockDataSource
from .utils.datasource import PlaybackDataSource, DummyImuDataSource, dataSourceFromJson
from .utils.misc import toJson, setDefaults, startStopInd, parallel

from .functions.quaternion import qmult, qmult_debugPlot
from .functions.quaternion import qinv, qinv_debugPlot
from .functions.quaternion import qrel, qrel_debugPlot
from .functions.quaternion import rotate, rotate_debugPlot
from .functions.quaternion import quatTransform, quatTransform_debugPlot
from .functions.quaternion import quatProject, quatProject_debugPlot
from .functions.quaternion import eulerAngles, eulerAngles_debugPlot
from .functions.quaternion import quatFromEulerAngles, quatFromEulerAngles_debugPlot
from .functions.quaternion import quatAngle, quatAngle_debugPlot
from .functions.quaternion import quatAxis, quatAxis_debugPlot
from .functions.quaternion import quatFromAngleAxis, quatFromAngleAxis_debugPlot
from .functions.quaternion import quatToRotMat, quatToRotMat_debugPlot
from .functions.quaternion import quatFromRotMat, quatFromRotMat_debugPlot
from .functions.quaternion import quatToRotVec, quatToRotVec_debugPlot
from .functions.quaternion import quatFromRotVec, quatFromRotVec_debugPlot
from .functions.quaternion import quatToGyrStrapdown, quatToGyrStrapdown_debugPlot
from .functions.quaternion import quatFromGyrStrapdown, quatFromGyrStrapdown_debugPlot
from .functions.quaternion import quatFrom2Axes, quatFrom2Axes_debugPlot
from .functions.quaternion import quatFromVectorObservations, quatFromVectorObservations_debugPlot
from .functions.quaternion import headingInclinationAngle, headingInclinationAngle_debugPlot
from .functions.quaternion import slerp, slerp_debugPlot
from .functions.quaternion import quatInterp, quatInterp_debugPlot
from .functions.quaternion import randomQuat, randomQuat_debugPlot
from .functions.quaternion import averageQuat, averageQuat_debugPlot
from .functions.quaternion import quatUnwrap, quatUnwrap_debugPlot
from .functions.quaternion import posScalar, posScalar_debugPlot
from .functions.quaternion import deltaQuat, deltaQuat_debugPlot
from .functions.quaternion import addHeading, addHeading_debugPlot

from .functions.oriest import oriEstVQF, oriEstVQF_debugPlot
from .functions.oriest import oriEstBasicVQF, oriEstBasicVQF_debugPlot
from .functions.oriest import oriEstOfflineVQF, oriEstOfflineVQF_debugPlot
from .functions.oriest import oriEstMadgwick, oriEstMadgwick_debugPlot
from .functions.oriest import oriEstMahony, oriEstMahony_debugPlot
from .functions.oriest import oriEstIMU, oriEstIMU_debugPlot

from .functions.reset import resetAlignment, resetAlignment_debugPlot
from .functions.reset import resetHeading, resetHeading_debugPlot

from .functions.calibration import calibrateMagnetometerSimple, calibrateMagnetometerSimple_debugPlot

from .functions.utils import wrapToPi, wrapToPi_debugPlot
from .functions.utils import wrapTo2Pi, wrapTo2Pi_debugPlot
from .functions.utils import nanUnwrap, nanUnwrap_debugPlot
from .functions.utils import angleBetween2Vecs, angleBetween2Vecs_debugPlot
from .functions.utils import timeVec, timeVec_debugPlot
from .functions.utils import vecnorm, vecnorm_debugPlot
from .functions.utils import normalized, normalized_debugPlot
from .functions.utils import allUnitNorm, allUnitNorm_debugPlot
from .functions.utils import randomUnitVec, randomUnitVec_debugPlot
from .functions.utils import vecInterp, vecInterp_debugPlot
from .functions.utils import nanInterp, nanInterp_debugPlot
from .functions.utils import rms, rms_debugPlot

from .functions.sync import SyncMapper, syncOptImu, syncOptImu_debugPlot

from .functions.opt_imu_alignment import alignOptImu, alignOptImu_debugPlot
from .functions.opt_imu_alignment import alignOptImuByMinimizingRmse, alignOptImuByMinimizingRmse_debugPlot

from .functions.heading_correction import headingCorrection, headingCorrection_debugPlot
from .functions.heading_correction import removeHeadingDriftForStraightWalk_debugPlot, removeHeadingDriftForStraightWalk

from .functions.joint_axis_est_hinge_olsson import jointAxisEstHingeOlsson, jointAxisEstHingeOlsson_debugPlot

from .blocks.core import Block
from .blocks.oriest import OriEstVQFBlock, OriEstMadgwickBlock, OriEstMahonyBlock, OriEstIMUBlock
from .blocks.utils import LowpassFilterBlock
