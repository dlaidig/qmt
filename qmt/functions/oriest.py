# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
# SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import numpy as np
from qmt.utils.misc import setDefaults, startStopInd
from qmt.utils.plot import AutoFigure
from qmt.functions.utils import vecnorm, normalized
from qmt.functions.quaternion import eulerAngles, _plotQuatEuler, qmult, rotate


def _calcAccMagDisAngle(quat, acc, mag):
    accE = normalized(rotate(quat, acc))
    accDis = np.arccos(np.clip(accE[..., 2], -1, 1))
    if mag is not None:
        magE = rotate(quat, mag)
        magDis = np.abs(np.arctan2(magE[..., 0], magE[..., 1]))
    else:
        magDis = np.zeros(quat.shape[0])
    return np.column_stack([accDis, magDis])


def oriEstVQF(gyr, acc, mag=None, params=None, debug=False, plot=False):
    """
    VQF orientation estimation algorithm.

    See https://arxiv.org/abs/2203.17024 and https://github.com/dlaidig/vqf for more information about this algorithm.

    If potential real-time capability is not needed, use the offline version :meth:`qmt.oriEstOfflineVQF` for improved
    accuray. This algorithm is also available as an online data processing block: :class:`qmt.OriEstVQFBlock`.

    :param gyr: Nx3 array with gyroscope measurements [rad/s]
    :param acc: Nx3 array with accelerometer measurements [m/s^2]
    :param mag: Nx3 array with magnetometer measurements or None [any unit]
    :param params: A dictionary of parameters for orientation estimation. ``Ts`` is mandatory and specifies the sample
        time of measurement data in seconds. All other parameters are optional and the default values should be
        sufficient for most applications. See https://vqf.readthedocs.io/ for the documentation of the all
        possible parameters.
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: Nx4 quaternion output array
        - **debug**: dict with debug values (only if debug==True)
    """
    params = params.copy()
    Ts = params.pop('Ts')

    acc = np.ascontiguousarray(acc, dtype=float)
    gyr = np.ascontiguousarray(gyr, dtype=float)
    if mag is not None:
        mag = np.ascontiguousarray(mag, dtype=float)

    assert(acc.shape == gyr.shape)
    assert(acc.shape[1] == 3)
    assert(mag is None or mag.shape == acc.shape)

    from vqf import VQF
    vqf = VQF(Ts, **params)
    if debug or plot:
        out = vqf.updateBatchFullState(gyr, acc, mag)
    else:
        out = vqf.updateBatch(gyr, acc, mag)
    quat = out['quat6D'] if mag is None else out['quat9D']

    if debug or plot:
        disagreement = _calcAccMagDisAngle(quat, acc, mag)
        debugData = dict(
            quat=quat,
            quat_norm=vecnorm(quat),
            quat_euler=eulerAngles(quat),
            diagreement=disagreement
        )
        debugData.update(out)
        if plot:
            oriEstVQF_debugPlot(debugData, plot)
        if debug:
            return quat, debugData
        pass

    return quat


def oriEstVQF_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('oriEstVQF'))
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)

        style = _plotQuatEuler(ax1, ax2, debug, 'quat', 'output')

        ax3.plot(np.rad2deg(debug['bias']), style)
        ax3.plot(np.rad2deg(debug['biasSigma']), style+'C3')
        ax3.plot(np.nan, color='C2', lw=6, alpha=0.3)
        ax3.legend(['x', 'y', 'z', 'σ', 'rest'])
        for start, stop in zip(*startStopInd(debug['restDetected'])):
            ax3.axvspan(start, stop, color='C2', alpha=0.2)
        ax3.set_title(f'estimated bias and uncertainty σ in °/s, {debug["bias"].shape}')
        ax3.grid()

        ax4.plot(np.rad2deg(debug['diagreement']), style)
        ax4.plot(np.nan, color='C3', lw=6, alpha=0.3)
        ax4.legend(['acc', 'mag', 'magDist'])
        for start, stop in zip(*startStopInd(debug['magDistDetected'])):
            ax4.axvspan(start, stop, color='C3', alpha=0.2)
        ax4.set_title(f'diagreement in °, {debug["diagreement"].shape} ')
        ax4.grid()

        fig.tight_layout()


def oriEstBasicVQF(gyr, acc, mag=None, params=None, debug=False, plot=False):
    """
    Basic version of the VQF orientation estimation algorithm (no rest detection, no gyroscope bias estimation, no
    magnetic disturbance rejection).

    See https://arxiv.org/abs/2203.17024 and https://github.com/dlaidig/vqf for more information about this algorithm.

    :param gyr: Nx3 array with gyroscope measurements [rad/s]
    :param acc: Nx3 array with accelerometer measurements [m/s^2]
    :param mag: Nx3 array with magnetometer measurements or None [any unit]
    :param params: A dictionary of parameters for orientation estimation. ``Ts`` is mandatory and specifies the sample
        time of measurement data in seconds. The other two parameters are ``tauAcc`` and ``tauMag``, see
        https://vqf.readthedocs.io/ for more information.
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: Nx4 quaternion output array
        - **debug**: dict with debug values (only if debug==True)
    """
    params = params.copy()
    Ts = params.pop('Ts')

    acc = np.ascontiguousarray(acc, dtype=float)
    gyr = np.ascontiguousarray(gyr, dtype=float)
    if mag is not None:
        mag = np.ascontiguousarray(mag, dtype=float)

    assert(acc.shape == gyr.shape)
    assert(acc.shape[1] == 3)
    assert(mag is None or mag.shape == acc.shape)

    from vqf import BasicVQF
    vqf = BasicVQF(Ts, **params)
    if debug or plot:
        out = vqf.updateBatchFullState(gyr, acc, mag)
    else:
        out = vqf.updateBatch(gyr, acc, mag)
    quat = out['quat6D'] if mag is None else out['quat9D']

    if debug or plot:
        disagreement = _calcAccMagDisAngle(quat, acc, mag)
        debugData = dict(
            quat=quat,
            quat_norm=vecnorm(quat),
            quat_euler=eulerAngles(quat),
            diagreement=disagreement
        )
        debugData.update(out)
        if plot:
            oriEstBasicVQF_debugPlot(debugData, plot)
        if debug:
            return quat, debugData
        pass

    return quat


def oriEstBasicVQF_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('oriEstBasicVQF'))
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)

        style = _plotQuatEuler(ax1, ax2, debug, 'quat', 'output')

        ax3.axis('off')
        ax3.text(0.5, 0.5, 'oriEstBasicVQF does not estimate gyroscope bias', ha='center', transform=ax3.transAxes)

        ax4.plot(np.rad2deg(debug['diagreement']), style)
        ax4.set_title(f'diagreement in °, {debug["diagreement"].shape} ')
        ax4.legend(['acc', 'mag'])
        ax4.grid()
        fig.tight_layout()


def oriEstOfflineVQF(gyr, acc, mag=None, params=None, debug=False, plot=False):
    """
    Offline version of the VQF orientation estimation algorithm.

    See https://arxiv.org/abs/2203.17024 and https://github.com/dlaidig/vqf for more information about this algorithm.

    :param gyr: Nx3 array with gyroscope measurements [rad/s]
    :param acc: Nx3 array with accelerometer measurements [m/s^2]
    :param mag: Nx3 array with magnetometer measurements or None [any unit]
    :param params: A dictionary of parameters for orientation estimation. ``Ts`` is mandatory and specifies the sample
        time of measurement data in seconds. All other parameters are optional and the default values should be
        sufficient for most applications. See https://vqf.readthedocs.io/ for the documentation of the all
        possible parameters.
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: Nx4 quaternion output array
        - **debug**: dict with debug values (only if debug==True)
    """
    params = params.copy()
    Ts = params.pop('Ts')

    acc = np.ascontiguousarray(acc, dtype=float)
    gyr = np.ascontiguousarray(gyr, dtype=float)
    if mag is not None:
        mag = np.ascontiguousarray(mag, dtype=float)

    assert(acc.shape == gyr.shape)
    assert(acc.shape[1] == 3)
    assert(mag is None or mag.shape == acc.shape)

    from vqf import offlineVQF
    out = offlineVQF(gyr, acc, mag, Ts, params)
    quat = out['quat6D'] if mag is None else out['quat9D']

    if debug or plot:
        disagreement = _calcAccMagDisAngle(quat, acc, mag)
        debugData = dict(
            quat=quat,
            quat_norm=vecnorm(quat),
            quat_euler=eulerAngles(quat),
            diagreement=disagreement
        )
        debugData.update(out)
        if plot:
            oriEstVQF_debugPlot(debugData, plot)
        if debug:
            return quat, debugData
        pass

    return quat


def oriEstOfflineVQF_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('oriEstOfflineVQF'))
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)

        style = _plotQuatEuler(ax1, ax2, debug, 'quat', 'output')

        ax3.plot(np.rad2deg(debug['bias']), style)
        ax3.plot(np.rad2deg(debug['biasSigma']), style + 'C3')
        ax3.plot(np.nan, color='C2', lw=6, alpha=0.3)
        ax3.legend(['x', 'y', 'z', 'σ', 'rest'])
        for start, stop in zip(*startStopInd(debug['restDetected'])):
            ax3.axvspan(start, stop, color='C2', alpha=0.2)
        ax3.set_title(f'estimated bias and uncertainty σ in °/s, {debug["bias"].shape}')
        ax3.grid()

        ax4.plot(np.rad2deg(debug['diagreement']), style)
        ax4.plot(np.nan, color='C3', lw=6, alpha=0.3)
        ax4.legend(['acc', 'mag', 'magDist'])
        for start, stop in zip(*startStopInd(debug['magDistDetected'])):
            ax4.axvspan(start, stop, color='C3', alpha=0.2)
        ax4.set_title(f'diagreement in °, {debug["diagreement"].shape} ')
        ax4.grid()

        fig.tight_layout()


def oriEstMadgwick(gyr, acc, mag=None, params=None, debug=False, plot=False):
    """
    Madgwick's orientation estimation algorithm.

    See https://doi.org/10.1109/ICORR.2011.5975346 for more information about this algorithm. Based on the C++
    implementation by Sebastian Madgwick, available at https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/.

    This algorithm is also available as an online data processing block: :class:`qmt.OriEstMadgwickBlock`.

    :param gyr: Nx3 array with gyroscope measurements [rad/s]
    :param acc: Nx3 array with accelerometer measurements [m/s^2]
    :param mag: Nx3 array with magnetometer measurements or None [any unit]
    :param params: A dictionary of parameters for orientation estimation. ``Ts`` is mandatory, all other values are
        optional. Possible values:

        - **Ts**: sample time of measurement data in seconds
        - **beta**: algorithm gain
        - **initQuat**: 1x4 or (4,) array, quaternion of initial orientation

    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: Nx4 quaternion output array
        - **debug**: dict with debug values (only if debug==True)
    """
    defaults = dict(beta=0.1, initQuat=None)
    params = setDefaults(params, defaults, ['Ts'])

    acc = np.ascontiguousarray(acc, dtype=float)
    gyr = np.ascontiguousarray(gyr, dtype=float)
    if mag is not None:
        mag = np.ascontiguousarray(mag, dtype=float)

    assert(acc.shape == gyr.shape)
    assert(acc.shape[1] == 3)
    assert(mag is None or mag.shape == acc.shape)

    from qmt.cpp.madgwick import MadgwickAHRS
    obj = MadgwickAHRS(params['beta'], 1/params['Ts'])
    if params['initQuat'] is not None:
        initQuat = np.ascontiguousarray(params['initQuat'], dtype=float)
        assert initQuat.shape == (4,), 'invalid initQuat shape'
        obj.setState(initQuat)

    quat = obj.updateBatch(gyr, acc, mag)

    # adjust reference frame to ENU
    quat = qmult(np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], float), quat)

    if debug or plot:
        disagreement = _calcAccMagDisAngle(quat, acc, mag)
        debugData = dict(
            quat=quat,
            quat_norm=vecnorm(quat),
            quat_euler=eulerAngles(quat),
            diagreement=disagreement
        )
        if plot:
            oriEstMadgwick_debugPlot(debugData, plot)
        if debug:
            return quat, debugData

    return quat


def oriEstMadgwick_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('oriEstMadgwick'))
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)

        style = _plotQuatEuler(ax1, ax2, debug, 'quat', 'output')

        ax3.axis('off')
        ax3.text(0.5, 0.5, 'oriEstMadgwick does not estimate gyroscope bias', ha='center', transform=ax3.transAxes)

        ax4.plot(np.rad2deg(debug['diagreement']), style)
        ax4.set_title(f'diagreement in °, {debug["diagreement"].shape} ')
        ax4.legend(['acc', 'mag'])
        ax4.grid()
        fig.tight_layout()


def oriEstMahony(gyr, acc, mag=None, params=None, debug=False, plot=False):
    """
    Mahony's orientation estimation algorithm.

    See https://dx.doi.org/10.1109/TAC.2008.923738 for more information about this algorithm. Based on the C++
    implementation by Sebastian Madgwick, available at https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/.

    This algorithm is also available as an online data processing block: :class:`qmt.OriEstMahonyBlock`.

    :param gyr: Nx3 array with gyroscope measurements [rad/s]
    :param acc: Nx3 array with accelerometer measurements [m/s^2]
    :param mag: Nx3 array with magnetometer measurements or None [any unit]
    :param params: A dictionary of parameters for orientation estimation. ``Ts`` is mandatory, all other values are
        optional. Possible values:

        - **Ts**: sample time of measurement data in seconds
        - **Kp**: proportional gain
        - **Ki**: integral gain (for gyroscope bias estimation)
        - **initQuat**: 1x4 or (4,) array, quaternion of initial orientation
        - **initBias**: 1x3 or (3,) array, estimated bias of data

    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: Nx4 quaternion output array
        - **debug**: dict with debug values (only if debug==True)
    """
    defaults = dict(Kp=0.5, Ki=0.0, initQuat=None, initBias=None)
    params = setDefaults(params, defaults, ['Ts'])

    acc = np.ascontiguousarray(acc, dtype=float)
    gyr = np.ascontiguousarray(gyr, dtype=float)
    if mag is not None:
        mag = np.ascontiguousarray(mag, dtype=float)

    assert(acc.shape == gyr.shape)
    assert(acc.shape[1] == 3)
    assert(mag is None or mag.shape == acc.shape)

    from qmt.cpp.madgwick import MahonyAHRS
    obj = MahonyAHRS(params['Kp'], params['Ki'], 1/params['Ts'])
    if params['initQuat'] is not None or params['initBias'] is not None:
        if params['initQuat'] is None:
            initQuat = np.array([1.0, 0.0, 0.0, 0.0], float)
        else:
            initQuat = np.ascontiguousarray(params['initQuat'], dtype=float)
        if params['initBias'] is None:
            initBias = np.array([0.0, 0.0, 0.0], float)
        else:
            initBias = np.ascontiguousarray(params['initBias'], dtype=float)

        assert initQuat.shape == (4,), 'invalid initQuat shape'
        assert initBias.shape == (3,), 'invalid initBias shape'

        obj.setState(initQuat, initBias)

    quat, bias = obj.updateBatch(gyr, acc, mag)

    # adjust reference frame to ENU
    quat = qmult(np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], float), quat)

    if debug or plot:
        disagreement = _calcAccMagDisAngle(quat, acc, mag)
        debugData = dict(
            quat=quat,
            quat_norm=vecnorm(quat),
            quat_euler=eulerAngles(quat),
            diagreement=disagreement,
            bias=bias,
        )
        if plot:
            oriEstMahony_debugPlot(debugData, plot)
        if debug:
            return quat, debugData

    return quat


def oriEstMahony_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('oriEstMahony'))
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)

        style = _plotQuatEuler(ax1, ax2, debug, 'quat', 'output')

        ax3.plot(np.rad2deg(debug['bias']), style)
        ax3.legend('xyz')
        ax3.set_title(f'estimated bias in °/s, {debug["bias"].shape}')
        ax3.grid()

        ax4.plot(np.rad2deg(debug['diagreement']), style)
        ax4.set_title(f'diagreement in °, {debug["diagreement"].shape} ')
        ax4.legend(['acc', 'mag'])
        ax4.grid()
        fig.tight_layout()


def oriEstIMU(gyr, acc, mag=None, params=None, debug=False, plot=False):
    """
    OriEstIMU orientation estimation algorithm.

    See https://dx.doi.org/10.1016/j.ifacol.2017.08.1534 for more information about this algorithm.

    This algorithm is also available as an online data processing block: :class:`qmt.OriEstIMUBlock`.

    :param gyr: Nx3 array with gyroscope measurements [rad/s]
    :param acc: Nx3 array with accelerometer measurements [m/s^2]
    :param mag: Nx3 array with magnetometer measurements or None [any unit]
    :param params: A dictionary of parameters for orientation estimation. ``Ts`` is mandatory, all other values are
        optional. Possible values:

        - **Ts**: sample time of measurement data in seconds
        - **tauAcc**: time constants for acc correction (50% time) [must be >0], [in seconds]
        - **tauMag**: time constants for mag correction (50% time) [must be >0], [in seconds]
        - **zeta**: bias estimation strength [no unit]
        - **accRating**: enables raw rating of accelerometer, set to 0 to disable
        - **initQuat**: 1x4 or (4,) array, quaternion of initial orientation
        - **initBias**: 1x3 or (3,) array, estimated bias of data

    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: Nx4 quaternion output array
        - **debug**: dict with debug values (only if debug==True)
    """
    defaults = dict(tauAcc=1.0, tauMag=3.0, accRating=1.0, zeta=0.0, initQuat=None, initBias=None)
    params = setDefaults(params, defaults, ['Ts'])

    acc = np.ascontiguousarray(acc, dtype=float)
    gyr = np.ascontiguousarray(gyr, dtype=float)
    if mag is not None:
        mag = np.ascontiguousarray(mag, dtype=float)

    assert(acc.shape == gyr.shape)
    assert(acc.shape[1] == 3)
    assert(mag is None or mag.shape == acc.shape)

    from qmt.cpp.oriestimu import OriEstIMU
    oriEstImu = OriEstIMU(1/params['Ts'], params['tauAcc'], params['tauMag'], params['zeta'], params['accRating'])
    if params['initQuat'] is not None or params['initBias'] is not None:
        if params['initQuat'] is None:
            initQuat = np.array([1.0, 0.0, 0.0, 0.0], float)
        else:
            initQuat = np.ascontiguousarray(params['initQuat'], dtype=float)
        if params['initBias'] is None:
            initBias = np.array([0.0, 0.0, 0.0], float)
        else:
            initBias = np.ascontiguousarray(params['initBias'], dtype=float)

        assert initQuat.shape == (4,), 'invalid initQuat shape'
        assert initBias.shape == (3,), 'invalid initBias shape'

        oriEstImu.resetEstimation(initQuat, -initBias)

    quat, bias, disagreement = oriEstImu.updateBatch(acc, gyr, mag)

    if debug or plot:
        debugData = dict(
            quat=quat,
            quat_norm=vecnorm(quat),
            quat_euler=eulerAngles(quat),
            diagreement=disagreement,
            bias=-bias,
        )
        if plot:
            oriEstIMU_debugPlot(debugData, plot)
        if debug:
            return quat, debugData

    return quat


def oriEstIMU_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('oriEstIMU'))
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)

        style = _plotQuatEuler(ax1, ax2, debug, 'quat', 'output')

        ax3.plot(np.rad2deg(debug['bias']), style)
        ax3.legend('xyz')
        ax3.set_title(f'estimated bias in °/s, {debug["bias"].shape}')
        ax3.grid()

        ax4.plot(np.rad2deg(debug['diagreement']), style)
        ax4.set_title(f'diagreement in °, {debug["diagreement"].shape} ')
        ax4.legend(['acc', 'mag'])
        ax4.grid()
        fig.tight_layout()
