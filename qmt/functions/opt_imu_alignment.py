# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

from itertools import product

from qmt.functions.quaternion import eulerAngles, quatFromAngleAxis, qmult, rotate, qinv, quatToGyrStrapdown, \
    quatFrom2Axes, headingInclinationAngle, quatProject
from qmt.functions.utils import timeVec, nanInterp, rms, nanUnwrap
from qmt.utils.misc import setDefaults

import numpy as np
import scipy.signal
import scipy.optimize

from qmt.utils.plot import AutoFigure


def _nanButterFiltfilt(signal, rate, fc, N=2):
    is1D = signal.ndim == 1
    if is1D:
        signal = signal[:, None]
    [b, a] = scipy.signal.butter(N, fc / (rate / 2))
    signal_nonan = nanInterp(signal)
    filtered = scipy.signal.filtfilt(b, a, signal_nonan, axis=0)
    nanind = np.any(np.isnan(signal), axis=1)
    filtered[nanind] = np.nan
    if is1D:
        filtered.shape = (filtered.shape[0], )
    return filtered


def _central2ndOrderFiniteDifference(data, rate=1):
    """
    Calculates central 2nd order finite difference.

    f''(x) = (f(x-h) - 2f(x) + f(x+h)) / h^2

    The first and last value are always 0. If the input is a matrix, the difference is calculated for each column
    independently.

    :param data: array containing the input data (NxM)
    :param rate: sampling frequency of the signal
    :return: 2nd order difference (NxM)
    """
    diff = np.zeros(data.shape)
    diff[1:-1] += data[:-2]  # f(x-h)
    diff[1:-1] += -2 * data[1:-1]  # -2f(x)
    diff[1:-1] += data[2:]  # f(x+h)

    return diff / ((1.0 / rate) ** 2)  # / h^2


def _detectMovement(opt_quat, rate, th=0.3, restLength=5.0, shift=0.5, fc1=10.0, fc2=0.2, plot=False):
    opt_gyr = quatToGyrStrapdown(opt_quat, rate)
    opt_gyr = _nanButterFiltfilt(opt_gyr, rate, fc1)
    opt_gyr_norm = np.linalg.norm(opt_gyr, axis=1)
    opt_gyr_norm_lp = _nanButterFiltfilt(opt_gyr_norm, rate, fc2)

    with np.errstate(invalid='ignore'):
        rest = opt_gyr_norm_lp < th
    rest_diff = np.diff(np.concatenate(([0.], rest, [0.])))
    starts = np.argwhere(rest_diff == 1)
    stops = np.argwhere(rest_diff == -1)
    assert len(starts) == len(stops)

    rest_sections = [(start.item() if start.item() == 0 else start.item() + int(shift * rate),
                      stop.item() if stop.item() == opt_gyr_norm_lp.shape[0] else stop.item() - int(shift * rate))
                     for start, stop in zip(starts, stops) if stop - start > restLength * rate]

    movement_ind = np.ones((opt_gyr_norm_lp.shape[0]), dtype=bool)
    for start, stop in rest_sections:
        movement_ind[start:stop] = 0

    if plot:
        import matplotlib.pyplot as plt
        t = timeVec(N=opt_gyr.shape[0], rate=rate)
        axes = plt.gcf().subplots(3, 1)
        axes[0].plot(t, np.rad2deg(opt_gyr), '-', '', lw=1.0)
        axes[0].set_title(f'opt_gyr [°/s, {opt_gyr.shape} @ {rate:.1f} Hz]')
        axes[0].grid()

        axes[1].plot(t, opt_gyr_norm, label='orig')
        axes[1].plot(t, opt_gyr_norm_lp, label='lp')
        axes[1].axhline(th, color='r')
        axes[1].set_ylim(0, 3 * th)
        axes[1].set_title('opt_gyr_norm before and after filtering, with threshold')
        axes[1].grid()
        axes[1].legend()

        axes[2].plot(t, movement_ind, 'r', lw=4)
        axes[2].set_title('movement_ind')
        axes[2].grid()

    return movement_ind, rest_sections


def _inclQuat(x0, x1):
    norm = np.sqrt(x0**2 + x1**2)
    return quatFromAngleAxis(norm, [x0, x1, 0])


def _quatFromX(q_init, x):
    assert x.shape == (3,), x.shape
    n = np.linalg.norm(x)
    return qmult(q_init, quatFromAngleAxis(n, x))


def _addEntry(alignment, name, quat, delta=None, verbose=False):
    assert quat.shape == (4,)
    euler_deg = np.round(np.rad2deg(eulerAngles(quat)), 3)
    total_deg = np.round(np.rad2deg(2 * np.arccos(np.abs(quat[0]))), 3)
    alignment[f'{name}'] = quat
    alignment[f'{name}_euler_deg'] = euler_deg
    alignment[f'{name}_total_deg'] = total_deg
    if delta is not None:
        alignment[f'{name}_delta'] = delta
        alignment[f'{name}_delta_deg'] = np.rad2deg(delta)
    if verbose:
        with np.printoptions(precision=3, suppress=True):
            deltatext = f', delta: {np.rad2deg(delta):.2f}°' if delta is not None else ''
            print(f'alignment: {name} zxy: {euler_deg}°, total: {total_deg:.3f}°{deltatext}')


def alignOptImu(imu_gyr, imu_acc, imu_mag, opt_quat, opt_pos, rate, params=None, names=None, debug=False, plot=False):
    """
    Determines sensor-to-segment and imu-to-opt reference frame alignment quaternions.

    See Appendix C of https://doi.org/10.3390/data6070072 for more information about (a subset of) this algorithm.

    :param imu_gyr: gyroscope measurements in rad/s
    :param imu_acc: accelerometer measurements in m/s²
    :param imu_mag: magnetometer measurements
    :param opt_quat: orientation from optical motion capture
    :param opt_pos: position from optical motion capture
    :param rate: sampling rate in seconds
    :param params: additional parameters, defaults: fullEOpt2EImu=True, accbias=True, r=0.5, fc_gyr=10.0, fc_grav=0.1,
        fc_acc=10.0, fc_pos=10.0, fc_mag=1.0, fast=False, init_quats=None, verbose=False
    :param names: names for different segments
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **alignment**: dict with alignment information
        - **debug**: dict with debug values (only if debug==True)
    """
    defaults = dict(fullEOpt2EImu=True, accbias=True, r=0.5, fc_gyr=10.0, fc_grav=0.1, fc_acc=10.0, fc_pos=10.0,
                    fc_mag=1.0, fast=False, init_quats=None, verbose=False)
    params = setDefaults(params, defaults)

    if isinstance(imu_gyr, np.ndarray):
        imu_gyr = [imu_gyr]
    if isinstance(imu_acc, np.ndarray):
        imu_acc = [imu_acc]
    if isinstance(imu_mag, np.ndarray):
        imu_mag = [imu_mag]
    if isinstance(opt_quat, np.ndarray):
        opt_quat = [opt_quat]
    if isinstance(opt_pos, np.ndarray):
        opt_pos = [opt_pos]
    N = len(imu_gyr)
    unique_names = list(set(names)) if names is not None else [None]
    M = len(unique_names)
    assert len(imu_acc) == N
    assert imu_mag is None or len(imu_mag) == N
    assert len(opt_quat) == N
    assert opt_pos is None or len(opt_pos) == N
    assert len(imu_gyr) == 1 or M <= N

    segInd = [unique_names.index(n) for n in names] if names is not None else list(range(N))
    # print('segInd', segInd)

    alignment = dict()
    if len(imu_gyr) > 1:
        alignment['names'] = names

    def imu2segName(i):
        return f'qImu2Seg_{unique_names[i]}' if M > 1 else 'qImu2Seg'

    def alignHeading(qEOpt2EImu):
        fc = params['fc_mag']
        imu_mag_EImu = []
        for i in range(N):
            quat = qmult(opt_quat[i], alignment[imu2segName(i)])
            quat = qmult(qEOpt2EImu, quat)
            imu_mag_EImu.append(_nanButterFiltfilt(rotate(quat, imu_mag[i]), rate, fc))
        imu_mag_EImu = np.vstack(imu_mag_EImu)

        norm = np.linalg.norm(imu_mag_EImu, axis=1)
        dip = np.arccos(imu_mag_EImu[:, 2]/norm)

        median_norm = np.nanmedian(norm)
        median_dip = np.nanmedian(dip)
        delta = nanUnwrap(np.arctan2(-imu_mag_EImu[:, 0], imu_mag_EImu[:, 1])[:, None])[:, 0]

        norm_th = 0.05
        dip_th = np.deg2rad(5)

        with np.errstate(invalid='ignore'):
            while True:
                valid = np.logical_and(
                    np.logical_and(norm/median_norm > (1 - norm_th), norm/median_norm < (1 + norm_th)),
                    np.logical_and(dip > median_dip - dip_th, dip < median_dip + dip_th))
                if np.mean(valid) >= 0.5:
                    break
                norm_th += 0.01
                dip_th += np.deg2rad(1)

        if norm_th != 0.05:
            print(f'warning: mag disturbance thresholds increased to norm_th={norm_th:.2f} and '
                  f'dip_th={np.rad2deg(dip_th):.1f}°, valid: {100 * np.mean(valid):.1f}%')

        # assert np.mean(valid) > 0.4, f'magnetic field seems to be too disturbed to estimate heading ' \
        #                              f'({100*np.mean(valid):.1f}% valid)'

        delta = np.nanmean(delta[valid])
        qEOpt2EImu = qmult(quatFromAngleAxis(-delta, [0, 0, 1]), qEOpt2EImu)

        return qEOpt2EImu, delta, norm_th, dip_th

    initQuats = params['init_quats']
    if params['fast']:
        assert initQuats is None
        initQuats = np.array([[1, 0, 0, 0]], float)
    if initQuats is None:
        initQuats = []
        for ax1, sign1, ax2, sign2 in product(range(3), (1, -1), range(3), (1, -1)):
            if ax1 == ax2:
                continue
            x = np.zeros(3)
            x[ax1] = sign1
            y = np.zeros(3)
            y[ax2] = sign2
            initQuats.append(quatFrom2Axes(x=x, y=y, z=[0, 0, 0]))
        initQuats = np.array(initQuats)

    if opt_pos is not None:
        fc_pos = params['fc_pos']
        fc_acc = params['fc_acc']
        opt_acc_nograv_EOpt = []
        for i in range(N):
            acc = _central2ndOrderFiniteDifference(_nanButterFiltfilt(opt_pos[i], rate, fc_pos), rate)
            acc = _nanButterFiltfilt(acc, rate, fc_acc)
            opt_acc_nograv_EOpt.append(acc)
        opt_acc_nograv_EOpt = np.vstack(opt_acc_nograv_EOpt)
    else:
        opt_acc_nograv_EOpt = None
        fc_acc = params['fc_grav']

    opt_gyr_lp = [quatToGyrStrapdown(opt_quat[i], rate) for i in range(N)]

    # low pass filter the rotation rates with a cutoff frequency of 10 Hz to increase robustness
    if params['fc_gyr'] > 0:
        opt_gyr_lp = [_nanButterFiltfilt(g, rate, params['fc_gyr']) for g in opt_gyr_lp]
        imu_gyr_lp = [_nanButterFiltfilt(imu_gyr[i], rate, params['fc_gyr']) for i in range(N)]
    else:
        imu_gyr_lp = imu_gyr

    def costFn(x, r, qImu2SegInit, getInfo=False):
        qImu2Seg = [_quatFromX(qImu2SegInit[i], x[3*i:3*i+3]) for i in range(M)]
        bias = [x[3*M+3*i:3*M+3*i+3] for i in range(N)]
        # print(x, qImu2Seg, bias)

        opt_gyr_rot = [rotate(qinv(qImu2Seg[segInd[i]]), opt_gyr_lp[i]) for i in range(N)]

        imu_acc_EOpt = []
        accbias = x[-4:-1] if params['accbias'] else np.zeros(3)
        for i in range(N):
            qImu2EOpt = qmult(opt_quat[i], qImu2Seg[segInd[i]])
            acc = rotate(qImu2EOpt, imu_acc[i]-accbias)
            acc = _nanButterFiltfilt(acc, rate, fc_acc)
            imu_acc_EOpt.append(acc)
        imu_acc_EOpt = np.vstack(imu_acc_EOpt)

        qEOpt2EImu = (_inclQuat(x[-3-3], x[-2-3]) if params['accbias'] else _inclQuat(x[-3], x[-2])) \
            if params['fullEOpt2EImu'] else _inclQuat(0, 0)
        g = x[-1]

        grav_EOpt = rotate(qinv(qEOpt2EImu), np.array([0, 0, g], float))

        gyr_diff = [imu_gyr_lp[i] - bias[i][None] - opt_gyr_rot[i] for i in range(N)]
        gyr_cost = rms(np.linalg.norm(np.vstack(gyr_diff), axis=1))

        if opt_acc_nograv_EOpt is None:
            acc_diff = imu_acc_EOpt - grav_EOpt
        else:
            acc_diff = imu_acc_EOpt - (opt_acc_nograv_EOpt + grav_EOpt)
        acc_cost = rms(np.linalg.norm(acc_diff, axis=1))

        cost = r*gyr_cost + (1-r)*acc_cost
        if getInfo:
            return dict(
                cost=cost,
                gyr_cost=gyr_cost,
                acc_cost=acc_cost,
                qImu2Seg=qImu2Seg,
                qEOpt2EImu=qEOpt2EImu,
                bias=bias,
                g=g,
                accbias=accbias,
            )
        return cost

    best = None
    log = []
    for qImu2SegInit in initQuats:
        r = params['r']
        assert 0.0 <= r <= 1.0
        args = (r, M*[qImu2SegInit])
        init = np.zeros((3*N+3*M+3+3 if params['accbias'] else 3*N+3*M+3) if params['fullEOpt2EImu'] else 3*N+3*M+1+3)
        init[-1] = 9.8
        res = scipy.optimize.minimize(costFn, init, args=args, method='BFGS')
        info = costFn(res.x, *args, getInfo=True)
        entry = [res.x, info, res.fun]
        log.append(entry)
        if best is None or best[-1] > res.fun:
            best = entry

    info = best[1]
    success_th = 1.01 * best[-1]
    success_rate = len([e for e in log if e[-1] <= success_th]) / len(log)
    if success_rate < 0.5:
        print(f'warning: low success_rate: {success_rate*100:.1f} %')
    info['sucess_rate'] = success_rate
    info['init_vals'] = len(log)

    if params['verbose']:
        print(f'success rate: {success_rate*100:.1f} %')

    for i in range(M):
        _addEntry(alignment, imu2segName(i), info['qImu2Seg'][i], verbose=params['verbose'])

    qEOpt2EImu = best[1]['qEOpt2EImu']
    delta = None
    if imu_mag is not None:
        _addEntry(alignment, 'qEOpt2EImu_6D', qEOpt2EImu, verbose=params['verbose'])
        qEOpt2EImu, delta, info['norm_th'], info['dip_th'] = alignHeading(qEOpt2EImu)
    _addEntry(alignment, 'qEOpt2EImu', qEOpt2EImu, delta, verbose=params['verbose'])

    info['opt_pos_available'] = opt_pos is not None
    del info['qImu2Seg']
    del info['qEOpt2EImu']
    alignment['info'] = info
    del params['verbose']
    alignment['params'] = params

    if debug or plot:
        from qmt import oriEstIMU
        imu_quat = [  # calculate some IMU orientation estimate to be able to calculate errors
            oriEstIMU(imu_gyr[i], imu_acc[i], None if imu_mag is None else imu_mag[i], dict(Ts=1/rate))
            for i in range(N)
        ]
        qRelEarth = qmult(np.vstack(imu_quat), qinv(np.vstack(opt_quat)))
        heading_error_before, inclination_error_before = headingInclinationAngle(qRelEarth)
        opt_quat_aligned = [qmult(qmult(qEOpt2EImu, opt_quat[i]), alignment[imu2segName(i)]) for i in range(N)]
        qRelEarth = qmult(np.vstack(imu_quat), qinv(np.vstack(opt_quat_aligned)))
        heading_error, inclination_error = headingInclinationAngle(qRelEarth)
        debugData = dict(
            alignment=alignment,
            heading_error_before=heading_error_before,
            inclination_error_before=inclination_error_before,
            heading_error=heading_error,
            inclination_error=inclination_error,
        )
        if plot:
            alignOptImu_debugPlot(debugData, plot)
        if debug:
            return alignment, debugData
    return alignment


def alignOptImu_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('alignOptImu'))

        ax1 = fig.add_subplot(211)
        ax1.plot(np.rad2deg(np.abs(debug['heading_error_before'])), label='heading')
        ax1.plot(np.rad2deg(debug['inclination_error_before']), label='inclination')
        ax1.set_title('heading and inclination error before alignment [°]')
        ax1.legend()
        ax1.grid()

        ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
        ax2.plot(np.rad2deg(np.abs(debug['heading_error'])), label='heading')
        ax2.plot(np.rad2deg(debug['inclination_error']), label='inclination')
        ax2.set_title('heading and inclination error after alignment [°]')
        ax2.legend()
        ax2.grid()

        fig.tight_layout()


def alignOptImuByMinimizingRmse(imu_quat, opt_quat, params=None, debug=False, plot=False):
    """
    Determines sensor-to-segment and imu-to-opt reference frame alignment quaternions that minimize the orientation
    difference.

    Warning: Use this function with care because systematic orientation estimation errors can easily be optimized away.
    The results are probably only trustworthy if the input orientations are quite accurate (slow movements, calibrated
    IMU, no magnetic disturbances, good time synchronization, ...) and if the performed movement contains a large
    variety of different orientations.

    :param imu_quat: IMU quaternion
    :param opt_quat: quaternion from optical motion capture
    :param params: optional parameters to control algorithm behavior, defaults: qImu2Seg=True, qEOpt2EImu=True,
        delta=True, initQuats=None, verbose=False
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **alignent**: dict with alignment quaternions
        - **debug**: dict with debug values (only if debug==True)
    """

    params = setDefaults(params, dict(qImu2Seg=True, qEOpt2EImu=True, delta=True, initQuats=None, verbose=False))
    if params['initQuats'] is not None:
        assert len(params['initQuats']) == 2
        assert params['initQuats'][0].shape == (4,)
        assert params['initQuats'][1].shape == (4,)

        qImu2Seg_init = params['initQuats'][0]
        proj = quatProject(params['initQuats'][1], [0, 0, 1])
        delta_init = proj['projAngle']
        qEOpt2EImu_6D_init = proj['resQuat']
        assert np.isclose(qEOpt2EImu_6D_init[-1], 0)
    else:
        qImu2Seg_init = np.array([1, 0, 0, 0], float)
        qEOpt2EImu_6D_init = np.array([1, 0, 0, 0], float)
        delta_init = 0.0

    def unpackX(x):
        if len(x) in (3, 5):
            qImu2Seg = _quatFromX(qImu2Seg_init, x[:3])
        else:
            qImu2Seg = np.array([1, 0, 0, 0], float)
        if len(x) in (2, 5):
            qEOpt2EImu_6D = qmult(qEOpt2EImu_6D_init, _inclQuat(x[-2], x[-1]))
        else:
            qEOpt2EImu_6D = np.array([1, 0, 0, 0], float)
        return qImu2Seg, qEOpt2EImu_6D

    def costFn(x):
        qImu2Seg, qEOpt2EImu_6D = unpackX(x)

        opt_quat_aligned = qmult(qmult(qEOpt2EImu_6D, opt_quat), qImu2Seg)
        qRelEarth = qmult(imu_quat, qinv(opt_quat_aligned))

        heading, inclination = headingInclinationAngle(qRelEarth)
        return rms(inclination)

    init = np.zeros((3 if params['qImu2Seg'] else 0) + (2 if params['qEOpt2EImu'] else 0), float)
    res = scipy.optimize.minimize(costFn, init, method='BFGS')
    inclRmse = res.fun
    qImu2Seg, qEOpt2EImu_6D = unpackX(res.x)

    def costFn(x):
        qEOpt2EImu = qmult(quatFromAngleAxis(x.item(), [0, 0, 1]), qEOpt2EImu_6D)

        opt_quat_aligned = qmult(qmult(qEOpt2EImu, opt_quat), qImu2Seg)
        qRelEarth = qmult(imu_quat, qinv(opt_quat_aligned))

        heading, inclination = headingInclinationAngle(qRelEarth)
        return rms(heading)

    res = scipy.optimize.minimize(costFn, np.array([delta_init], float), method='BFGS')
    headingRmse = res.fun
    delta = res.x.item()
    qEOpt2EImu = qmult(quatFromAngleAxis(delta, [0, 0, 1]), qEOpt2EImu_6D)

    alignment = {}
    _addEntry(alignment, 'qImu2Seg', qImu2Seg, verbose=params['verbose'])
    _addEntry(alignment, 'qEOpt2EImu_6D', qEOpt2EImu_6D, verbose=params['verbose'])
    _addEntry(alignment, 'qEOpt2EImu', qEOpt2EImu, delta, verbose=params['verbose'])
    alignment.update(dict(delta=delta, inclRmse=inclRmse, headingRmse=headingRmse))

    if debug or plot:
        qRelEarth = qmult(imu_quat, qinv(opt_quat))
        heading_error_before, inclination_error_before = headingInclinationAngle(qRelEarth)
        opt_quat_aligned = qmult(qmult(qEOpt2EImu, opt_quat), qImu2Seg)
        qRelEarth = qmult(imu_quat, qinv(opt_quat_aligned))
        heading_error, inclination_error = headingInclinationAngle(qRelEarth)
        debugData = dict(
            alignment=alignment,
            heading_error_before=heading_error_before,
            inclination_error_before=inclination_error_before,
            heading_error=heading_error,
            inclination_error=inclination_error,
        )
        if plot:
            alignOptImuByMinimizingRmse_debugPlot(debugData, plot)
        if debug:
            return alignment, debugData
    return alignment


def alignOptImuByMinimizingRmse_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('alignOptImuByMinimizingRmse'))

        ax1 = fig.add_subplot(211)
        ax1.plot(np.rad2deg(np.abs(debug['heading_error_before'])), label='heading')
        ax1.plot(np.rad2deg(debug['inclination_error_before']), label='inclination')
        ax1.set_title('heading and inclination error before alignment [°]')
        ax1.legend()
        ax1.grid()

        ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
        ax2.plot(np.rad2deg(np.abs(debug['heading_error'])), label='heading')
        ax2.plot(np.rad2deg(debug['inclination_error']), label='inclination')
        ax2.set_title('heading and inclination error after alignment [°]')
        ax2.legend()
        ax2.grid()

        fig.tight_layout()
