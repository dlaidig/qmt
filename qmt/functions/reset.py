# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import numpy as np
from qmt.functions.quaternion import rotate, quatFrom2Axes, qmult, qinv, quatFromAngleAxis, eulerAngles
from qmt.functions.utils import wrapToPi, vecnorm
from qmt.utils.plot import AutoFigure
from qmt.utils.misc import startStopInd


def resetAlignment(q, reset, x=[1, 0, 0], xCs=0, y=[0, 1, 0], yCs=0, z=[0, 0, 0], zCs=0, exactAxis=None,
                   debug=False, plot=False):
    """
    Align multiple quaternions using a reset instant, e.g., from sensor orientations to segment orientations.

    A prerequisite for using this method is that during the reset, all M target (segment) orientations are aligned (and
    that the quaternion have the correct heading). It is then only necessary to know two different axes of the target
    coordinate system, either in one of the local coordinate systems of any of the M sensors or in the global coordinate
    system (i.e., if the common target coordinate system has a vertical axis during reset).

    If reset is set to 1, the all segment CSs are assumed to be aligned and new relative quaternions are calculated and
    stored.
    For values other than 1, the last stored value are applied.
    Two axes of the segment CS must be given in any sensor CS or in global CS. One of the 2 given axes is
    assumed to be exact and the second axis is transformed to be orthogonal to the exact axis.
    Alternatively, both axes can be adjusted equally. One of the three axes has to be [0 0 0].
    The parameters xCs, yCs, zCs describe if the provided axes are given in any of the sensor CS (from 0 to M-1) or in
    global (-1) coordinates.

    :param q: (M, N, 4) quaternion input array, M is the number of sensors, N is the number of samples
    :param reset:  (N,) boolean reset input array
    :param x: (N, 3) or (3,) x-axis coordinates
    :param xCs: coordinate system for x, valid values: -1..M-1
    :param y: (N, 3) or (3,) y-axis coordinates
    :param yCs: coordinate system for y, valid values: -1..M-1
    :param z: (N, 3) or (3,) z-axis coordinates
    :param zCs: coordinate system for z, valid values: -1..M-1
    :param exactAxis: axis that is assumed to be exact ('x', 'y', 'z', or None to adjust both equally)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: (M, N, 4) quaternion output array
        - **debug**: dict with debug values (only if debug==True)
    """
    q = np.asarray(q, float)
    reset = np.asarray(reset, bool)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    M = q.shape[0]
    N = q.shape[1]

    # check the dimensions
    assert q.shape == (M, N, 4)
    assert reset.shape == (N,)
    assert x.shape == (N, 3) or x.shape == (3,), f'invalid x shape: {x.shape}'
    assert y.shape == (N, 3) or y.shape == (3,), f'invalid y shape: {y.shape}'
    assert z.shape == (N, 3) or z.shape == (3,), f'invalid z shape: {z.shape}'
    assert isinstance(xCs, int), 'xCs should be a integer'
    assert isinstance(yCs, int), 'yCs should be a integer'
    assert isinstance(zCs, int), 'zCs should be a integer'
    if exactAxis == '' or exactAxis == 'None':
        exactAxis = None

    if min(xCs, yCs, zCs) < -1 or max(xCs, yCs, zCs) > M-1:
        raise RuntimeError('parameters xCs, yCs and zCs must be in range -1..M-1')

    # express all axes in the global CS
    xrot = x
    yrot = y
    zrot = z

    if xCs != -1:
        xrot = rotate(q[xCs], x)
    if yCs != -1:
        yrot = rotate(q[yCs], y)
    if zCs != -1:
        zrot = rotate(q[zCs], z)

    # obtain quaternion describing segment CS orientation relative to global CS
    qSegToGlobal = quatFrom2Axes(xrot, yrot, zrot, exactAxis)

    # R = np.sum(reset)
    qReset = np.zeros((M, np.sum(reset), 4))
    # determine index for a and b that corresponds to the previous reset time (-1 before first reset)
    ind = np.cumsum(reset) - 1

    # determine relative orientations at the reset instants
    qReset = qmult(qinv(q[:, reset]), qSegToGlobal[reset])

    # determine relative orientations from previous reset ([1 0 0 0] at the beginning)
    qRel = np.zeros((M, N, 4))
    qRel[:, :, 0] = 1  # initialize with [1 0 0 0]
    qRel[:, ind >= 0] = qReset[:, ind[ind >= 0]]

    output = qmult(q, qRel)
    assert output.shape == qRel.shape == q.shape

    if debug or plot:
        debugData = dict(
            q=q,
            q_norm=vecnorm(q),
            q_euler=eulerAngles(q),
            qSegToGlobal=qSegToGlobal,
            qReset=qReset,
            qRel=qRel,
            qRel_norm=vecnorm(qRel),
            qRel_euler=eulerAngles(qRel),
            output=output,
            output_norm=vecnorm(output),
            output_euler=eulerAngles(output),
            reset=reset,
            M=M,
            N=N
        )
        if plot:
            resetAlignment_debugPlot(debugData, plot)
        if debug:
            return output, debugData
    return output


def resetAlignment_debugPlot(debug, fig=None):
    from qmt.functions.quaternion import _plotQuatEuler
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('resetAlignment'))
        output = debug['output']
        output_norm = debug['output_norm']
        output_euler = debug['output_euler']
        qRel = debug['qRel']
        qRel_norm = debug['qRel_norm']
        qRel_euler = debug['qRel_euler']
        reset = debug['reset']
        M = debug['M']

        ax = fig.subplots(M, 3, sharex=True).reshape(M, 3)

        for i in range(M):
            dbg = {
                f'output[{i}]': output[i],
                f'output[{i}]_norm': output_norm[i],
                f'output[{i}]_euler': output_euler[i],
                f'qRel[{i}]': qRel[i],
                f'qRel[{i}]_norm': qRel_norm[i],
                f'qRel[{i}]_euler': qRel_euler[i],
            }
            _plotQuatEuler(ax[i, 0], ax[i, 1], dbg, f'output[{i}]')
            _plotQuatEuler(ax[i, 2], None, dbg, f'qRel[{i}]', 'sensor-to-segment')

        starts, stops = startStopInd(reset)
        for start, stop in zip(starts, stops):
            for i in range(M):
                for j in range(3):
                    ax[i, j].axvspan(start, stop, color='C8', lw=1.5, alpha=0.8)

        fig.tight_layout()


def resetHeading(q, reset, base=0, deltaOffset=0, debug=False, plot=False):
    """
    Adjust the heading of a orientation quaternion so that the relative angle to the other input quaternion becomes
    as small as possible.

    If reset is set to 1, the input quaternions are assumed to be aligned and the heading difference is calculated and
    stored.
    For values other than 1, the last stored value are applied.
    The heading of the base quaternion is assumed to be correct and therefore the other quaternion will be adjusted by
    the calculated heading difference. In addition, all quaternions are adjusted by the input angle deltaOffset (heading
    offset).

    :param q: (M, N, 4) quaternion input array, M is the number of sensors, N is the number of samples
    :param reset: (N,) boolean reset input array
    :param deltaOffset: heading offset array in rad, (N,) or scalar
    :param base: index of the quaternion that is not adjusted, valid value is a natural number from 0 to M-1
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: (M, N, 4) quaternion output array
        - **debug**: dict with debug values (only if debug==True)
    """
    q = np.asarray(q, float)
    reset = np.asarray(reset, bool)
    deltaOffset = np.asarray(deltaOffset, float)

    M = q.shape[0]
    N = q.shape[1]

    if M < 2:
        raise RuntimeError('the number of sensors should be larger than 1')

    assert q.shape == (M, N, 4)
    assert reset.shape == (N,)
    assert deltaOffset.shape == (N,) or deltaOffset.shape == tuple()
    assert isinstance(base, int), 'base should be a integer'
    if base < 0 or base > M-1:
        raise RuntimeError('parameter base must be in range 0..M-1')

    R = np.sum(reset)
    a = np.zeros((M, R))
    b = np.zeros((M, R))
    # determine index for a and b that corresponds to the previous reset time (-1 before first reset)
    ind = np.cumsum(reset) - 1

    # calculate heading angle that minimizes the residual rotation during the reset instants
    a = q[base, reset, 0] * q[:, reset, 0] + q[base, reset, 1] * q[:, reset, 1] + q[base, reset, 2] * q[:, reset, 2] + \
        q[base, reset, 3] * q[:, reset, 3]
    b = q[base, reset, 3] * q[:, reset, 0] + q[base, reset, 2] * q[:, reset, 1] - q[base, reset, 1] * q[:, reset, 2] - \
        q[base, reset, 0] * q[:, reset, 3]
    deltaDiff = 2 * np.arctan2(b, a)

    # determine heading angle from previous reset (zero at the beginning)
    delta = np.zeros((M, N))
    delta[:, ind >= 0] = deltaDiff[:, ind[ind >= 0]]
    delta[base] = 0
    delta += deltaOffset  # add heading offset
    delta = wrapToPi(delta)

    output = qmult(quatFromAngleAxis(delta, [0, 0, 1]), q)

    if debug or plot:
        debugData = dict(
            q=q,
            q_norm=vecnorm(q),
            q_euler=eulerAngles(q),
            output=output,
            output_norm=vecnorm(output),
            output_euler=eulerAngles(output),
            reset=reset,
            base=base,
            deltaOffset=deltaOffset,
            delta=delta,
            M=M,
            N=N,
        )
        if plot:
            resetHeading_debugPlot(debugData, plot)
        if debug:
            return output, debugData
    return output


def resetHeading_debugPlot(debug, fig=None):
    from qmt.functions.quaternion import _plotQuatEuler
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('resetHeading'))
        output = debug['output']
        output_norm = debug['output_norm']
        output_euler = debug['output_euler']
        delta = debug['delta']
        reset = debug['reset']
        M = debug['M']

        ax = fig.subplots(M, 3, sharex=True)

        for i in range(M):
            dbg = {
                f'output[{i}]': output[i],
                f'output[{i}]_norm': output_norm[i],
                f'output[{i}]_euler': output_euler[i]
            }
            style = _plotQuatEuler(ax[i, 0], ax[i, 1], dbg, f'output[{i}]')

            ax[i, 2].plot(np.rad2deg(delta[i]), style)
            ax[i, 2].set_title(f'heading angle delta[{i}], {delta[i].shape}')
            ax[i, 2].grid()

        starts, stops = startStopInd(reset)
        for start, stop in zip(starts, stops):
            for i in range(M):
                for j in range(3):
                    ax[i, j].axvspan(start, stop, color='C8', lw=1.5, alpha=0.8)

        fig.tight_layout()
