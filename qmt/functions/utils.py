# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
import math

import numpy as np

from qmt.utils.plot import AutoFigure


def wrapToPi(angles, debug=False, plot=False):
    """
    Wraps angles to interval -π... π by adding/subtracting multiples of 2π.

    :param angles: input angles in rad, numpy array or scalar
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: output angles with same shape as input angles
        - **debugData**: debug: dict with debug values (only if debug==True)
    """
    angles = np.asarray(angles, float)
    with np.errstate(invalid='ignore'):
        output = (angles + np.pi) % (2 * np.pi) - np.pi

    if debug or plot:
        debugData = dict(angles=angles, output=output)
        if plot:
            wrapToPi_debugPlot(debugData, plot)
        if debug:
            return output, debugData
    return output


def wrapToPi_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('wrapToPi'))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        angles = debug['angles']
        output = debug['output']
        shape = angles.shape if angles.ndim < 2 else (-1, angles.shape[-1])
        style = '.-' if (angles.shape[0] if angles.ndim > 0 else angles.size) <= 100 else '-'

        ax1.set_title(f'input angle in °, {angles.shape}')
        ax1.plot(np.rad2deg(angles.reshape(shape)), style)

        ax2.set_title(f'output angle in °, {output.shape}')
        ax2.plot(np.rad2deg(output.reshape(shape)), style)
        ax2.plot(np.rad2deg(angles.reshape(shape)), style, color='k', alpha=0.3, lw=1)

        for ax in (ax1, ax2):
            ax.grid()
        fig.tight_layout()


def wrapTo2Pi(angles, debug=False, plot=False):
    """
    Wraps angles to interval 0...2π by adding/subtracting multiples of 2π.

    :param angles: input angles in rad, numpy array or scalar
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: output angles with same shape as input angles
        - **debugData**: debug: dict with debug values (only if debug==True)
    """
    angles = np.asarray(angles, float)
    with np.errstate(invalid='ignore'):
        positiveInput = angles > 0
        output = np.mod(angles, 2 * np.pi)
    if output.ndim == 0:  # does not support item assignment
        if output == 0 and positiveInput:
            output = np.array(2 * np.pi, float)
    else:
        output[np.logical_and((output == 0), positiveInput)] = 2 * np.pi

    if debug or plot:
        debugData = dict(angles=angles, output=output)
        if plot:
            wrapTo2Pi_debugPlot(debugData, plot)
        if debug:
            return output, debugData
    return output


def wrapTo2Pi_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('wrapTo2Pi'))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        angles = debug['angles']
        output = debug['output']
        shape = angles.shape if angles.ndim < 2 else (-1, angles.shape[-1])
        style = '.-' if (angles.shape[0] if angles.ndim > 0 else angles.size) <= 100 else '-'

        ax1.set_title(f'input angle in °, {angles.shape}')
        ax1.plot(np.rad2deg(angles.reshape(shape)), style)

        ax2.set_title(f'output angle in °, {output.shape}')
        ax2.plot(np.rad2deg(output.reshape(shape)), style)
        ax2.plot(np.rad2deg(angles.reshape(shape)), style, color='k', alpha=0.3, lw=1)

        for ax in (ax1, ax2):
            ax.grid()
        fig.tight_layout()


def nanUnwrap(angle, debug=False, plot=False):
    """
    Unwraps a signal that might contain NaNs. The angle is also shifted so that the mean is in the range -pi...pi.

    The input can be a 1D or 2D array. Unwrapping is performed along the first axis. In the 2D case, it is expected
    that all angles in the same row will be NaN at the same time.

    :param angle: input angle signal in rad, shape (N, 3) or (N,)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: unwrapped output angles with same shape as input angles
        - **debugData**: debug: dict with debug values (only if debug==True)
    """
    angle = np.asarray(angle, float)
    assert angle.ndim <= 2
    is1D = angle.ndim == 1
    if is1D:
        angle = angle.reshape(-1, 1)

    output = np.copy(angle)
    ind = ~np.isnan(output[:, 0])
    output[ind] = np.unwrap(output[ind], axis=0)
    for i in range(output.shape[1]):
        while np.nanmean(output[:, i]) > np.pi:
            output[:, i] -= 2 * np.pi
        while np.nanmean(output[:, i]) < -np.pi:
            output[:, i] += 2 * np.pi

    if is1D:
        angle = angle.flatten()
        output = output.flatten()

    if debug or plot:
        debugData = dict(angle=angle, output=output)
        if plot:
            nanUnwrap_debugPlot(debugData, plot)
        if debug:
            return output, debugData
    return output


def nanUnwrap_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('nanUnwrap'))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        angle = debug['angle']
        output = debug['output']
        shape = angle.shape if angle.ndim < 2 else (-1, angle.shape[-1])
        style = '.-' if (angle.shape[0] if angle.ndim > 0 else angle.size) <= 100 else '-'

        ax1.set_title(f'input angle in °, {angle.shape}')
        ax1.plot(np.rad2deg(angle.reshape(shape)), style)

        ax2.set_title(f'output angle in °, {output.shape}')
        ax2.plot(np.rad2deg(output.reshape(shape)), style)
        ax2.plot(np.rad2deg(angle.reshape(shape)), style, color='k', alpha=0.3, lw=1)

        for ax in (ax1, ax2):
            ax.grid()
        fig.tight_layout()


def angleBetween2Vecs(vec1, vec2, debug=False, plot=False):
    """
    Calculates angle between two 3D vectors in rad.

    Multiples vectors can be passed with an array with the last dimension having length 3, and numpy broadcasting is
    supported.

    >>> qmt.angleBetween2Vecs([1, 0, 0], [0, 1, 0])
    angle = array([1.57079633])

    :param vec1: first input vector, (..., 3)
    :param vec2: second input vector, (..., 3)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **angle**: angle output array, shape (...,)
        - **debug**: dict with debug values (only if debug==True)
    """

    vec1 = np.asarray(vec1, float)
    vec2 = np.asarray(vec2, float)

    angle = np.arccos(np.sum(normalized(vec1)*normalized(vec2), axis=-1))

    if debug or plot:
        debugData = dict(
            vec1=vec1,
            vec2=vec2,
            angle=angle,
        )
        if plot:
            angleBetween2Vecs_debugPlot(debugData, plot)
        if debug:
            return angle, debugData

    return angle


def angleBetween2Vecs_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('angleBeween2Vecs'))
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)

        vec1 = debug['vec1']
        vec2 = debug['vec2']
        angle = debug['angle']
        style = '.-' if angle.size <= 100 else '-'

        ax1.plot(vec1.reshape(-1, 3), style)
        ax1.set_title(f'first input vector, {debug["vec1"].shape}')
        ax1.legend('xyz')

        ax2.plot(vec2.reshape(-1, 3), style)
        ax2.set_title(f'second input vector, {debug["vec2"].shape}')
        ax2.legend('xyz')

        ax3.plot(np.rad2deg(angle.flatten()), '.-')
        ax3.set_title(f'angle in degree between the vectors in °, {debug["angle"].shape}')

        for ax in (ax1, ax2, ax3):
            ax.grid()
        fig.tight_layout()


def timeVec(N=None, T=None, rate=None, Ts=None, debug=False, plot=False):
    """
    Creates a time vector with a fixed sampling rate based on two parameters.

    Valid combinations of parameters:

    - N and rate
    - N and Ts
    - T and rate
    - T and Ts
    - N and T

    >>> qmt.timeVec(N=10, rate=100)
    array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    >>> qmt.timeVec(N=10, Ts=0.01)
    array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    >>> qmt.timeVec(T=0.05, Ts=0.01)
    array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05])

    :param N: number of samples
    :param T: end time (the start time is always 0)
    :param rate: sampling rate
    :param Ts: sampling time, i.e. 1/rate
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **t**: time vector as (N,) array
        - **debug**: dict with debug values (only if debug==True)
    """
    if N is not None and rate is not None:
        assert T is None and Ts is None
        t = np.arange(0, N, 1) / rate
    elif N is not None and Ts is not None:
        assert T is None and rate is None
        t = np.arange(0, N, 1) * Ts
    elif T is not None and rate is not None:
        assert N is None and Ts is None
        N = math.floor(T * rate) + 1
        t = np.arange(0, N, 1) / rate
    elif T is not None and Ts is not None:
        assert N is None and rate is None
        N = math.floor(T / Ts) + 1
        t = np.arange(0, N, 1) * Ts
    elif N is not None and T is not None:
        assert rate is None and Ts is None
        t = np.linspace(0, T, N)
    else:
        raise RuntimeError('invalid combination of parameters')

    if debug or plot:
        debugData = dict(t=t, N=N, T=T, rate=rate, Ts=Ts)
        if plot:
            timeVec_debugPlot(debugData, plot)
        if debug:
            return t, debugData
    return t


def timeVec_debugPlot(debug, fig=None):
    t = debug['t']
    N = debug['N']
    T = debug['T']
    rate = debug['rate']
    Ts = debug['Ts']

    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('timeVec'))
        ax = fig.add_subplot(111)
        ax.plot(t, '.-')
        ax.set_title(f'output time vector, {t.shape}')
        text = f'N={N}\nT={T}\nrate={rate}\nTs={Ts}\nt[0]={t[0]}\nt[1]={t[1]}\nt[{N-1}]={t[N-1]}'
        ax.text(0.05, 0.95, text, va='top', transform=ax.transAxes)
        ax.grid()
        fig.tight_layout()


def vecnorm(vec, debug=False, plot=False):
    """
    Calculates the norm along the last axis.

    :param vec: input array, shape (..., M)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **norm**: norm output array, shape (...,)
        - **debug**: dict with debug values (only if debug==True)
    """
    vec = np.asarray(vec, float)
    norm = np.linalg.norm(vec, axis=-1)
    if debug or plot:
        debugData = dict(
            vec=vec,
            norm=norm,
        )
        if plot:
            vecnorm_debugPlot(debugData, plot)
        if debug:
            return norm, debugData
    return norm


def vecnorm_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('vecnorm'))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        vec = debug['vec']
        norm = debug['norm']
        style = '.-' if norm.size <= 100 else '-'

        ax1.set_title(f'input vector, {vec.shape}')
        ax1.plot(vec.reshape(-1, vec.shape[-1]), style)

        ax2.set_title(f'output norm, {norm.shape}')
        ax2.plot(norm.flatten(), style)

        for ax in (ax1, ax2):
            ax.grid()
        fig.tight_layout()


def normalized(vec, debug=False, plot=False):
    """
    Divides each vector (along the last axis) by its norm.

    >>> qmt.normalized([1, 1, 1])
    v_norm = array([0.57735027, 0.57735027, 0.57735027])

    :param vec: input array, shape (..., M)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **out**: normalized output array, shape (..., M)
        - **debug**: dict with debug values (only if debug==True)
    """

    vec = np.asarray(vec, float)
    out = vec / np.linalg.norm(vec, axis=-1)[..., None]

    if debug or plot:
        debugData = dict(
            vec=vec,
            vec_norm=vecnorm(vec),
            out=out,
            out_norm=vecnorm(out),
        )
        if plot:
            normalized_debugPlot(debugData, plot)
        if debug:
            return out, debugData

    return out


def normalized_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('normalized'))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        vec = debug['vec']
        out = debug['out']
        style = '.-' if vec.size <= 3*100 else '-'

        ax1.set_title(f'input vector (gray: norm), {vec.shape}')
        ax1.plot(vec.reshape(-1, vec.shape[-1]), style)
        ax1.plot(debug['vec_norm'].flatten(), style, color='k', alpha=0.5, lw=1)
        ax1.grid()

        ax2.set_title(f'normalized output vector (gray: norm), {out.shape}')
        ax2.plot(out.reshape(-1, out.shape[-1]), style)
        ax2.plot(debug['out_norm'].flatten(), style, color='k', alpha=0.5, lw=1)
        ax2.grid()
        fig.tight_layout()


def allUnitNorm(vec, debug=False, plot=False):
    """
    Checks if all elements have unit norm.

    Calculation of the norm is performed along the last axis (see :func:`qmt.vecnorm`) and True is returned if all
    entries have a norm that is close to one. Entries with NaNs are ignored.

    :param vec: input vector or quaternion array
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: True if all elements have unit norm, else False
        - **debug**: dict with debug values (only if debug==True)
    """
    norm = vecnorm(vec)
    res = np.allclose(norm[~np.isnan(norm)], 1)

    if debug or plot:
        isnan = np.isnan(norm)
        isclose = np.isclose(norm, 1)
        debugData = dict(
            vec=vec,
            norm=norm,
            isnan=isnan,
            isclose=isclose,
            notclose=~(isnan | isclose),
            res=res,
        )
        if plot:
            allUnitNorm_debugPlot(debugData, plot)
        if debug:
            return res, debugData
    return res


def allUnitNorm_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('allUnitNorm'))
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)

        N = debug['norm'].size
        style = '.-' if N <= 100 else '-'

        vec = debug['vec'].reshape((N, -1))
        norm = debug['norm'].flatten()

        ax1.plot(vec, style)
        ax1.plot(norm, style, color='k', lw=1, label='norm')
        ax1.set_title(f'input vector, {debug["vec"].shape}, and norm')
        ax1.legend()

        ax2.axhline(0, color='r')
        ax2.plot(norm - 1, style)
        maxdiff = np.nanmax(np.abs(norm - 1))
        ax2.set_title(f'deviation of norm from 1 (norm - 1), max diff: {maxdiff}')

        x = np.arange(N)
        y = np.ones_like(x)
        ax3.plot(x[debug['notclose'].flatten()], 2*y[debug['notclose'].flatten()], '.C3', label='not close')
        ax3.plot(x[debug['isnan'].flatten()], 1*y[debug['isnan'].flatten()], '.C1', label='is NaN')
        ax3.plot(x[debug['isclose'].flatten()], 0*y[debug['isclose'].flatten()], '.C2', label='is close')
        ax3.set_title(f'unit quaternion check result: {debug["res"]}')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['close', 'NaN', 'NOT close'])
        ax3.set_ylim(-0.2, 2.2)
        ax3.legend()

        for ax in (ax1, ax2, ax3):
            ax.grid()


def randomUnitVec(N=None, M=3, debug=False, plot=False):
    """
    Generate random unit vectors.

    M denotes the number of elements in the vector (default: 3).

    If N is None (default), a single random quaternion is returned as an array with shape (M,).
    If N is an integer, N random quaternions are returned as an (N, M) array.
    If N is a tuple, it denotes the shape of the output vectors, e.g. N=(5, 20) returns 100 random vectors as a
    (5, 20, 3) array.

    See also :meth:`qmt.randomQuat`.

    :param N: number of vectors to generate
    :param M: number of elements of each unit vector (default: 3)
    :return:
        - **vec**: vector output array, (..., M)
        - **debug**: dict with debug values (only if debug==True)
    """

    assert isinstance(M, int)
    if N is None:
        shape = (M,)
    elif isinstance(N, tuple):
        shape = N + (M,)
    else:
        shape = (N, M)

    # Use normal distribution (instead of uniform distribution) since normalizing the resulting vectors will
    # distribute them equally on the sphere surface.
    vec = normalized(np.random.standard_normal(shape))

    if debug or plot:
        debugData = dict(
            vec=vec,
            norm=vecnorm(vec),
            N=N,
            M=M,
        )
        if plot:
            randomUnitVec_debugPlot(debugData, plot)
        if debug:
            return vec, debugData
    return vec


def randomUnitVec_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('randomUnitVec'))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        vec = debug['vec'].reshape(-1, debug['vec'].shape[-1])
        norm = debug['norm'].flatten()
        style = '.-' if vec.shape[0] <= 100 else '-'

        ax1.plot(vec, style)
        ax2.plot(norm, style, color='k', lw=1, label='norm')
        ax1.set_title(f'output vector, {debug["vec"].shape}')
        ax2.set_title('norm')
        ax1.legend('xyz')
        ax1.grid()
        ax2.grid()
        fig.tight_layout()


def vecInterp(vec, ind, extend=True, debug=False, plot=False):
    """
    Interpolates an array at (non-integer) indices using linear interpolation.

    Sampling indices are in the range 0..N-1. For values outside of this range, depending on "extend", the first/last
    element or NaN is returned.

    See also :meth:`qmt.quatInterp`.

    :param vec: array of input data, (N, P)
    :param ind: vector containing the sampling indices, shape (M,)
    :param extend: if true, the input data is virtually extended by the first/last value
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **out**: interpolated values, (M, P) or (P,) if ind is scalar
        - **debug**: dict with debug values (only if debug==True)
    """
    vec = np.asarray(vec, float)
    ind = np.asarray(ind, float)
    isScalar = ind.ndim == 0
    ind = np.atleast_1d(ind)
    N = vec.shape[0]
    M = ind.shape[0]
    P = vec.shape[1]
    assert vec.shape == (N, P)
    assert ind.shape == (M,)

    ind0 = np.clip(np.floor(ind).astype(int), 0, N - 1)
    ind1 = np.clip(np.ceil(ind).astype(int), 0, N - 1)

    v0 = vec[ind0]
    v1 = vec[ind1]
    v_1_0 = v1 - v0
    t01 = ind - ind0

    out = v0 + t01[:, None] * v_1_0

    if not extend:
        out[ind < 0] = np.nan
        out[ind > N - 1] = np.nan

    if isScalar:
        out = out.reshape((P,))

    if debug or plot:
        debugData = dict(
            vec=vec,
            N=N,
            M=M,
            P=P,
            ind=ind,
            out=out,
        )
        if plot:
            vecInterp_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def vecInterp_debugPlot(debug, fig):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('vecInterp'))
        ax = fig.subplots(1, 1)
        style = '.-' if debug['N'] <= 100 else '-'

        ax.plot(debug['vec'], style)
        ax.set_prop_cycle(None)
        ax.plot(debug['ind'], debug['out'].reshape(-1, debug['P']), 'x-')
        ax.set_title(f'input vec [{debug["vec"].shape}, circles]; interpolated vec: [{debug["out"].shape}, crosses]')
        ax.grid()
        fig.tight_layout()


def nanInterp(signal, quatdetect=True, debug=False, plot=False):
    """
    Fills NaN samples by linear interpolation or quaternion slerp based on the previous and next non-NaN sample.

    NaN samples at the beginning and end are set to the first/last non-NaN sample. A sample is defined as a NaN sample
    if at least one element is NaN.

    :param signal: input signal, shape (N, M) or (N,)
    :param quatdetect: if True and M==4, use slerp instead of linear interpolation
    :param debug: enables returning debug data
    :param plot: enables the debug plot
    :return:
        - **out**: interpolated signal with the same shape
        - **debug**: dict with debug values (only if debug==True)
    """

    signal = np.asarray(signal, float)
    out = signal.copy()

    is1D = out.ndim == 1
    if is1D:
        out = out[:, None]

    nanind = np.any(np.isnan(out), axis=1)
    validind = ~nanind

    # cf https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    xp = validind.nonzero()[0]
    ind = np.interp(nanind.nonzero()[0], xp, np.arange(xp.shape[0]))

    if quatdetect and out.shape[1] == 4:
        from qmt.functions.quaternion import quatInterp
        out[nanind] = quatInterp(out[validind], ind)
        mode = 'quatInterp'
    else:
        out[nanind] = vecInterp(out[validind], ind)
        mode = 'vecInterp'

    if is1D:
        out.shape = signal.shape

    if debug or plot:
        debugData = dict(signal=signal, out=out, nanind=nanind, mode=mode)
        if plot:
            nanInterp_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def nanInterp_debugPlot(debug, fig):
    signal = debug['signal']
    out = debug['out']
    nanind = debug['nanind']
    style = '.-' if debug['nanind'].size <= 100 else '-'

    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('nanInterp'))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        ax1.plot(signal, style, label='in')
        ax1.legend(loc='upper left')
        ax1b = ax1.twinx()
        ax1b.plot(nanind, 'r', label='nan')
        ax1b.legend(loc='upper right')
        ax1.set_title(f'input signal, shape {signal.shape}, and nan indices (red)')
        ax2.plot(out, style, label='out')
        ax2.set_title(f'out (interpolation mode: {debug["mode"]}), shape {out.shape}')
        for ax in ax1, ax2:
            ax.grid()
        fig.tight_layout()


def rms(x, axis=0, debug=False, plot=False):
    """
    Calculates root-mean-square (RMS) values.

    For a 1D array, the RMS over all elements is calculated.
    For arrays with more dimensions, the RMS is calculated along a specified axis (default: 0).
    NaNs in input array will be ignored.

    :param x: input array
    :param axis: axis along which the means are computed
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **out**: root mean square
        - **debug**: dict with debug values (only if debug==True)
    """

    x = np.asarray(x, float)
    out = np.sqrt(np.nanmean(x ** 2, axis=axis))

    if debug or plot:
        debugData = dict(x=x, out=out, axis=axis)
        if plot:
            rms_debugPlot(debugData, plot)
        if debug:
            return out, debugData

    return out


def rms_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('rms'))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        style = '.-' if max(debug['x'].size, debug['out'].size) <= 400 else '-'

        ax1.plot(debug['x'], style)
        ax1.set_title(f'input x, {debug["x"].shape}')
        ax1.grid()

        ax2.plot(debug['out'], style)
        ax2.set_title(f'out, {debug["out"].shape}')
        ax2.grid()

        fig.tight_layout()
