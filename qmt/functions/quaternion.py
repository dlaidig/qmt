# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
import qmt
import numpy as np

from qmt.utils.plot import AutoFigure
from qmt.functions.utils import normalized, angleBetween2Vecs, wrapToPi, allUnitNorm, vecnorm, nanUnwrap, randomUnitVec


def qmult(q1, q2, strict=True, debug=False, plot=False):
    """
    Quaternion multiplication.

    If two Nx4 arrays are given, they are multiplied row-wise. Alternatively, one of the inputs can be a single
    quaternion which is then multiplied to all rows of the other input array:

    >>> qmt.qmult([[1, 0, 0, 0], [0, 0, 0, 1]], [0.5, 0.5, 0.5, 0.5])
    array([[ 0.5,  0.5,  0.5,  0.5],
           [-0.5, -0.5,  0.5,  0.5]])

    If one quaternion is NaN, the corresponding output quaternion will be NaN.
    If strict is True and inputs are not unit quaternions, a ValueError will be raised.
    For advanced use: input arrays with more dimensions (and different shapes) are also supported, as long as they can
    be broadcast to a common shape and the last dimension has length 4.

    Equivalent Matlab function: :mat:func:`+qmt.qmult`.

    For examples, see :ref:`tutorial_py_quaternion_functions`.

    :param q1: quaternion input array, shape: (..., 4)
    :param q2: quaternion input array, shape: (..., 4)
    :param strict: if set to true, an error is raised if q1 or q2 do not contain unit quaternions
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: quaternion output array, shape: (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    # convert the inputs to floating point arrays if necessary
    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)

    # check the norms
    if strict:
        if not allUnitNorm(q1):
            raise ValueError('q1 does not contain unit quaternions')
        if not allUnitNorm(q2):
            raise ValueError('q2 does not contain unit quaternions')

    # broadcast the arrays if necessary
    q1orig = q1
    q2orig = q2
    if q1.shape != q2.shape:
        q1, q2 = np.broadcast_arrays(q1, q2)
    assert q1.shape[-1] == 4

    # actual quaternion multiplication
    q3 = np.zeros(q1.shape, float)
    q3[..., 0] = q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3]
    q3[..., 1] = q1[..., 0] * q2[..., 1] + q1[..., 1] * q2[..., 0] + q1[..., 2] * q2[..., 3] - q1[..., 3] * q2[..., 2]
    q3[..., 2] = q1[..., 0] * q2[..., 2] - q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] + q1[..., 3] * q2[..., 1]
    q3[..., 3] = q1[..., 0] * q2[..., 3] + q1[..., 1] * q2[..., 2] - q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0]

    if debug or plot:
        debugData = dict(
            q1=q1orig,
            q2=q2orig,
            q3=q3,
            q1_euler=eulerAngles(q1orig),
            q2_euler=eulerAngles(q2orig),
            q3_euler=eulerAngles(q3),
            q1_norm=vecnorm(q1orig),
            q2_norm=vecnorm(q2orig),
            q3_norm=vecnorm(q3),
        )
        if plot:
            qmult_debugPlot(debugData, plot)
        if debug:
            return q3, debugData
    return q3


def _plotQuatEuler(ax0, ax1, debug, name, desc=''):
    q = debug[name]
    euler = debug[f'{name}_euler']
    norm = debug[f'{name}_norm']

    q_shape = q.shape
    euler_shape = euler.shape
    assert q_shape[:-1] == euler_shape[:-1]
    assert q_shape[:-1] == norm.shape

    q = q.reshape((-1, 4))
    style = '.-' if q.shape[0] <= 100 else '-'
    ax0.plot(q, style)
    ax0.plot(norm.flatten(), style, color='k', alpha=0.3, lw=1)
    ax0.legend(['w', 'x', 'y', 'z', 'norm'])
    ax0.set_title(f'{desc} quaternion {name}: {q_shape}'.strip())
    ax0.grid()

    euler = euler.reshape((-1, 3))
    if ax1 is not None:
        ax1.plot(np.rad2deg(euler), style)
        ax1.legend(["z", "y'", "x''"])
        ax1.set_title(f'{name} Euler angles [°]')
        ax1.grid()

    return style


def qmult_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        axes = fig.subplots(3, 2, sharex=True)
        fig.suptitle(AutoFigure.title('qmult'))

        for i in range(3):
            val = f'q{i+1}'
            desc = 'input' if i + 1 != 3 else 'output'
            _plotQuatEuler(axes[i, 0], axes[i, 1], debug, val, desc)

        fig.tight_layout()


def qinv(q, debug=False, plot=False):
    """
    Calculates the inverse/conjugate of unit quaternions.

    :param q: quaternion input array, shape (..., 4)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: quaternion output array, (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    # convert the input to floating point arrays if necessary
    q = np.asarray(q, float)

    # check the norm
    if not allUnitNorm(q):
        raise ValueError('q does not contain unit quaternions')

    assert q.shape[-1] == 4

    # calculate inverse
    out = q.copy()
    out[..., 1:] *= -1

    if debug or plot:
        debugData = dict(
            q=q,
            out=out,
            q_euler=eulerAngles(q),
            out_euler=eulerAngles(out),
            q_norm=vecnorm(q),
            out_norm=vecnorm(out),
        )
        if plot:
            qinv_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def qinv_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        axes = fig.subplots(2, 2, sharex=True)
        fig.suptitle(AutoFigure.title('qinv'))

        for i, val in enumerate(['q', 'out']):
            desc = 'input' if i == 0 else 'output'
            _plotQuatEuler(axes[i, 0], axes[i, 1], debug, val, desc)

        fig.tight_layout()


def qrel(q1, q2, debug=False, plot=False):
    """
    Calculates relative quaternions, i.e., ``out = qmult(inv(q1), q2)``.

    With respect to broadcasting and NaN handling, the function works like :func:`qmt.qmult`. Inputs are always
    checked for unit norm.

    :param q1: quaternion input array, shape: (..., 4)
    :param q2: quaternion input array, shape: (..., 4)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: quaternion output array, shape: (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    # convert the inputs to floating point arrays if necessary
    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)

    # check the norms
    if not allUnitNorm(q1):
        raise ValueError('q1 does not contain unit quaternions')
    if not allUnitNorm(q2):
        raise ValueError('q2 does not contain unit quaternions')

    # broadcast the arrays if necessary
    q1orig = q1
    q2orig = q2
    if q1.shape != q2.shape:
        q1, q2 = np.broadcast_arrays(q1, q2)
    assert q1.shape[-1] == 4

    # actual relative quaternion calculation
    q3 = np.zeros(q1.shape, float)
    q3[..., 0] = q1[..., 0] * q2[..., 0] + q1[..., 1] * q2[..., 1] + q1[..., 2] * q2[..., 2] + q1[..., 3] * q2[..., 3]
    q3[..., 1] = q1[..., 0] * q2[..., 1] - q1[..., 1] * q2[..., 0] - q1[..., 2] * q2[..., 3] + q1[..., 3] * q2[..., 2]
    q3[..., 2] = q1[..., 0] * q2[..., 2] + q1[..., 1] * q2[..., 3] - q1[..., 2] * q2[..., 0] - q1[..., 3] * q2[..., 1]
    q3[..., 3] = q1[..., 0] * q2[..., 3] - q1[..., 1] * q2[..., 2] + q1[..., 2] * q2[..., 1] - q1[..., 3] * q2[..., 0]

    if debug or plot:
        debugData = dict(
            q1=q1orig,
            q2=q2orig,
            q3=q3,
            q1_euler=eulerAngles(q1orig),
            q2_euler=eulerAngles(q2orig),
            q3_euler=eulerAngles(q3),
            q1_norm=vecnorm(q1orig),
            q2_norm=vecnorm(q2orig),
            q3_norm=vecnorm(q3),
        )
        if plot:
            qrel_debugPlot(debugData, plot)
        if debug:
            return q3, debugData
    return q3


def qrel_debugPlot(debug, fig):
    with AutoFigure(fig) as fig:
        axes = fig.subplots(3, 2, sharex=True)
        fig.suptitle(AutoFigure.title('qrel'))

        for i in range(3):
            val = f'q{i + 1}'
            desc = 'input' if i + 1 != 3 else 'output'
            _plotQuatEuler(axes[i, 0], axes[i, 1], debug, val, desc)

        fig.tight_layout()


def rotate(q, v, debug=False, plot=False):
    """
    Rotates vectors with the given quaternions.

    The rotated vector out is calculated as

    ``[0, out] = qmult(q, qmult([0, v], qinv(q)))``

    If a Nx4 quaternion array and a Nx3 vector array are given, rotation is calculated row-wise. Alternatively, one of
    the inputs can be a single quaternion or single vector, which is then applied to all rows of the other array. More
    complicated input shapes, following the numpy broadcasting rules, are also supported.

    The input quaternions are checked to have unit norm. If the inputs are NaN, the corresponding output will be NaN.

    :param q: quaternion input array, shape (..., 4)
    :param v: vector input array, shape (..., 3)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: vector output array, (..., 3)
        - **debug**: dict with debug values (only if debug==True)
    """

    # convert the inputs to floating point arrays if necessary
    q = np.asarray(q, float)
    v = np.asarray(v, float)

    # check shapes and quaternion norm
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    if not allUnitNorm(q):
        raise ValueError('q does not contain unit quaternions')

    # determine output shape via numpy broadcasting rules
    shape = np.broadcast_shapes(q.shape[:-1] + (3,), v.shape)

    # perform actual calculation
    out = np.zeros(shape)
    out[..., 0] = (1 - 2 * q[..., 2] ** 2 - 2 * q[..., 3] ** 2) * v[..., 0] \
        + 2 * v[..., 1] * (q[..., 2] * q[..., 1] - q[..., 0] * q[..., 3]) \
        + 2 * v[..., 2] * (q[..., 0] * q[..., 2] + q[..., 3] * q[..., 1])

    out[..., 1] = 2 * v[..., 0] * (q[..., 0] * q[..., 3] + q[..., 2] * q[..., 1]) \
        + v[..., 1] * (1 - 2 * q[..., 1] ** 2 - 2 * q[..., 3] ** 2) \
        + 2 * v[..., 2] * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])

    out[..., 2] = 2 * v[..., 0] * (q[..., 3] * q[..., 1] - q[..., 0] * q[..., 2]) \
        + 2 * v[..., 1] * (q[..., 0] * q[..., 1] + q[..., 3] * q[..., 2]) \
        + v[..., 2] * (1 - 2 * q[..., 1] ** 2 - 2 * q[..., 2] ** 2)

    if debug or plot:
        debugData = dict(
            q=q,
            v=v,
            q_euler=eulerAngles(q),
            q_norm=vecnorm(q),
            out=out
        )
        if plot:
            rotate_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def rotate_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)
        fig.suptitle(AutoFigure.title('rotate'))

        style = _plotQuatEuler(ax1, ax2, debug, 'q', 'input')
        v = debug['v'].reshape(-1, 3)
        ax3.plot(v, style)
        ax3.set_title(f'input vector v: {debug["v"].shape}')
        ax3.legend('xyz')
        ax3.grid()

        out = debug['out'].reshape(-1, 3)
        ax4.plot(out, style)
        ax4.set_title(f'rotated vector out: {debug["out"].shape}')
        ax4.legend('xyz')
        ax4.sharey(ax3)
        ax4.grid()

        fig.tight_layout()


def quatTransform(qTrafo, q, debug=False, plot=False):
    """
    Transforms a rotation quaternion with a given transformation quaternion, i.e., ``qTrafo*q*inv(qTrafo)``,

    >>> qmt.quatTransform(np.array([0, 0, 1, 0]), np.array([0, 1, 0, 0]))
    array([ 0., -1.,  0.,  0.])

    :param qTrafo: transformation quaternion input array, (..., 4)
    :param q: rotation quaternion input array, (..., 4)

    :return:
        - **out**: quaternion output array, (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    out = qmult(qmult(qTrafo, q), qinv(qTrafo))

    if debug or plot:
        qTrafo = np.asarray(qTrafo, float)
        q = np.asarray(q, float)
        debugData = dict(
            qTrafo=qTrafo,
            qTrafo_euler=eulerAngles(qTrafo),
            qTrafo_norm=vecnorm(qTrafo),
            q=q,
            q_euler=eulerAngles(q),
            q_norm=vecnorm(q),
            out=out,
            out_euler=eulerAngles(out),
            out_norm=vecnorm(out),
        )
        if plot:
            quatTransform_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def quatTransform_debugPlot(debug, fig):
    with AutoFigure(fig) as fig:
        axes = fig.subplots(3, 2, sharex=True)
        fig.suptitle(AutoFigure.title('quatTransform'))

        for i, val in enumerate(['qTrafo', 'q', 'out']):
            desc = 'input' if i != 2 else 'output'
            _plotQuatEuler(axes[i, 0], axes[i, 1], debug, val, desc)
        fig.tight_layout()


def quatProject(q, axis, debug=False, plot=False):
    """
    Calculate quaternion projection along a given axis.

    The quaternion projection is defined as the rotation around the specified axis that results in the smallest possible
    residual rotation. The axis of this residual rotation is always orthogonal to the given projection axis, which is
    similar to vector projection (cf. https://en.wikipedia.org/wiki/Vector_projection).

    The output is a dictionary that contains the angles and quaternions for both the projection rotation and the
    residual rotation. Both rotations can be combined so that q = qmult(projQuat, resQuat).

    >>> qmt.quatProject(np.array([[0.5, 0.5, 0.5, 0.5], [0, 0, 0, 1]]), np.array([1, 1, 0]))
        {'projAngle': array([1.91063324, 0.        ]),
         'resAngle': array([ 1.04719755, -3.14159265]),
         'projQuat': array([[0.57735027, 0.57735027, 0.57735027, 0.        ],
                [1.        , 0.        , 0.        , 0.        ]]),
         'resQuat': array([[ 0.8660254 , -0.28867513,  0.28867513,  0.28867513],
                [ 0.        ,  0.        ,  0.        ,  1.        ]])}

    :param q: quaternion input array, (..., 4)
    :param axis: projection axis vector, (..., 3)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **outputs**: dict with projAngle (...,), resAngle (...,), projQuat (..., 4), resQuat (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    q = np.asarray(q, float)
    axis = np.asarray(axis, float)

    assert q.shape[-1] == 4
    assert axis.shape[-1] == 3
    if not allUnitNorm(q):
        raise ValueError('q does not contain unit quaternions')

    # determine output shape via numpy broadcasting rules
    shape = np.broadcast_shapes(q.shape, axis.shape[:-1] + (4,))

    axis = normalized(axis)

    projAngle = wrapToPi(2 * np.arctan2(np.sum(axis * q[..., 1:], axis=-1), q[..., 0]))
    projQuat = quatFromAngleAxis(projAngle, axis)
    resQuat = qrel(projQuat, q)
    resAngle = quatAngle(resQuat)

    assert projQuat.shape == resQuat.shape == shape

    outputs = dict(
        projAngle=projAngle,
        resAngle=resAngle,
        projQuat=projQuat,
        resQuat=resQuat,
    )

    if debug or plot:
        debugData = dict(
            q=q,
            axis=axis,
            projAngle=projAngle,
            resAngle=resAngle,
            projQuat=projQuat,
            resQuat=resQuat,
            projQuat_norm=vecnorm(projQuat),
            resQuat_norm=vecnorm(resQuat),
            projQuat_euler=eulerAngles(projQuat),
            resQuat_euler=eulerAngles(resQuat),
        )
        if plot:
            quatProject_debugPlot(debugData, plot)
        if debug:
            outputs['debug'] = debugData
    return outputs


def quatProject_debugPlot(debug, fig):
    with AutoFigure(fig) as fig:
        axes = fig.subplots(3, 2, sharex=True)
        fig.suptitle(AutoFigure.title('quatProject'))

        for i, val in enumerate(['projQuat', 'resQuat']):
            style = _plotQuatEuler(axes[i, 0], axes[i, 1], debug, val)

        axes[2, 0].plot(np.rad2deg(debug['projAngle']).flatten(), style)
        axes[2, 0].set_title(f'projection angle in °, {debug["projAngle"].shape}')
        axes[2, 0].grid()

        axes[2, 1].plot(np.rad2deg(debug['resAngle']).flatten(), style)
        axes[2, 1].set_title(f'residual angle in °, {debug["resAngle"].shape}')
        axes[2, 1].grid()
        fig.tight_layout()


def eulerAngles(q, axes='zyx', intrinsic=True, debug=False, plot=False):
    """
    Calculate Euler angles from quaternions.

    All possible rotation sequences are supported and can be specified with the axes parameter. Examples for valid
    values (that all have the same meaning) are 'zxy', 'ZXY', 312, '312'. By default, intrinsic angles are calculated.
    Set intrinsic to False to calculate extrinsic angles.

    In the case of gimbal lock (second angle is 90° or -90° for Tait-Byran angles, or 0° or -180° for proper Euler
    angles), the last angle is set to zero. Note that this is an arbitrary choice, but the returned angles still
    represent the correct rotation.

    :param q: input quaternion array, (..., 4)
    :param axes: rotation sequence
    :param intrinsic: calculate intrinsic angles if True, extrinsic if False
    :return:
        - **out**: output array with Euler angles in rad, (..., 3)
        - **debug**: dict with debug values (only if debug==True)
    """

    # check the axis parameter and derive constants a, b, c, and d
    axisIdentifiers = {
        1: 1, '1': 1, 'x': 1, 'X': 1, 'i': 1,
        2: 2, '2': 2, 'y': 2, 'Y': 2, 'j': 2,
        3: 3, '3': 3, 'z': 3, 'Z': 3, 'k': 3,
    }
    if len(axes) != 3:
        raise ValueError('invalid Euler rotation sequence')
    origAxes = axes
    if intrinsic:
        axes = axes[::-1]
    try:
        a = axisIdentifiers[axes[0]]
        b = axisIdentifiers[axes[1]]
        c = axisIdentifiers[axes[2]]
        d = 'invalid'
        if a == c:
            d = (set([1, 2, 3]) - set([a, b])).pop()
    except KeyError:
        raise ValueError('invalid Euler rotation sequence')
    if b == a or b == c:
        raise ValueError('invalid Euler rotation sequence')

    # sign factor depending on the axes order
    if b == (a % 3) + 1:  # cyclic order
        s = 1
    else:  # anti-cyclic order
        s = -1

    q = np.asarray(q, float)
    assert q.shape[-1] == 4
    if not allUnitNorm(q):
        raise ValueError('q does not contain unit quaternions')

    if a == c:  # proper Euler angles
        angle1 = np.arctan2(q[..., a] * q[..., b] - s * q[..., d] * q[..., 0],
                            q[..., b] * q[..., 0] + s * q[..., a] * q[..., d])
        angle2 = np.arccos(np.clip(q[..., 0] ** 2 + q[..., a] ** 2 - q[..., b] ** 2 - q[..., d] ** 2, -1, 1))
        angle3 = np.arctan2(q[..., a] * q[..., b] + s * q[..., d] * q[..., 0],
                            q[..., b] * q[..., 0] - s * q[..., a] * q[..., d])

        gimbalLock = np.logical_or(np.abs(angle2) < 1e-7, np.abs(angle2-np.pi) < 1e-7)

    else:  # Tait-Bryan
        angle1 = np.arctan2(2 * (q[..., a] * q[..., 0] + s * q[..., b] * q[..., c]),
                            q[..., 0] ** 2 - q[..., a] ** 2 - q[..., b] ** 2 + q[..., c] ** 2)
        angle2 = np.arcsin(np.clip(2 * (q[..., b] * q[..., 0] - s * q[..., a] * q[..., c]), -1, 1))
        angle3 = np.arctan2(2 * (s * q[..., a] * q[..., b] + q[..., c] * q[..., 0]),
                            q[..., 0] ** 2 + q[..., a] ** 2 - q[..., b] ** 2 - q[..., c] ** 2)

        gimbalLock = np.logical_or(np.abs(angle2-np.pi/2) < 1e-7, np.abs(angle2+np.pi/2) < 1e-7)

    if intrinsic:
        out = np.stack((angle3, angle2, angle1), axis=-1)
    else:
        out = np.stack((angle1, angle2, angle3), axis=-1)

    if np.any(gimbalLock):
        # get quaternion corresponding to second angle (which is well-defined)
        axis2 = [0, 0, 0]
        axis2[b-1] = 1
        quat2 = quatFromAngleAxis(angle2, axis2)

        # get quaternion corresponding to first angle (assuming the third angle is zero)
        quat1 = qmult(q, qinv(quat2)) if intrinsic else qmult(qinv(quat2), q)

        # get angle along first rotation axis
        axis1 = [0, 0, 0]
        axis1[c - 1 if intrinsic else a - 1] = 1
        angle1 = quatProject(quat1, axis1)['projAngle']

        # overwrite the gimbal lock entries
        out[gimbalLock, 0] = angle1[gimbalLock]
        out[gimbalLock, 2] = 0

    if debug or plot:
        debugData = dict(
            q=q,
            q_norm=vecnorm(q),
            out=out,
            axes=origAxes,
            intrinsic=intrinsic,
            gimbal_lock=gimbalLock,
        )
        if plot:
            eulerAngles_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def _eulerLegend(axes, intrinsic):
    assert len(axes) == 3
    if intrinsic:
        return axes[0], axes[1]+"'", axes[2]+"''"
    else:
        return axes[0], axes[1], axes[2]


def _eulerString(axes, intrinsic):
    return '-'.join(_eulerLegend(axes, intrinsic))


def eulerAngles_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('eulerAngles'))
        (ax1, ax2, ax3) = fig.subplots(3, 1, sharex=False)

        dbg = dict(q=debug['q'], q_norm=debug['q_norm'], q_euler=debug['out'])
        style = _plotQuatEuler(ax1, ax2, dbg, 'q', 'input')
        ax2.set_title(f'output Euler {_eulerString(debug["axes"], debug["intrinsic"])} angles [°]: '
                      f'{debug["out"].shape}')
        ax2.legend(_eulerLegend(debug["axes"], debug["intrinsic"]))

        out = debug['out'].reshape(-1, 3)
        ax3.plot(np.rad2deg(nanUnwrap(out)), style)
        ax3.set_title(f'unwrapped output {_eulerString(debug["axes"], debug["intrinsic"])} Euler angles [°]')
        ax3.legend(_eulerLegend(debug["axes"], debug["intrinsic"]))
        ax3.grid()
        fig.tight_layout()


def quatFromEulerAngles(angles, axes='zyx', intrinsic=True, debug=False, plot=False):
    """
    Calculate quaternions from Euler angles.

    All possible rotation sequences are supported and can be specified with the axes parameter. Examples for valid
    values (that all have the same meaning) are 'zxy', 'ZXY', 312, '312'. By default, intrinsic angles are used.
    Set intrinsic to False to use extrinsic angles.

    :param angles: input Euler angle array, (..., 3)
    :param axes: rotation sequence
    :param intrinsic: calculate intrinsic angles if True, extrinsic if False
    :return:
        - **out**: output array with Euler angles in rad, (..., 3)
        - **debug**: dict with debug values (only if debug==True)
    """

    # check the axis parameter and derive constants a, b, c, and d
    axisIdentifiers = {
        1: 1, '1': 1, 'x': 1, 'X': 1, 'i': 1,
        2: 2, '2': 2, 'y': 2, 'Y': 2, 'j': 2,
        3: 3, '3': 3, 'z': 3, 'Z': 3, 'k': 3,
    }
    if len(axes) != 3:
        raise ValueError('invalid Euler rotation sequence')

    out = np.array([1.0, 0.0, 0.0, 0.0], float)
    for i, identifier in enumerate(axes):
        axis = [0, 0, 0]
        axis[axisIdentifiers[identifier]-1] = 1.0
        qRot = quatFromAngleAxis(angles[..., i], axis)
        if intrinsic:
            out = qmult(out, qRot)
        else:
            out = qmult(qRot, out)

    if debug or plot:
        debugData = dict(
            angles=angles,
            out=out,
            out_norm=vecnorm(out),
            axes=axes,
            intrinsic=intrinsic,
        )
        if plot:
            quatFromEulerAngles_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def quatFromEulerAngles_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatFromEulerAngles'))
        (ax1, ax2) = fig.subplots(2, 1, sharex=False)

        dbg = dict(q=debug['out'], q_norm=debug['out_norm'], q_euler=debug['angles'])
        _plotQuatEuler(ax2, ax1, dbg, 'q', 'input')
        ax1.set_title(f'input Euler {_eulerString(debug["axes"], debug["intrinsic"])} angles [°]: '
                      f'{debug["angles"].shape}')
        ax2.set_title(f'output quaternion: {debug["out"].shape}')
        ax1.legend(_eulerLegend(debug["axes"], debug["intrinsic"]))
        fig.tight_layout()


def quatAngle(q, debug=False, plot=False):
    """
    Returns the quaternion rotation angle in the range -pi..pi.

    Because the output range of the angle is -pi..pi, roundtrip conversion in combination with :meth:`qmt.quatAxis`
    and :meth:`qmt.quatFromAngleAxis` can return -q instead of q, which expresses the same rotation. To uniquely
    compare the outputs, use :meth:`qmt.posScalar`.

    >>> np.rad2deg(qmt.quatAngle(np.array([[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]])))
        array([  0., 120.])

    :param q: input quaternion array, (..., 4)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: rotation angles in rad, shape (...,)
        - **debug**: dict with debug values (only if debug==True)
    """

    q = np.asarray(q, float)
    assert q.shape[-1] == 4
    if not allUnitNorm(q):
        raise ValueError('q does not contain unit quaternions')

    angle = wrapToPi(2 * np.arctan2(vecnorm(q[..., 1:]), q[..., 0]))
    # note: the implementation with arctan2 is more numerically stable than the following variant:
    # angle = wrapToPi(2 * np.arccos(np.clip(q[..., 0], -1, 1)))

    if debug or plot:
        debugData = dict(
            q=q,
            q_euler=eulerAngles(q),
            q_norm=vecnorm(q),
            angle=angle,
        )
        if plot:
            quatAngle_debugPlot(debugData, plot)
        if debug:
            return angle, debugData
    return angle


def quatAngle_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatAngle'))
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)

        style = _plotQuatEuler(ax1, ax2, debug, 'q', 'input')

        ax3.plot(np.rad2deg(debug['angle']), style)
        ax3.set_title(f'output angle [°], {debug["angle"].shape}')
        ax3.grid()

        fig.tight_layout()


def quatAxis(q, debug=False, plot=False):
    """
    Returns the quaternion rotation axis.

    For identity quaternions, the arbitrary axis [1 0 0] is returned.

    >>> qmt.quatAxis(np.array([[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]]))
        array([[1.        , 0.        , 0.        ],
               [0.57735027, 0.57735027, 0.57735027]])

    :param q: input quaternion array, (..., 4)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: rotation axis, (..., 3)
        - **debug**: dict with debug values (only if debug==True)
    """

    q = np.asarray(q, float)
    assert q.shape[-1] == 4
    if not allUnitNorm(q):
        raise ValueError('q does not contain unit quaternions')

    axis = q[..., 1:].copy()
    norm = vecnorm(axis)
    axis[norm < np.finfo(np.float64).eps] = np.array([1, 0, 0], float)
    axis = qmt.normalized(axis)

    if debug or plot:
        debugData = dict(
            q=q,
            q_euler=eulerAngles(q),
            q_norm=vecnorm(q),
            axis=axis,
        )
        if plot:
            quatAxis_debugPlot(debugData, plot)
        if debug:
            return axis, debugData
    return axis


def quatAxis_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatAxis'))
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)

        style = _plotQuatEuler(ax1, ax2, debug, 'q', 'input')

        axis = debug['axis'].reshape(-1, 3)
        ax3.plot(axis, style)
        ax3.plot(vecnorm(axis), style, color='k', alpha=0.3, lw=1)
        ax3.set_title(f'output axis, {debug["axis"].shape}')
        ax3.legend(['x', 'y', 'z', 'norm'])
        ax3.grid()

        fig.tight_layout()


def quatFromAngleAxis(angle, axis, debug=False, plot=False):
    """
    Create quaternion that represents a rotation of a given angle around a given axis.

    If the same number of angles and axes are given, the quaternion is calculated row-wise based on the corresponding
    entries. It is possible to only supply a single angle which is then combined with all axes and vice versa.
    Furthermore, higher-dimensional inputs, following the numpy broadcasting rules, are also supported.

    If angle is 0, the output will be an identity quaternion.
    If axis is zero vector, a ValueError will be raised unless the corresponding angle is 0.

    >>> qmt.quatFromAngleAxis([0, 1, 2], [1, 0, 0])
    array([[1.        , 0.        , 0.        , 0.        ],
           [0.87758256, 0.47942554, 0.        , 0.        ],
           [0.54030231, 0.84147098, 0.        , 0.        ]])

    :param angle: angle in rad, shape (...,)
    :param axis: rotation axis vector, (..., 3)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: quaternion output array (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    # convert the inputs to floating point arrays if necessary
    angle = np.asarray(angle, float)
    axis = np.asarray(axis, float)

    assert axis.shape[-1] == 3, f'invalid axis shape: {axis.shape}'
    shape = np.broadcast_shapes(axis.shape[:-1], angle.shape)
    angle_brodcasted = np.broadcast_to(angle, shape)
    axis_brodcasted = np.broadcast_to(axis, shape + (3,))

    norm = vecnorm(axis)
    zeroaxis = norm < np.finfo(float).eps

    if np.any(zeroaxis):  # only perform check if there are axis values with zero norm
        if not np.allclose(angle_brodcasted[vecnorm(axis_brodcasted) < np.finfo(float).eps], 0):
            raise ValueError('some axis values are zero and the corresponding angle is non-zero')

        # to avoid indexing with zeroaxis and ~zeroaxis later:
        # set norm to 1 to avoid division by zero (angle is checked to be zero, i.e., sin(angle/2) == 0)
        if norm.ndim == 0:  # item assignment not possible in scalar case
            norm = np.float64(1)
        else:
            norm[zeroaxis] = 1

    q = np.zeros(shape + (4,), float)
    q[..., 0] = np.cos(angle / 2)
    q[..., 1] = np.sin(angle / 2) * axis[..., 0] / norm
    q[..., 2] = np.sin(angle / 2) * axis[..., 1] / norm
    q[..., 3] = np.sin(angle / 2) * axis[..., 2] / norm

    if debug or plot:
        debugData = dict(
            angle=angle,
            axis=axis,
            axis_norm=vecnorm(axis),
            q=q,
            q_euler=eulerAngles(q),
            q_norm=vecnorm(q),
        )
        if plot:
            quatFromAngleAxis_debugPlot(debugData, plot)
        if debug:
            return q, debugData
    return q


def quatFromAngleAxis_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)
        fig.suptitle(AutoFigure.title(debug.get('function', 'quatFromAngleAxis')))

        style = _plotQuatEuler(ax3, ax4, debug, 'q', 'outut')

        axis = debug['axis'].reshape(-1, 3)
        ax1.plot(axis, style)
        ax1.plot(vecnorm(axis), style, color='k', alpha=0.3, lw=1)
        ax1.set_title(f'axis: {debug["axis"].shape}')
        ax1.legend(['x', 'y', 'z', 'norm'])
        ax1.grid()

        angle = debug['angle'].reshape(-1)
        ax2.plot(np.rad2deg(angle), style)
        ax2.set_title(f'angle [°]: {debug["angle"].shape}')
        ax2.grid()

        fig.tight_layout()


def quatToRotMat(q, debug=False, plot=False):
    """
    Converts quaternions to rotation matrices.

    >>> qmt.quatToRotMat([[1, 0, 0, 0], [0, 1, 0, 0]])
    array([[[ 1.  0.  0.],
            [ 0.  1.  0.],
            [ 0.  0.  1.]]
            [[ 1.  0.  0.],
            [ 0. -1.  0.],
            [ 0.  0. -1.]]])

    :param q: quaternion input array, (..., 4)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **R**: rotation matrix output array, (..., 3, 3)
        - **debug**: dict with debug values (only if debug==True)
    """
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix

    # convert the inputs to floating point arrays if necessary
    q = np.asarray(q, float)

    assert q.shape[-1] == 4
    if not allUnitNorm(q):
        raise ValueError('q does not contain unit quaternions')

    # actual calculation of quaternion to rotation matrix
    R = np.zeros(q.shape[:-1] + (3, 3), float)
    R[..., 0, 0] = 1 - 2 * q[..., 2] ** 2 - 2 * q[..., 3] ** 2
    R[..., 0, 1] = 2 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0])
    R[..., 0, 2] = 2 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0])
    R[..., 1, 0] = 2 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0])
    R[..., 1, 1] = 1 - 2 * q[..., 1] ** 2 - 2 * q[..., 3] ** 2
    R[..., 1, 2] = 2 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])
    R[..., 2, 0] = 2 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0])
    R[..., 2, 1] = 2 * (q[..., 1] * q[..., 0] + q[..., 2] * q[..., 3])
    R[..., 2, 2] = 1 - 2 * q[..., 1] ** 2 - 2 * q[..., 2] ** 2

    if debug or plot:
        debugData = dict(
            q=q,
            q_euler=eulerAngles(q),
            q_norm=vecnorm(q),
            R=R,
        )
        if plot:
            quatToRotMat_debugPlot(debugData, plot)
        if debug:
            return R, debugData
    return R


def _plotRotmat(ax, R, desc):
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    shape = R.shape
    R = R.reshape((-1, 3, 3))
    N = R.shape[0]

    ax.plot([-1.1, 1.1], [0, 0], [0, 0], color='0.5', lw=0.5)
    ax.plot([0, 0], [-1.1, 1.1], [0, 0], color='0.5', lw=0.5)
    ax.plot([0, 0], [0, 0], [-1.1, 1.1], color='0.5', lw=0.5)
    for m in range(N):
        for n in range(3):
            style = '-|>' if n == 0 else '->'
            a = Arrow3D([0, R[m, n, 0]], [0, R[m, n, 1]], [0, R[m, n, 2]],
                        mutation_scale=20, arrowstyle=style, color=f'C{m % 10}', alpha=0.8)
            ax.add_artist(a)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.set_title(f'{desc} R, {shape} (filled: x)')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.text(1.1, 0, 0, 'x', ha='center', va='center')
    ax.text(0, 1.1, 0, 'z', ha='center', va='center')
    ax.text(0, 0, 1.1, 'z', ha='center', va='center')
    ax.grid()


def quatToRotMat_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatToRotMat'))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(122, projection='3d')

        _plotQuatEuler(ax1, ax2, debug, 'q', 'input')
        _plotRotmat(ax3, debug['R'], 'output')
        fig.tight_layout()


def quatFromRotMat(R, method='auto', debug=False, plot=False):
    """
    Convert rotation matrices to quaternions.

    >>> qmt.quatFromRotMat([[1,0,0],[0,1,0],[0,0,1]])
    array([1.  0.  0. 0.])

    The input must be a rotation matrix with shape (3, 3) or multiple rotation matrices with shape (N, 3, 3).
    Higher-dimensional inputs are also supported as long as the last two axes have length 3.,

    :param R: rotation matrix input array, (..., 3, 3)
    :param method: method to use, must be 'auto' (default), 'copysign', 0, 1, 2 or 3
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: quaternion output array, (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    # convert the inputs to floating point arrays if necessary
    R = np.asarray(R, float)

    # check shape of input
    assert R.ndim >= 2 and R.shape[-2:] == (3, 3)

    # calculation of quaternion
    w_sq = (1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) / 4
    x_sq = (1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]) / 4
    y_sq = (1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2]) / 4
    z_sq = (1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2]) / 4

    q = np.zeros(R.shape[:-2] + (4,), float)

    if method == 'auto':  # use the largest value to avoid numerical problems
        methods = np.argmax(np.array([w_sq, x_sq, y_sq, z_sq]), axis=0)
    elif method == 'copysign':
        q[..., 0] = np.sqrt(w_sq)
        q[..., 1] = np.copysign(np.sqrt(x_sq), R[..., 2, 1] - R[..., 1, 2])
        q[..., 2] = np.copysign(np.sqrt(y_sq), R[..., 0, 2] - R[..., 2, 0])
        q[..., 3] = np.copysign(np.sqrt(z_sq), R[..., 1, 0] - R[..., 0, 1])
    elif method not in (0, 1, 2, 3):
        raise RuntimeError('invalid method, must be "copysign", "auto", 0, 1, 2 or 3')

    if method == 0 or method == 'auto':
        ind = methods == 0 if method == 'auto' else slice(None)
        q[ind, 0] = np.sqrt(w_sq[ind])
        q[ind, 1] = (R[ind, 2, 1] - R[ind, 1, 2]) / (4 * q[ind, 0])
        q[ind, 2] = (R[ind, 0, 2] - R[ind, 2, 0]) / (4 * q[ind, 0])
        q[ind, 3] = (R[ind, 1, 0] - R[ind, 0, 1]) / (4 * q[ind, 0])
    if method == 1 or method == 'auto':
        ind = methods == 1 if method == 'auto' else slice(None)
        q[ind, 1] = np.sqrt(x_sq[ind])
        q[ind, 0] = (R[ind, 2, 1] - R[ind, 1, 2]) / (4 * q[ind, 1])
        q[ind, 2] = (R[ind, 1, 0] + R[ind, 0, 1]) / (4 * q[ind, 1])
        q[ind, 3] = (R[ind, 0, 2] + R[ind, 2, 0]) / (4 * q[ind, 1])
    if method == 2 or method == 'auto':
        ind = methods == 2 if method == 'auto' else slice(None)
        q[ind, 2] = np.sqrt(y_sq[ind])
        q[ind, 0] = (R[ind, 0, 2] - R[ind, 2, 0]) / (4 * q[ind, 2])
        q[ind, 1] = (R[ind, 1, 0] + R[ind, 0, 1]) / (4 * q[ind, 2])
        q[ind, 3] = (R[ind, 2, 1] + R[ind, 1, 2]) / (4 * q[ind, 2])
    if method == 3 or method == 'auto':
        ind = methods == 3 if method == 'auto' else slice(None)
        q[ind, 3] = np.sqrt(z_sq[ind])
        q[ind, 0] = (R[ind, 1, 0] - R[ind, 0, 1]) / (4 * q[ind, 3])
        q[ind, 1] = (R[ind, 0, 2] + R[ind, 2, 0]) / (4 * q[ind, 3])
        q[ind, 2] = (R[ind, 2, 1] + R[ind, 1, 2]) / (4 * q[ind, 3])

    if debug or plot:
        debugData = dict(
            R=R,
            q=q,
            q_euler=eulerAngles(q),
            q_norm=vecnorm(q),
        )
        if plot:
            quatFromRotMat_debugPlot(debugData, plot)
        if debug:
            return q, debugData

    return q


def quatFromRotMat_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatFromRotMat'))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224)

        _plotQuatEuler(ax2, ax3, debug, 'q', 'output')
        _plotRotmat(ax1, debug['R'], 'input')
        fig.tight_layout()


def quatToRotVec(quat, debug=False, plot=False):
    """
    Converts quaternions to rotation vectors, i.e., vectors whose length is the rotation angle and whose direction is
    the rotation axis.

    Each input quaternion is converted independently. In order to determine a rotation vector that represents
    the change of orientation in a time series, use :meth:`quatToGyrStrapdown` and set the sampling rate to 1 Hz.

    :param quat: input quaternions (..., 4)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: output rotation vectors in rad,, (..., 3)
        - **debug**: dict with debug values (only if debug==True)
    """
    angle = quatAngle(quat)
    axis = quatAxis(quat)
    rotvec = angle[..., None] * axis

    if debug or plot:
        debugData = dict(
            quat=quat,
            quat_euler=eulerAngles(quat),
            quat_norm=vecnorm(quat),
            angle=angle,
            axis=axis,
            rotvec=rotvec,
            rotvec_norm=vecnorm(rotvec),
        )
        if plot:
            quatToRotVec_debugPlot(debugData, plot)
        if debug:
            return rotvec, debugData
    return rotvec


def quatToRotVec_debugPlot(debug, fig):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatToRotVec'))
        (ax1, ax2, ax3) = fig.subplots(3, 1, sharex=False)

        style = _plotQuatEuler(ax1, ax2, debug, 'quat', 'input')
        ax3.plot(np.rad2deg(debug['rotvec']))
        ax3.plot(np.rad2deg(debug['rotvec_norm']), style, color='k', alpha=0.3, lw=1)
        ax3.set_title(f'rotvec output in °, {debug["rotvec"].shape}')
        ax3.legend(['x', 'y', 'z', 'norm'])
        ax3.grid()
        fig.tight_layout()


def quatFromRotVec(rotvec, debug=False, plot=False):
    """
    Converts rotation vectors, i.e., vectors whose length is the rotation angle and whose direction is
    the rotation axis, to quaternions.

    Each input rotation vector is converted independently. In order to determine a quaternion that represents
    the change of orientation in a time series, use :meth:`quatFromGyrStrapdown` and set the sampling rate to 1 Hz.

    This function is just a convenient way to call ``quatFromAngleAxis(vecnorm(rotvec), rotvec)``.

    :param rotvec: input rotation vectors in rad, (..., 3)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: output quaternion array, (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    quat = quatFromAngleAxis(vecnorm(rotvec), rotvec, debug=debug or plot)
    if debug or plot:
        quat, debugData = quat
        debugData['function'] = 'quatFromRotVec'  # set correct title in debug plot
        if plot:
            quatFromRotVec_debugPlot(debugData, plot)
        if debug:
            return quat, debugData
    return quat


def quatFromRotVec_debugPlot(debug, fig=None):
    quatFromAngleAxis_debugPlot(debug, fig)


def quatToGyrStrapdown(q, rate, debug=False, plot=False):
    """
    Generate simulated gyroscope measurements from a quaternion timeseries.

    Each output angular rate sample represents the change of orientation between the current quaternion and the
    previous quaternion. Use :meth:`quatToRotVec` and multiply with the sampling rate if you want to independently
    convert rotation quaternions to the corresponding gyroscope measurements.

    >>> qmt.quatToGyrStrapdown(np.array([[0, 0, 1, 0], [0, 0, 0, 1]]), 0.2)
    array([[ 0.          0.          0.        ],
          [-0.62831853  0.          0.        ]])

    :param q: quaternion input array, (N, 4)
    :param rate: sampling rate in Hz
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **gyr**: angular rate output array in rad/s, (N, 3)
        - **debug**: dict with debug values (only if debug==True)
    """
    q = np.asarray(q, float).copy()

    N = q.shape[0]
    assert q.shape == (N, 4)
    if not allUnitNorm(q):
        raise ValueError('q does not contain unit quaternions')

    gyr = np.zeros((N, 3))
    dq = qrel(q[:-1], q[1:])
    dq = posScalar(dq)
    gyr[1:] = quatToRotVec(dq) * rate

    if debug or plot:
        debugData = dict(
            q=q,
            q_euler=eulerAngles(q),
            q_norm=vecnorm(q),
            gyr=gyr,
            gyr_norm=vecnorm(gyr),
        )
        if plot:
            quatToGyrStrapdown_debugPlot(debugData, plot)
        if debug:
            return gyr, debugData

    return gyr


def quatToGyrStrapdown_debugPlot(debug, fig):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatToGyrStrapdown'))
        (ax1, ax2, ax3) = fig.subplots(3, 1, sharex=False)

        style = _plotQuatEuler(ax1, ax2, debug, 'q', 'input')
        ax3.plot(np.rad2deg(debug['gyr']))
        ax3.plot(np.rad2deg(debug['gyr_norm']), style, color='k', alpha=0.3, lw=1)
        ax3.set_title(f'gyr output in °/s, {debug["gyr"].shape}')
        ax3.legend(['x', 'y', 'z', 'norm'])
        ax3.grid()
        fig.tight_layout()


def quatFromGyrStrapdown(gyr, rate, debug=False, plot=False):
    """
    Create strap-down integrated quaternion timeseries from gyroscope measurements.

    Starting with an initial orientation of [1 0 0 0], each input angular rate sample is converted to a rotation
    quaternion and multiplied to the previous orientation. Use :meth:`quatFromRotVec` after dividing by the sampling
    rate if you want to independently convert gyroscope measurements to the corresponding rotation quaternions.

    Note that, to avoid looping over the data in Python, this function is implemented in C++.

    :param gyr: angular rate input array, (N, 3)
    :param rate: sampling rate in Hz
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **out**: quaternion output array in rad/s, (N, 4)
        - **debug**: dict with debug values (only if debug==True)
    """
    gyr = np.ascontiguousarray(gyr, float)
    N = gyr.shape[0]
    assert gyr.shape == (N, 3)

    from qmt.cpp.quaternion import quatFromGyrStrapdown as c_quatFromGyrStrapdown
    out = c_quatFromGyrStrapdown(gyr, rate)

    if debug or plot:
        debugData = dict(
            gyr=gyr,
            gyr_norm=vecnorm(gyr),
            out=out,
            out_euler=eulerAngles(out),
            out_norm=vecnorm(out),
        )
        if plot:
            quatFromGyrStrapdown_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def quatFromGyrStrapdown_debugPlot(debug, fig):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatFromGyrStrapdown'))
        (ax1, ax2, ax3) = fig.subplots(3, 1, sharex=False)

        style = _plotQuatEuler(ax2, ax3, debug, 'out', 'output')
        ax1.plot(np.rad2deg(debug['gyr']))
        ax1.plot(np.rad2deg(debug['gyr_norm']), style, color='k', alpha=0.3, lw=1)
        ax1.set_title(f'gyr input in °/s, {debug["gyr"].shape}')
        ax1.legend(['x', 'y', 'z', 'norm'])
        ax1.grid()
        fig.tight_layout()


def quatFrom2Axes(x=None, y=None, z=None, exactAxis=None, verbose=False, debug=False, plot=False):
    """
    Determine quaternion from two given basis vectors.

    The orientation of CS A relative to CS B is determined from 2 basis vectors of CS A in B's coordinates.
    The axes x, y and z are 3-dimensional vectors and one of them has to be None or [0, 0, 0].
    exactAxis is None, 'x', 'y' or 'z' and describes which of the two non-zero axes is assumed to be exact. If None is
    passed, both axes will be adjusted equally.

    >>> q = qmt.quatFrom2Axes(x=[1, 1, 0], y=[1, 0, 1])
    array([0.44403692, 0.7690945 , 0.44403692, 0.11897933])

    :param x: x-axis input array, None or (N, 3) or (3,)
    :param y: y-axis input array, None or (N, 3) or (3,)
    :param z: z-axis input array, None or (N, 3) or (3,)
    :param exactAxis: describes which axis should be adjusted if they are non-orthogonal; options: None, 'x', 'y', 'z'
    :param verbose: enables printing of a non-orthogonality warning if the angle between the axes is < 30°
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: quaternion output array, (N, 4) or (4,)
        - **debug**: dict with debug values (only if debug==True)
    """
    axisIdentifiers = {
        1: 1, '1': 1, 'x': 1, 'X': 1, 'i': 1,
        2: 2, '2': 2, 'y': 2, 'Y': 2, 'j': 2,
        3: 3, '3': 3, 'z': 3, 'Z': 3, 'k': 3,
    }

    x = np.zeros(3, float) if x is None else np.asarray(x, float)
    y = np.zeros(3, float) if y is None else np.asarray(y, float)
    z = np.zeros(3, float) if z is None else np.asarray(z, float)

    # if the input quaternions are 1D arrays, we also want to return a 1D output
    is1D = max(x.ndim, y.ndim, z.ndim) < 2

    # but to be able to use the same indexing in all cases, make sure everything is in 2D arrays
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)

    N = max(x.shape[0], y.shape[0], z.shape[0])

    # check the dimensions
    assert x.shape == (N, 3) or x.shape == (1, 3)
    assert y.shape == (N, 3) or y.shape == (1, 3)
    assert z.shape == (N, 3) or z.shape == (1, 3)

    # find the zero axis
    zero = [not np.any(axis) for axis in (x, y, z)]
    if sum(zero) != 1:
        raise RuntimeError('exactly one axis has to be (always) zero and all other axes have to be non-zero')

    # determine index of missing axis that will be determined automatically
    zeroAxis = zero.index(True)

    R = np.column_stack(np.broadcast_arrays(x, y, z)).reshape(N, 3, 3)
    # get the column vectors from the matrices to get the transposed matrices
    x1 = R[:, :, 0]
    y1 = R[:, :, 1]
    z1 = R[:, :, 2]
    # get the transposed matrices
    R = np.column_stack([x1, y1, z1]).reshape(N, 3, 3)

    # check the angle between the two given axes
    if verbose or plot or debug:
        angle = angleBetween2Vecs(R[:, :, (zeroAxis + 1) % 3], R[:, :, (zeroAxis + 2) % 3])
        print(angle)
        with np.errstate(invalid='ignore'):
            nonOrthogonalAxesFound = np.any(angle < np.deg2rad(30))
        print(nonOrthogonalAxesFound)
    if verbose and nonOrthogonalAxesFound:
        nonOrthogonalCount = np.sum(angle < np.deg2rad(30))
        percentage = nonOrthogonalCount/N*100
        print(f'quatFrom2Axes: warning: the two given axes are far from orthogonal (<30°) for '
              f'{nonOrthogonalCount} of {N} samples ({percentage:.2f} %)')

    if exactAxis is not None:
        exactAxis = axisIdentifiers[str(exactAxis)] - 1
        if exactAxis == zeroAxis:
            raise RuntimeError('exactAxis has not to be one of the two zero-axis')

        # determine index of the approximate axis
        approxAxis = (set([0, 1, 2]) - set([exactAxis, zeroAxis])).pop()

        # make the approximate axis orthogonal to the exact axis
        R[:, :, approxAxis] = np.cross(np.cross(R[:, :, exactAxis], R[:, :, approxAxis]), R[:, :, exactAxis])

    else:  # equally adjust both axis
        # determine indices of the two given axes
        axis1Ind, axis2Ind = list(set([0, 1, 2]) - set([zeroAxis]))

        # calculate adjusted axes when assuming the other axis is exact
        axis1Adj = np.cross(np.cross(R[:, :, axis2Ind], R[:, :, axis1Ind]), R[:, :, axis2Ind])
        axis2Adj = np.cross(np.cross(R[:, :, axis1Ind], R[:, :, axis2Ind]), R[:, :, axis1Ind])

        # calculate the mean of the original axis and the adjust axes (however, both have to be normalized)
        R[:, :, axis1Ind] = (normalized(R[:, :, axis1Ind]) + normalized(axis1Adj)) / 2
        R[:, :, axis2Ind] = (normalized(R[:, :, axis2Ind]) + normalized(axis2Adj)) / 2

    # calculate missing axis from the other two
    R[:, :, zeroAxis] = np.cross(R[:, :, (zeroAxis + 1) % 3], R[:, :, (zeroAxis + 2) % 3])

    # normalize all rows
    for i in range(3):
        R[:, :, i] = normalized(R[:, :, i])

    if is1D:
        R = R.reshape((3, 3))
        x = x.reshape((3,))
        y = y.reshape((3,))
        z = z.reshape((3,))

    q = quatFromRotMat(R)

    if debug or plot:
        debugData = dict(
            x=x,
            y=y,
            z=z,
            exactAxis=exactAxis,
            R=R,
            q=q,
            q_norm=vecnorm(q),
            q_euler=eulerAngles(q),
            nonOrthogonalAxesFound=nonOrthogonalAxesFound,
            angle=angle,
            is1D=is1D,
        )
        if plot:
            quatFrom2Axes_debugPlot(debugData, plot)
        if debug:
            return q, debugData
    return q


def quatFrom2Axes_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatFrom2Axes'))
        (ax1, ax2, ax3), (ax4, ax5, ax6) = fig.subplots(2, 3, sharex=True)

        style = _plotQuatEuler(ax5, ax6, debug, 'q', 'output')

        for ax, val in (ax1, 'x'), (ax2, 'y'), (ax3, 'z'):
            ax.plot(debug[val].reshape(-1, 3), style)
            ax.legend('xyz')
            ax.grid()
            ax.set_title(f'{val} axis, {debug[val].shape}')

        ax4.plot(np.rad2deg(debug["angle"]), style)
        ax4.set_title('angle between 2 axes in °')
        ax4.text(0, 3, f'warning: {debug["nonOrthogonalAxesFound"]}')
        ax4.axhspan(0, 30, color='r', alpha=0.2)
        ax4.grid()
        fig.tight_layout()


def quatFromVectorObservations(v, w, weights=None, debug=False, plot=False):
    """
    Calculates quaternion from multiple (weighted) vector observations (Wahba's problem).

    The output is a quaternion that (tries to) satisfy ``qmt.rotate(quat, v) ≈ w``.
    Note that the input vectors v and w will be normalized internally.

    This function uses the SVD implementation as described in *Markley, F. L. Attitude Determination using Vector
    Observations and the Singular Value Decomposition, Journal of the Astronautical Sciences, 1988, 38:245–258*
    (cf. https://en.wikipedia.org/wiki/Wahba%27s_problem).

    :param v: input vectors v, shape (N, 3)
    :param w: input vectors w, shape (N, 3)
    :param weights: optional weight vector, shape (N,)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: output quaternion, shape (4,)
        - **debug**: dict with debug values (only if debug==True)
    """
    v = np.asarray(v, float)
    w = np.asarray(w, float)

    v = normalized(v)
    w = normalized(w)

    N = v.shape[0]
    assert N >= 2, 'at least two observations are required'

    if weights is None:
        weights = np.ones(N)
    else:
        weights = np.asarray(weights, float)

    assert v.shape == (N, 3), (v.shape, N)
    assert w.shape == (N, 3), (w.shape, N)
    assert weights.shape == (N,)

    B = np.einsum('bi,bo,b->oi', v, w, weights)
    assert B.shape == (3, 3)
    # the einsum is equivalent to:
    # B = np.zeros((3, 3))
    # for vi, wi, weight in zip(v, w, weights):
    #     B += weight * np.outer(wi, vi)

    u, s, vh = np.linalg.svd(B)

    M = np.diag([1.0, 1.0, np.linalg.det(u) * np.linalg.det(vh)])
    R = u @ M @ vh
    quat = quatFromRotMat(R)

    if debug or plot:
        debugData = dict(
            v=v,
            w=w,
            weights=weights,
            v_rot=rotate(quat, v),
            residual_angle=angleBetween2Vecs(rotate(quat, v), w),
            N=N,
            B=B,
            u=u,
            s=s,
            vh=vh,
            M=M,
            quat=quat,
            quat_norm=vecnorm(quat),
            quat_euler=eulerAngles(quat),
        )
        if plot:
            quatFromVectorObservations_debugPlot(debugData, plot)
        if debug:
            return quat, debugData
    return quat


def quatFromVectorObservations_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatFromVectorObservations'))
        ax1, ax2, ax3, ax4 = fig.subplots(4, 1, sharex=True)

        v = debug['v']
        w = debug['w']
        vrot = debug['v_rot']
        residualAngle = debug['residual_angle']
        style = '.-' if debug['N'] <= 100 else '-'

        ax1.set_title(f'first input vector v, {v.shape}')
        ax1.plot(v.reshape(-1, v.shape[-1]), style)
        ax1.grid()

        ax2.set_title(f'rotated first input vector rotate(q, v), {vrot.shape}')
        ax2.plot(vrot.reshape(-1, vrot.shape[-1]), style)
        ax2.grid()

        ax3.set_title(f'second input vector w, {w.shape}')
        ax3.plot(w.reshape(-1, w.shape[-1]), style)
        ax3.grid()

        for ax in ax1, ax2, ax3:
            ax.set_ylim(-1.05, 1.05)

        ax4.plot(np.rad2deg(residualAngle), style)
        ax4.set_title('angle between rotate(q, v) and w [°]')
        ax4.ticklabel_format(useOffset=False)
        ax4.grid()
        fig.tight_layout()


def headingInclinationAngle(quat, debug=False, plot=False):
    """
    Decompose quaternion into heading and inclination angle.

    The heading angle describes a rotation around the vertical z-axis, and the inclination angle describes a rotation
    around a horizontal axis. Note that this decomposition is not reversible (a third angle that describes the
    horizontal axis is missing).

    The result of this function can also be obtained via ``qmt.quatProject(quat, [0, 0, 1])``: `heading` is equal
    to `projAngle` and `inclination` is equal to `abs(resAngle)`.

    :param quat: input quaternion array, (..., 4)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **heading**: heading angle in rad, shape (...,)
        - **inclination**: inclination angle in rad, shape (...,)
        - **debug**: dict with debug values (only if debug==True)
    """
    quat = np.asarray(quat, float)
    assert quat.shape[-1] == 4

    heading = wrapToPi(2 * np.arctan2(quat[..., 3], quat[..., 0]))
    inclination = 2 * np.arccos(np.clip(np.sqrt(quat[:, 3] ** 2 + quat[:, 0] ** 2), -1, 1))

    if debug or plot:
        debugData = dict(
            quat=quat,
            quat_norm=vecnorm(quat),
            quat_euler=eulerAngles(quat),
            heading=heading,
            inclination=inclination,
        )
        if plot:
            headingInclinationAngle_debugPlot(debugData, plot)
        if debug:
            return heading, inclination, debugData
    return heading, inclination


def headingInclinationAngle_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('headingInclinationAngle'))
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)

        style = _plotQuatEuler(ax1, ax2, debug, 'quat', 'input')

        ax3.plot(np.rad2deg(debug['heading'].flatten()), style)
        ax3.set_title(f'heading angle in °, {debug["heading"].shape}')
        ax3.grid()

        ax4.plot(np.rad2deg(debug['inclination'].flatten()), style)
        ax4.set_title(f'inclination angle in °, {debug["heading"].shape}')
        ax4.grid()

        fig.tight_layout()


def slerp(q0, q1, t, debug=False, plot=False):
    """
    Spherical linear interpolation of quaternions.

    This function interpolates between q0 and q1 at an interpolation parameter t that is between 0 (for q0) and 1 (for
    q1). Both q0 and q1 can either be a single quaternion or an array of multiple quaternions, and t can be scalar or
    an array with matching shape (numpy broadcasting rules are applied).

    This function calculates the shortest path interpolation. If q0 and q1 are very similar, a fallback to linear
    interpolation (lerp) is implemented for numeric reasons.

    See :meth:`quatInterp` for an alternative function to use slerp that is useful for interpolating time series of
    quaternions at different sampling times.

    :param q0: first input quaternion (t=0), numpy array with shape (..., 4)
    :param q1: second input quaternion (t=1), numpy array with shape (..., 4)
    :param t: interpolation factor (between 0 and 1), scalar or (...) numpy array
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **out**: interpolated quaternion, shape (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    # useful links:
    # https://en.wikipedia.org/wiki/Slerp
    # https://blog.magnum.graphics/backstage/the-unnecessarily-short-ways-to-do-a-quaternion-slerp/
    # http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/

    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    t = np.asarray(t, float)

    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    # calculate dot product
    d = np.sum(q0 * q1, axis=-1)

    # ensure that the dot product is positive by (virtually) flipping q1
    # (this is done by the flip variable that is -1 or 1)
    flip = 2 * (d >= 0) - 1
    d = np.abs(d)

    # calculate factors used for slerp
    # (note that the division by sin(theta) is intentially omitted as we will normalize anyway)
    theta = np.arccos(np.minimum(d, 1))
    k0 = np.sin((1 - t) * theta)
    k1 = np.sin(t * theta)

    # use linear interpolation (lerp) if the quaternions are very close to each other
    lerp = d > 0.999999
    if np.any(lerp):
        if k1.ndim == 0:  # item assignment not possible in scalar case
            k1 = np.float64(t)
            k0 = np.float64(1 - t)
        else:
            k1[lerp] = (np.ones_like(theta) * t)[lerp]
            k0[lerp] = 1 - k1[lerp]

    out = normalized(k0[..., None] * q0 + (flip * k1)[..., None] * q1)

    if debug or plot:
        debugData = dict(
            q0=q0,
            q0_norm=vecnorm(q0),
            q0_euler=eulerAngles(q0),
            q1=q1,
            q1_norm=vecnorm(q1),
            q1_euler=eulerAngles(q1),
            t=t,
            flip=flip,
            theta=theta,
            lerp=lerp,
            k0=k0,
            k1=k1,
            out=out,
            out_norm=vecnorm(out),
            out_euler=eulerAngles(out),
        )
        if plot:
            slerp_debugPlot(debugData, plot)
        if debug:
            return out, debugData

    return out


def slerp_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('slerp'))
        (ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8) = fig.subplots(4, 2, sharex=True)

        _plotQuatEuler(ax1, ax2, debug, 'q0', 'input')
        _plotQuatEuler(ax3, ax4, debug, 'q1', 'input')
        style = _plotQuatEuler(ax7, ax8, debug, 'out', 'output')

        ax5.plot(debug['t'].flatten(), label='t')
        ax5.grid()
        ax5.legend()
        ax5.set_title(f'interpolation factor t, {debug["t"].shape}')

        ax6.plot(debug['flip'].flatten(), style, alpha=0.5, label='flip')
        ax6.plot(debug['lerp'].flatten(), style, alpha=0.5, label='lerp fallback')
        ax6.grid()
        ax6.legend()
        ax6.set_title(f'flip, {debug["flip"].shape}; lerp, {debug["lerp"].shape}')

        fig.tight_layout()


def quatInterp(quat, ind, extend=True, debug=False, plot=False):
    """
    Interpolates quaternion timeseries at (non-integer) sampling times using slerp.

    Sampling indices are in the range 0..N-1. For values outside of this range, depending on "extend", the first/last
    element or NaN is returned. If the input consists of 2 quaternions and ind=0.5, the result is the 50/50
    interpolation between the two input quaternions.

    See also :meth:`qmt.vecInterp` for linear interpolation and :meth:`slerp` for a more generic slerp implementation.

    >>> qmt.quatInterp([[1, 0, 0, 0], [0, 0, 1, 0]], [0, 0.1, 0.5, 1])
    array([[1.        , 0.        , 0.        , 0.        ],
       [0.98768834, 0.        , 0.15643447, 0.        ],
       [0.70710678, 0.        , 0.70710678, 0.        ],
       [0.        , 0.        , 1.        , 0.        ]])

    :param quat: array of input quaternions (N, 4)
    :param ind: vector containing the sampling indices for desired output, (M,)
    :param extend: if True, the input data is virtually extended by the first/last value
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **out**: interpolated quaternion output array, (M, 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    quat = np.asarray(quat, float)
    ind = np.asarray(ind, float)
    isScalar = ind.ndim == 0
    ind = np.atleast_1d(ind)
    N = quat.shape[0]
    M = ind.shape[0]
    assert quat.shape == (N, 4)
    assert ind.shape == (M,)

    ind0 = np.clip(np.floor(ind).astype(int), 0, N-1)
    ind1 = np.clip(np.ceil(ind).astype(int), 0, N-1)
    q0 = quat[ind0]
    q1 = quat[ind1]

    out = slerp(q0, q1, ind-ind0)

    if not extend:
        out[ind < 0] = np.nan
        out[ind > N - 1] = np.nan

    if isScalar:
        out = out.reshape((4,))

    if debug or plot:
        debugData = dict(
            quat=quat,
            quat_norm=vecnorm(quat),
            quat_euler=eulerAngles(quat),
            N=N,
            M=M,
            ind=ind,
            out=out,
            out_norm=vecnorm(out),
            out_euler=eulerAngles(out),
        )
        if plot:
            quatInterp_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def quatInterp_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatInterp'))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        _plotQuatEuler(ax1, ax2, debug, 'quat', 'input')

        ax1.set_prop_cycle(None)
        ax1.plot(debug['ind'], debug['out'].reshape(-1, 4), 'x-', alpha=0.5)
        ax1.plot(debug['ind'], debug['out_norm'].flatten(), 'x-', color='k', alpha=0.3, lw=1)
        ax1.set_title(f'input quat [{debug["quat"].shape}, circles]; '
                      f'interpolated quat: [{debug["out"].shape}, crosses]')

        ax2.set_prop_cycle(None)
        ax2.plot(debug['ind'], np.rad2deg(debug['out_euler'].reshape(-1, 3)), 'x-', alpha=0.5)
        print(np.rad2deg(debug['out_euler'].reshape(-1, 3)).shape)

        fig.tight_layout()


def randomQuat(N=None, angle=None, debug=False, plot=False):
    """
    Generate random quaternions.

    If N is None (default), a single random quaternion is returned as an array with shape (4,).
    If N is an integer, N random quaternions are returned as an (N, 4) array.
    If N is a tuple, it denotes the shape of the output quaternions, e.g. N=(5, 20) returns 100 random quaternions as a
    (5, 20, 4) array.

    If the optional parameter angle is not None, only the axis is randomly generated and quaternions with the random
    axis and the given angle are returned.

    >>> qmt.randomQuat(N=2)
        array([[ 0.07232174  0.37107947 -0.27374854 -0.88438189]
              [ 0.35838808  0.25856014 -0.07512213  0.89390229]])

    See also :meth:`qmt.randomUnitVec`.

    :param N: number of quaternions to generate
    :param angle: if not None, the generated quaternions will have a random axis, but this fixed angle (in rad)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: quaternion output array, (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    if angle is None:
        quat = randomUnitVec(N, 4)
    else:
        axis = randomUnitVec(N, 3)
        quat = quatFromAngleAxis(angle, axis)

    if debug or plot:
        debugData = dict(
            quat=quat,
            quat_euler=eulerAngles(quat),
            quat_norm=vecnorm(quat),
        )
        if plot:
            randomQuat_debugPlot(debugData, plot)
        if debug:
            return quat, debugData
    return quat


def randomQuat_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('randomQuat'))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        _plotQuatEuler(ax1, ax2, debug, 'quat', 'output')
        fig.tight_layout()


def averageQuat(quats, debug=False, plot=False):
    """
    Calculates the average of multiple quaternions.

    For more details, see *Markey, Averaging Quaternions* (http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf).

    :param quats: input quaternion array, shape (..., 4)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: averaged quaternion, shape (4,)
        - **debug**: dict with debug values (only if debug==True)
    """

    quats = np.asarray(quats, float)
    assert quats.shape[-1] == 4
    quats = quats.reshape((-1, 4))
    if not allUnitNorm(quats):
        raise ValueError('quats does not contain unit quaternions')

    M = np.einsum('bi,bo->io', quats, quats)
    assert M.shape == (4, 4)
    # the einsum is equivalent to:
    # M = np.zeros((4, 4))
    # for q in quats:
    #     M += np.outer(q, q)
    w, v, = np.linalg.eig(M)
    ind = np.argmax(w)
    quat = v[:, ind]

    if debug or plot:
        debugData = dict(
            quats=quats,
            quats_euler=eulerAngles(quats),
            quats_norm=vecnorm(quats),
            quat=quat,
            quat_euler=eulerAngles(quat),
            quat_norm=vecnorm(quat),
        )
        if plot:
            averageQuat_debugPlot(debugData, plot)
        if debug:
            return quat, debugData
    return quat


def averageQuat_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('averageQuat'))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        N = debug['quats'].shape[0]
        _plotQuatEuler(ax1, ax2, debug, 'quats')
        ax1.set_prop_cycle(None)
        ax2.set_prop_cycle(None)
        for i, q in enumerate(debug['quat']):
            ax1.axhline(q, color='k', ls='--')
            ax1.plot(-0.01*N, q, '>', color=f'C{i}')
            ax1.plot(1.01*N, q, '<', color=f'C{i}')
        for i, angle in enumerate(np.rad2deg(debug['quat_euler'])):
            ax2.axhline(angle, color='k', ls='--')
            ax2.plot(-0.01*N, angle, '>', color=f'C{i}')
            ax2.plot(1.01*N, angle, '<', color=f'C{i}')
        fig.tight_layout()


def quatUnwrap(quat, init=(1, 0, 0, 0), debug=False, plot=False):
    """
    Unwraps quaternion timeseries to prevent unnecessary jumps in plots.

    Unwraps a sequence of quaternions by multiplying elements with -1 if that reduces the Euclidean norm of the
    difference to the predecessor. This does not change the rotation but prevents unnecessary jumps when plotting the
    quaternion. For an alternative function, see :func:`qmt.posScalar`.

    :param quat: quaternion array, (N, 4)
    :param init: quaternion to compare quat[0] to, default: [1, 0, 0, 0]
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: Nx4 quaternion output array
        - **debug**: dict with debug values (only if debug==True)
    """
    assert quat.ndim == 2
    assert quat.shape[1] == 4
    assert allUnitNorm(quat)

    init = np.asarray(init, float)
    assert init.shape == (4,)

    valid = ~np.any(np.isnan(quat), axis=1)

    dist = np.sum((quat[valid] - np.vstack([init, quat[valid][:-1]]))**2, axis=1)

    # quick test to confirm that 2 is the correct boundary:
    # q = qmt.randomQuat(10000)
    # dist1 = np.linalg.norm(q - q[0], axis=1)
    # dist2 = np.linalg.norm(q + q[0], axis=1)
    # dist = np.sum((q - q[0]) ** 2, axis=1)
    # plt.figure()
    # plt.plot(dist[dist1 > dist2])
    # plt.plot(dist[dist2 > dist1])
    wrap = dist > 2
    changeind = np.zeros(quat.shape[0], bool)
    changeind[valid] = (np.cumsum(wrap) % 2).astype(bool)

    out = quat.copy()
    out[changeind] *= -1

    if debug or plot:
        debugData = dict(
            quat=quat,
            quat_norm=vecnorm(quat),
            quat_euler=eulerAngles(quat),
            init=init,
            valid=valid,
            dist=dist,
            wrap=wrap,
            changeind=changeind,
            out=out,
            out_norm=vecnorm(out),
            out_euler=eulerAngles(out),
        )
        if plot:
            quatUnwrap_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def quatUnwrap_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('quatUnwrap'))
        axes = fig.subplots(3, 1, sharex=True)

        style = _plotQuatEuler(axes[0], None, debug, 'quat', 'input')

        axes[1].plot(np.arange(debug['quat'].shape[0])[debug['valid']], debug['dist'], style, label='distance')
        axes[1].plot(debug['changeind'], style,  label='change')
        axes[1].legend()
        axes[1].grid()
        axes[1].set_title('distance (change if distance > 2)')

        _plotQuatEuler(axes[2], None, debug, 'out', 'output')
        fig.tight_layout()


def posScalar(q, debug=False, plot=False):
    """
    Returns a copy of the input quaternion that uniquely describes rotations by making the scalar part positive.

    Quaternions with negative scalar part are multiplied with -1, which does not influence the described rotation.
    While this makes the resulting quaternion unique, it might lead to jumps when plotting a quaternion timeseries.
    For an alternative function, see :func:`qmt.quatUnwrap`.

    If the scalar part is (close to) zero, the first non-zero component is made positive in order to ensure a unique
    output in those edge cases.

    >>>  qmt.posScalar([[1, 0, 1, 0], [-1, 1, 1, 1]])
        array([[ 1.,  0.,  1.,  0.],
           [ 1., -1., -1., -1.]])

    :param q: quaternion input array, (..., 4)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **output**: quaternion output array, (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    q = np.asarray(q, float)
    assert q.shape[-1] == 4

    # change quaternions with negative scalar compenent
    ind = ((q[..., 0] < 0) |
           (np.isclose(q[..., 0], 0) & (q[..., 1] < 0)) |
           (np.isclose(q[..., 0], 0) & np.isclose(q[..., 1], 0) & (q[..., 2] < 0)) |
           (np.isclose(q[..., 0], 0) & np.isclose(q[..., 1], 0) & np.isclose(q[..., 2], 0) & (q[..., 3] < 0)))
    out = np.zeros(q.shape, float)
    out[ind] = -q[ind]
    out[~ind] = q[~ind]

    if debug or plot:
        debugData = dict(
            q=q,
            q_norm=vecnorm(q),
            q_euler=eulerAngles(q),
            out=out,
            out_norm=vecnorm(out),
            out_euler=eulerAngles(out),
        )
        if plot:
            posScalar_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def posScalar_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        fig.suptitle(AutoFigure.title('posScalar'))
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)

        _plotQuatEuler(ax1, ax2, debug, 'q', 'input')
        _plotQuatEuler(ax3, ax4, debug, 'out', 'out')
        fig.tight_layout()


def deltaQuat(delta, debug=False, plot=False):
    """
    Generates rotation quaternions around the vertical z-axis.

    The output of this method is supposed to be left-multiplied to orientation quaternions in order to change the
    heading by the angle delta.

    This function is just a convenient way to call ``quatFromAngleAxis(delta, [0, 0, 1])``. See
    :meth:`qmt.addHeading` for a method to directly apply this heading offset to quaternions.

    :param delta: input heading angles in rad, shape (....)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **quat**: output quaternion array, (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    quat = quatFromAngleAxis(delta, np.array([0, 0, 1], float), debug=debug or plot)
    if debug or plot:
        quat, debugData = quat
        debugData['function'] = 'deltaQuat'  # set correct title in debug plot
        if plot:
            deltaQuat_debugPlot(debugData, plot)
        if debug:
            return quat, debugData
    return quat


def deltaQuat_debugPlot(debug, fig=None):
    quatFromAngleAxis_debugPlot(debug, fig)


def addHeading(q, delta, debug=False, plot=False):
    """
    Add heading angle offset to existing quaternions.

    :param q: input quaternion wrt. a z-up coordinate system, (..., 4)
    :param delta: heading angle in rad, shape (...)
    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **out**: quaternions with modified heading angle., (..., 4)
        - **debug**: dict with debug values (only if debug==True)
    """

    out = qmult(deltaQuat(delta), q)

    if debug or plot:
        debugData = dict(
            q=q,
            q_euler=eulerAngles(q),
            q_norm=vecnorm(q),
            headingAngle=delta,
            deltaQuat=deltaQuat(delta),
            out=out,
            out_euler=eulerAngles(out),
            out_norm=vecnorm(out),

        )
        if plot:
            addHeading_debugPlot(debugData, plot)
        if debug:
            return out, debugData
    return out


def addHeading_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        axes = fig.subplots(3, 2, sharex=True)
        fig.suptitle(AutoFigure.title('addHeading'))

        for i, val in [(0, 'q'), (2, 'out')]:
            desc = 'input' if i == 0 else 'output'
            style = _plotQuatEuler(axes[i, 0], axes[i, 1], debug, val, desc)

        headingAngle = np.asarray(debug['headingAngle'])
        axes[1, 0].plot(np.rad2deg(headingAngle.flatten()), style)
        axes[1, 0].set_title(f'heading angle in °: {headingAngle.shape}')
        axes[1, 0].grid()
        axes[1, 1].plot(debug['deltaQuat'].reshape(-1, 4), style)
        axes[1, 1].set_title('heading quaternion')
        axes[1, 1].legend('wxyz')
        axes[1, 1].grid()
        fig.tight_layout()
