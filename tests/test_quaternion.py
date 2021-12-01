# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import qmt
import numpy as np
import pytest

qmult_data = [  # [q1, q2, res] with q1*q2=res
    [
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float),
        np.array([[0.5, 0.5, 0.5, 0.5], [1/np.sqrt(2), 1/np.sqrt(2), 0, 0]], float),
        np.array([[0.5, 0.5, 0.5, 0.5], [-1/np.sqrt(2),  1/np.sqrt(2), 0, 0]], float)
    ],
]

quat_data = [  # various unit quaternions
    np.array([1, 0, 0, 0], float),
    np.array([0, 1, 0, 0], float),
    np.array([0.5, 0.5, 0.5, 0.5], float),
    np.array([0.5, 0.5, -0.5, -0.5], float),
    np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0], float),
    np.array([-1/np.sqrt(2),  1/np.sqrt(2), 0, 0], float),
]

vec_data = [  # various vectors (same length as quat_data)
    np.array([1, 0, 0], float),
    np.array([0, 1, 0], float),
    np.array([0, 0, 1], float),
    np.array([1, 2, 3], float),
    np.array([0, 0, 0], float),
    np.array([-1, -1, -1], float),
]

euler_axes = ['zyx', 'zxy', 'yxz', 'yzx', 'xzy', 'xyz', 'zxz', 'zyz', 'yxy', 'yzy', 'xyx', 'xzx']
euler_data = np.deg2rad(np.array([
    [0, 0, 0],
    [90, 0, 0],
    [-90, 0, 0],
    [0, 90, 0],
    [0, -90, 0],
    [0, 180, 0],
    [0, -180, 0],
    [23, 0, 0],
    [-23, 0, 0],
    [23, 90, 0],
    [-23, 90, 0],
    [23, -90, 0],
    [-23, -90, 0],
    [10, 20, 30],
    [-10, -20, -30],
    [-179, 10, 179],
    [-179, -42, 179],
], float))


@pytest.mark.parametrize('q1,q2,res', qmult_data)
def test_qmult(q1, q2, res):
    out = qmt.qmult(q1, q2)
    np.testing.assert_allclose(out, res)


def test_qmult_non_unit():
    q1 = np.array([[1, 0, 0, 0.1], [0, 1, 0, 0]], float)
    q2 = np.array([[0.5, 0.5, 0.5, 0.5], [1/np.sqrt(2), 1/np.sqrt(2), 0, 0]], float)

    qmt.qmult(q1, q2, strict=False)
    with pytest.raises(ValueError, match='.*unit quaternions.*'):
        qmt.qmult(q1, q2)


def test_qmult_high_dim():
    np.random.seed(42)
    q1 = qmt.randomQuat(100)
    q2 = qmt.randomQuat(100)

    out = qmt.qmult(q1.reshape((5, 20, 4)), q2.reshape((5, 20, 4))).reshape(100, 4)
    ref = qmt.qmult(q1, q2)
    np.testing.assert_allclose(out, ref)

    out = qmt.qmult(q1.reshape((5, 20, 4)), [0.5, 0.5, 0.5, 0.5])
    assert out.shape == (5, 20, 4)
    out = out.reshape(100, 4)
    ref = qmt.qmult(q1, [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(out, ref)


@pytest.mark.parametrize('q1,q2,_', qmult_data)
@pytest.mark.matlab
def test_qmult_pyvsmat(q1, q2, _):
    out = qmt.qmult(q1, q2)
    out2 = qmt.matlab.qmult(q1, q2, nargout=1)
    np.testing.assert_allclose(out, out2)


@pytest.mark.parametrize('q1,q2,_', qmult_data)
def test_qinv(q1, q2, _):
    out = qmt.qinv(q1)
    res = q1.copy()
    res[:, 1:] *= -1
    np.testing.assert_allclose(out, res)


@pytest.mark.parametrize('q1,q2,_', qmult_data)
def test_qrel(q1, q2, _):
    out = qmt.qrel(q1, q2)
    res = qmt.qmult(qmt.qinv(q1), q2)
    np.testing.assert_allclose(out, res)


@pytest.mark.parametrize('q, v', zip(quat_data, vec_data))
def test_rotate(q, v):
    out = qmt.rotate(q, v)
    res = qmt.qmult(qmt.qmult(q, [0] + v.tolist(), strict=False), qmt.qinv(q), strict=False)
    np.testing.assert_allclose(res[0], 0)
    res = res[1:]
    np.testing.assert_allclose(out, res)


def test_rotate_vectorized():
    q = np.array(quat_data)
    v = np.array(vec_data)
    assert q.shape[0] == v.shape[0]

    # (N, 4) and (N, 3)
    out = qmt.rotate(q, v)
    v_quat = np.column_stack([np.zeros(v.shape[0]), v])
    res = qmt.qmult(qmt.qmult(q, v_quat, strict=False), qmt.qinv(q), strict=False)
    np.testing.assert_allclose(res[:, 0], 0)
    res = res[:, 1:]
    np.testing.assert_allclose(out, res)

    # (N, 4) and (N,)
    out = qmt.rotate(q, v[0])
    v_quat = np.concatenate([np.zeros(1), v[0]])
    res = qmt.qmult(qmt.qmult(q, v_quat, strict=False), qmt.qinv(q), strict=False)
    np.testing.assert_allclose(res[:, 0], 0)
    res = res[:, 1:]
    assert out.shape == v.shape
    np.testing.assert_allclose(out, res)

    # (N, ) and (N, 3)
    out = qmt.rotate(q[3], v)
    v_quat = np.column_stack([np.zeros(v.shape[0]), v])
    res = qmt.qmult(qmt.qmult(q[3], v_quat, strict=False), qmt.qinv(q[3]), strict=False)
    np.testing.assert_allclose(res[:, 0], 0)
    res = res[:, 1:]
    assert out.shape == v.shape
    np.testing.assert_allclose(out, res)


def test_quatFromAngleAxis_roundtrip():
    q = np.array(quat_data)
    angle = qmt.quatAngle(q)
    axis = qmt.quatAxis(q)
    q2 = qmt.quatFromAngleAxis(angle, axis)
    # using posScalar is needed since quatAngle returns angles in range -pi...pi
    np.testing.assert_allclose(qmt.posScalar(q), qmt.posScalar(q2), atol=1e-10)


@pytest.mark.parametrize('axes', euler_axes)
@pytest.mark.parametrize('intrinsic', [True, False])
def test_quatFromToEulerAngles_roundtrip(axes, intrinsic):
    if axes[0] == axes[2]:  # second angle needs to be positive
        ref = euler_data[euler_data[:, 1] >= 0]
    else:
        ref = euler_data[np.abs(euler_data[:, 1]) < np.deg2rad(90.1)]
    quat = qmt.quatFromEulerAngles(ref, axes, intrinsic)
    angles = qmt.eulerAngles(quat, axes, intrinsic)
    np.testing.assert_allclose(qmt.wrapToPi(ref-angles), 0, atol=1e-8)


@pytest.mark.parametrize('axes', euler_axes)
@pytest.mark.parametrize('intrinsic', [True, False])
def test_quatToFromEulerAngles_roundtrip(axes, intrinsic):
    angles = qmt.eulerAngles(quat_data, axes, intrinsic)
    quat = qmt.quatFromEulerAngles(angles, axes, intrinsic)
    np.testing.assert_allclose(qmt.posScalar(quat_data), qmt.posScalar(quat), atol=1e-8)


def test_quatFromToGyrStrapdown_roundtrip(example_data):
    gyr = example_data['lower_leg_right.gyr']
    rate = example_data['rate']

    quat = qmt.quatFromGyrStrapdown(gyr, rate)
    out = qmt.quatToGyrStrapdown(quat, rate)

    np.testing.assert_allclose(gyr[1:], out[1:], atol=1e-8)


def _alternativeQuatInterp(quat, ind, extend=True):
    """
    Alternative qmt.quatInterp implementation based on the calulcation of relative quaternion calculation and angle-
    axis decomposition.
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
    q_1_0 = qmt.qrel(q0, q1)

    # normalize the quaternion for positive w component to ensure
    # that the angle will be [0, 180°]
    invert_sign_ind = q_1_0[:, 0] < 0
    q_1_0[invert_sign_ind] = -q_1_0[invert_sign_ind]

    angle = 2 * np.arccos(np.clip(q_1_0[:, 0], -1, 1))
    axis = q_1_0[:, 1:]

    # copy over (almost) direct hits
    with np.errstate(invalid='ignore'):
        direct_ind = angle < 1e-06
    out = np.empty((M, 4))
    out[direct_ind] = q0[direct_ind]

    interp_ind = ~direct_ind
    t01 = ind - ind0
    q_t_0 = qmt.quatFromAngleAxis((t01*angle)[interp_ind], axis[interp_ind])
    out[interp_ind] = qmt.qmult(q0[interp_ind], q_t_0)

    if not extend:
        out[ind < 0] = np.nan
        out[ind > N - 1] = np.nan

    if isScalar:
        out = out.reshape((4,))

    return out


@pytest.mark.parametrize('q0', quat_data)
@pytest.mark.parametrize('q1', quat_data)
def test_slerp(q0, q1):
    t = 0.3
    out = qmt.slerp(q0, q1, t)

    ref = _alternativeQuatInterp([q0, q1], t)

    np.testing.assert_allclose(qmt.posScalar(out), qmt.posScalar(ref), atol=1e-10)


@pytest.mark.parametrize('extend', [True, False])
def test_quatInterp(extend):
    t = np.arange(-2, len(quat_data)+1, 0.123)
    out = qmt.quatInterp(quat_data, t, extend=extend)
    ref = _alternativeQuatInterp(quat_data, t, extend=extend)
    np.testing.assert_allclose(qmt.posScalar(out), qmt.posScalar(ref), atol=1e-10)
