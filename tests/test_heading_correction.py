# SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import qmt

headingCorrectionOptions = [
    # test 1D constraint
    dict(
        joint=np.array([0.0, 0.0, 1.0]),
        jointInfo={},
        estSettings={},
    ),
    # test 2D constraint
    dict(
        joint=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        jointInfo={},
        estSettings={'ratingMin': 0.1}
    ),
    # test 3D constraint
    dict(
        joint=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        jointInfo={
                'angle_ranges': np.deg2rad(np.array([
                    [-5, 40],
                    [-90, -70],
                    [-10, 40],
                ], float)),
                'convention': 'xyz'
            },
        estSettings={
            'ratingMin': 0.6,
        },
    ),
]


@pytest.mark.parametrize('options', headingCorrectionOptions)
@pytest.mark.matlab
def test_heading_correction_pyvsmat(example_data, options):
    joint = options['joint']
    jointInfo = options['jointInfo']
    estSettings = options['estSettings']
    seg1 = 'lower_leg_left'
    seg2 = 'foot_left'
    gyr1 = example_data[f'{seg1}.gyr']
    gyr2 = example_data[f'{seg2}.gyr']
    quat1 = qmt.normalized(example_data[f'{seg1}.quat'])
    quat2 = qmt.normalized(example_data[f'{seg2}.quat'])
    t = example_data['t']

    quat2Corr, delta, deltaFilt, rating, stateOut, debugData = \
        qmt.headingCorrection(gyr1, gyr2, quat1, quat2, t, joint, jointInfo, estSettings=estSettings, debug=True)

    quat2Corr_mat, delta_mat, deltaFilt_mat, rating_mat, stateOut_mat, debugData_mat = \
        qmt.matlab.headingCorrection(gyr1, gyr2, quat1, quat2, t, joint, jointInfo, estSettings, True)

    np.testing.assert_allclose(delta.squeeze(), delta_mat.squeeze(), atol=1e-4)
    np.testing.assert_allclose(deltaFilt.squeeze(), deltaFilt_mat.squeeze(), atol=1e-4)
