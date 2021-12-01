# SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

import qmt

estSettingsOptions = [
    dict(
        wa=3.0,
        wg=0.5,
        useSampleSelection=False,
        tol=1e-8,
    ),
    dict(
        wa=1000.0,
        wg=1.0,
        useSampleSelection=False,
        tol=1e-8,
    ),
    dict(
        wa=3.0,
        wg=0.5,
        useSampleSelection=True,
        dataSize=500,
        tol=1e-8,
    ),
]


@pytest.mark.parametrize('estSettings', estSettingsOptions)
@pytest.mark.matlab
def test_jointAxisEstHingeOlsson_pyvsmat(example_data, estSettings):
    seg1 = 'lower_leg_left'
    seg2 = 'foot_left'

    ind = np.logical_and(example_data['t'] > 95, example_data['t'] < 115)  # first walking phase
    gyr1 = example_data[f'{seg1}.gyr'][ind]
    gyr2 = example_data[f'{seg2}.gyr'][ind]
    acc1 = example_data[f'{seg1}.acc'][ind]
    acc2 = example_data[f'{seg2}.acc'][ind]

    jhat1, jhat2, debug = qmt.jointAxisEstHingeOlsson(acc1, acc2, gyr1, gyr2, estSettings=estSettings, debug=True)
    jhat1_mat, jhat2_mat, debug_mat = qmt.matlab.jointAxisEstHingeOlsson(acc1, acc2, gyr1, gyr2, estSettings)

    if estSettings.get('useSampleSelection', False):
        np.testing.assert_allclose(debug['sampleSelectionVars']['accSamples'],
                                   debug_mat['sampleSelectionVars']['accSamples'].T-1)
        np.testing.assert_allclose(debug['sampleSelectionVars']['gyrSamples'],
                                   debug_mat['sampleSelectionVars']['gyrSamples'].T-1)

    np.testing.assert_allclose(jhat1, jhat1_mat, atol=1e-3)
    np.testing.assert_allclose(jhat2, jhat2_mat, atol=1e-3)
