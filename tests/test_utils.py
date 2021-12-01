# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import qmt
import numpy as np
import pytest


nanInterp_data = [
    [
        np.array([
            [np.nan, np.nan],
            [0, 1],
            [0.5, 1.5],
            [2, 2],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [3, 4],
            [np.nan, np.nan],
            [4, 6],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ], float),
        np.array([
            [0, 1],
            [0, 1],
            [0.5, 1.5],
            [2, 2],
            [2.25, 2.5],
            [2.5, 3.0],
            [2.75, 3.5],
            [3, 4],
            [3.5, 5],
            [4, 6],
            [4, 6],
            [4, 6],
        ], float),
    ],
]


@pytest.mark.parametrize('inputs,outputs', nanInterp_data)
def test_nanInterp(inputs, outputs):
    out = qmt.nanInterp(inputs)
    np.testing.assert_allclose(out, outputs)


@pytest.mark.parametrize('inputs,outputs', nanInterp_data)
def test_nanInterp_1D(inputs, outputs):
    for col in range(inputs.shape[1]):
        out = qmt.nanInterp(inputs[:, col])
        np.testing.assert_allclose(out, outputs[:, col])
