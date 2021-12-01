# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT


import qmt
import pytest


@pytest.mark.matlab
def test_transplant():
    d = dict(foo=42)
    assert getattr(qmt.matlab.instance, 'class')(d) == 'containers.Map'
    assert getattr(qmt.matlab.instance, 'class')(qmt.Struct(d)) == 'struct'
