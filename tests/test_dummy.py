# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import qmt


def test_dummy():
    out = qmt.pythonDummy()
    assert out == 42
