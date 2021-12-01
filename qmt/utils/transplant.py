# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import os

from qmt.utils.struct import Struct

try:
    import transplant
except ImportError:
    class transplant:
        class MatlabStruct:
            pass

        class Matlab:
            def __init__(self, **kwargs):
                raise RuntimeError('Transplant is not installed.')


def _getMatlabPath():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'matlab'))


class Matlab(transplant.Matlab):
    """
    Transplant's Matlab class that encodes qmt.Struct instances as Matlab structs.
    """
    def _encode_values(self, data):
        # on exit, Struct can be None
        if Struct is not None and isinstance(data, Struct):
            return super()._encode_values(transplant.MatlabStruct(data))
        else:
            return super()._encode_values(data)


class MatlabWrapper:
    """
    Provides access to the qmt Matlab functions. A Matlab instance is automatically started in background on the
    first access. If there is no ``matlab`` executable in your path, call
    :meth:`~qmt.utils.transplant.MatlabWrapper.init` manually.

    See :ref:`tutorial_py_matlab_interface` for more details.
    """
    def __init__(self):
        self._transplant = None

    @property
    def running(self):
        return self._transplant is not None

    def init(self, **kwargs):
        """
        Starts Matlab and sets up the path for the qmt functions. If there is no ``matlab`` executable in your path,
        call this function manually and provide the full path to Matlab::

            qmt.matlab.init(executable='/usr/local/MATLAB/R2017b/bin/matlab')

        :param kwargs: Parameters passed to transplant.
        :return: None
        """
        assert self._transplant is None, 'Matlab is already started, call exit() first'
        self._transplant = Matlab(**kwargs)
        self._transplant.addpath(_getMatlabPath())

    @property
    def instance(self):
        """
        Gives access to the `Transplant <https://github.com/bastibe/transplant>`_ Matlab class that can be used to call
        other Matlab functions and execute arbitrary Matlab code.
        ::

            m = qmt.matlab.instance
            m.version()
            m.eval('1234*(0.1+0.1+0.1-0.3)^(1/10)')
        """
        if self._transplant is None:
            print('Starting Matlab...')
            self.init()
        return self._transplant

    def exit(self):
        """
        Quits Matlab.
        """
        t = self._transplant
        if t is None:
            return
        self._transplant = None
        t.exit()

    def __getattr__(self, name):
        return getattr(self.instance.qmt, name)


matlab = MatlabWrapper()
