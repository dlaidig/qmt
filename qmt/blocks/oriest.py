# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
import math

import numpy as np

import qmt


def _calcAccMagDisAngle(quat, acc, mag):
    try:  # using math.cos and the VQF functions is a lot faster
        from vqf import VQF
        accE = VQF.quatRotate(quat, acc)
        magE = VQF.quatRotate(quat, mag)
        VQF.normalize(accE)
        VQF.clip(accE, -1, 1)
        return np.array([math.acos(accE[2]), abs(math.atan2(magE[0], magE[1]))], float)
    except ImportError:
        accE = qmt.normalized(qmt.rotate(quat, acc))
        magE = qmt.rotate(quat, mag)
        return np.array([np.arccos(np.clip(accE[2], -1, 1)), np.abs(np.arctan2(magE[0], magE[1]))], float)


class OriEstVQFBlock(qmt.Block):
    """
    VQF orientation estimation algorithm.

    See https://arxiv.org/abs/2203.17024 and https://github.com/dlaidig/vqf for more information about this algorithm.

    This algorithm is also available as a function: :meth:`qmt.oriEstVQF`.
    """
    def __init__(self, Ts):
        """
        :param Ts: sampling time in seconds
        """
        super().__init__()
        self.Ts = Ts
        from vqf import VQF
        self.obj = VQF(Ts)
        self.params = self.obj.params

    def command(self, command):
        assert command == ['reset']
        self.obj.resetState()

    def setParams(self, params):
        super().setParams(params)
        for name, value in params.items():
            if name == 'tauAcc':
                self.obj.setTauAcc(value)
            elif name == 'tauMag':
                self.obj.setTauMag(value)
            elif name == 'motionBiasEstEnabled':
                self.obj.setMotionBiasEstEnabled(value)
            elif name == 'restBiasEstEnabled':
                self.obj.setRestBiasEstEnabled(value)
            elif name == 'magDistRejectionEnabled':
                self.obj.setMagDistRejectionEnabled(value)
            elif name in ('restThGyr', 'restThAcc', 'restThMag'):
                self.obj.setRestDetectionThresholds(self.params['restThGyr'], self.params['restThAcc'],
                                                    self.params['restThMag'])
            else:
                print(f'ignored param {name}')

    def step(self, gyr, acc, mag=None, debug=False):
        self.obj.update(gyr, acc, mag)
        quat = self.obj.getQuat6D() if mag is None else self.obj.getQuat9D()
        if not debug:
            return quat

        debug = {}
        state = self.obj.state
        debug['bias'] = state['bias']
        # note that state['lastAccDisAngle'] is different since it is based on the filtered acceleration
        debug['disAngle'] = _calcAccMagDisAngle(quat, acc, mag)
        debug['state'] = state
        debug['params'] = self.obj.params
        return quat, debug


class OriEstMadgwickBlock(qmt.Block):
    """
    Madgwicks's orientation estimation algorithm.

    See https://doi.org/10.1109/ICORR.2011.5975346 for more information about this algorithm. Based on the C++
    implementation by Sebastian Madgwick, available at https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/.

    This algorithm is also available as a function: :meth:`qmt.oriEstMadgwick`.
    """
    def __init__(self, Ts):
        """
        :param Ts: sampling time in seconds
        """
        super().__init__()
        self.Ts = Ts
        self.params['beta'] = 0.1
        from qmt.cpp.madgwick import MadgwickAHRS
        self.obj = MadgwickAHRS(self.params['beta'], 1/Ts)
        self.qE = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], float)  # adjusts reference frame to ENU

    def command(self, command):
        """
        Supported commands:

        - **['reset']**: resets the orientation estimation state
        """
        assert command == ['reset']
        self.obj.setState(np.array([1, 0, 0, 0], float))

    def setParams(self, params):
        """
        Supported parameters:

        - **beta**: algorithm gain
        """
        super().setParams(params)
        self.obj.setBeta(self.params['beta'])

    def step(self, gyr, acc, mag, debug=False):
        """
        :param gyr: gyroscope measurement [rad/s]
        :param acc: accelerometer measurement [m/s^2]
        :param mag:  magnetometer measurement or None [any unit]
        :param debug: enables debug output
        :return:
            - **quat**: orientation quaternion
            - **debug**: dict with debug values (only if debug==True)
        """
        quat = self.obj.update(gyr, acc, mag)
        quat = qmt.qmult(self.qE, quat)
        if not debug:
            return quat
        debug = dict(bias=np.zeros(3), disAngle=_calcAccMagDisAngle(quat, acc, mag))
        return quat, debug


class OriEstMahonyBlock(qmt.Block):
    """
    Mahony's orientation estimation algorithm.

    See https://dx.doi.org/10.1109/TAC.2008.923738 for more information about this algorithm. Based on the C++
    implementation by Sebastian Madgwick, available at https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/.

    This algorithm is also available as a function: :meth:`qmt.oriEstMahony`.
    """
    def __init__(self, Ts):
        """
        :param Ts: sampling time in seconds
        """
        super().__init__()
        self.Ts = Ts
        self.params = {'Kp': 0.5, 'Ki': 0.0}
        from qmt.cpp.madgwick import MahonyAHRS
        self.obj = MahonyAHRS(self.params['Kp'], self.params['Ki'], 1/Ts)
        self.qE = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], float)  # adjusts reference frame to ENU

    def command(self, command):
        """
        Supported commands:

        - **['reset']**: resets the orientation estimation state
        """
        assert command == ['reset']
        self.obj.setState(np.array([1, 0, 0, 0], float), np.zeros(3))

    def setParams(self, params):
        """
        Supported parameters:

        - **Kp**: proportional gain
        - **Ki**: integral gain (for gyroscope bias estimation)
        """
        super().setParams(params)
        self.obj.setParams(self.params['Kp'], self.params['Ki'])

    def step(self, gyr, acc, mag, debug=False):
        """
        :param gyr: gyroscope measurement [rad/s]
        :param acc: accelerometer measurement [m/s^2]
        :param mag:  magnetometer measurement or None [any unit]
        :param debug: enables debug output
        :return:
            - **quat**: orientation quaternion
            - **debug**: dict with debug values (only if debug==True)
        """
        quat, bias = self.obj.update(gyr, acc, mag)
        quat = qmt.qmult(self.qE, quat)
        if not debug:
            return quat
        debug = dict(bias=bias, disAngle=_calcAccMagDisAngle(quat, acc, mag))
        return quat, debug


class OriEstIMUBlock(qmt.Block):
    """
    OriEstIMU orientation estimation algorithm.

    See https://dx.doi.org/10.1016/j.ifacol.2017.08.1534 for more information about this algorithm.

    This algorithm is also available as a function: :meth:`qmt.oriEstIMU`.
    """
    def __init__(self, Ts):
        """
        :param Ts: sampling time in seconds
        """
        super().__init__()
        self.Ts = Ts
        self.params = {'tauAcc': 2.0, 'tauMag': 2.0, 'zeta': 0.0, 'accRating': 1.0}
        from qmt.cpp.oriestimu import OriEstIMU
        self.obj = OriEstIMU(1/Ts, **self.params)

    def command(self, command):
        """
        Supported commands:

        - **['reset']**: resets the orientation estimation state
        """
        assert command == ['reset']
        self.obj.resetEstimation(np.array([1, 0, 0, 0], float), np.zeros(3))

    def setParams(self, params):
        """
        Supported parameters:

        - **tauAcc**: time constants for acc correction (50% time) [must be >0], [in seconds]
        - **tauMag**: time constants for mag correction (50% time) [must be >0], [in seconds]
        - **zeta**: bias estimation strength [no unit]
        - **accRating**: enables raw rating of accelerometer, set to 0 to disable
        """
        super().setParams(params)
        self.obj.setFilterParameter(self.params['tauAcc'], self.params['tauMag'], self.params['zeta'])
        self.obj.setRating(self.params['accRating'])

    def step(self, gyr, acc, mag, debug=False):
        """
        :param gyr: gyroscope measurement [rad/s]
        :param acc: accelerometer measurement [m/s^2]
        :param mag:  magnetometer measurement or None [any unit]
        :param debug: enables debug output
        :return:
            - **quat**: orientation quaternion
            - **debug**: dict with debug values (only if debug==True)
        """
        quat, bias, error = self.obj.update(acc, gyr, mag)
        if not debug:
            return quat
        debug = dict(bias=-np.asarray(bias, float), disAngle=error)
        return quat, debug
