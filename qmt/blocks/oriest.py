# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
import numpy as np

import qmt


def _calcAccMagDisAngle(quat, acc, mag):
    accE = qmt.normalized(qmt.rotate(quat, acc))
    magE = qmt.rotate(quat, mag)
    return np.array([np.arccos(np.clip(accE[2], -1, 1)), np.abs(np.arctan2(magE[0], magE[1]))], float)


class MadgwickAHRSBlock(qmt.Block):
    """
    Madgwicks's orientation estimation algorithm.

    See https://doi.org/10.1109/ICORR.2011.5975346 for more information about this algorithm. Based on the C++
    implementation by Sebastian Madgwick, available at https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/.

    This algorithm is also available as a function: :meth:`qmt.madgwickAHRS`.
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


class MahonyAHRSBlock(qmt.Block):
    """
    Mahony's orientation estimation algorithm.

    See https://dx.doi.org/10.1109/TAC.2008.923738 for more information about this algorithm. Based on the C++
    implementation by Sebastian Madgwick, available at https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/.

    This algorithm is also available as a function: :meth:`qmt.mahonyAHRS`.
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
        debug = dict(bias=bias, disAngle=error)
        return quat, debug
