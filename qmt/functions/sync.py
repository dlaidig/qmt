# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import numpy as np
import scipy.signal
import scipy.optimize

from qmt.utils.struct import Struct
from qmt.functions.quaternion import quatInterp, quatToGyrStrapdown
from qmt.functions.utils import timeVec, vecInterp, rms
from qmt.utils.misc import setDefaults
from qmt.utils.plot import AutoFigure, extendYlim


class SyncMapper:
    """
    Converts timestamps and indices for different data streams based on specified time/index mappings.
    """
    def __init__(self, syncInfo):
        """
        syncInfo is a dict data structure specifying the available recordings and the sync mappings.
        example using YAML syntax and only using the fields used by this class
        more fields may be added for other applications):

        .. code-block:: yaml

            data:
                imu_data:
                    rate: 200
                video1_part1:
                    rate: 25
                video1_part2:
                    rate: 25
                video2:
                    rate: 50
            sync:
                - [video1_part1, t, 0, video2, t, 1.85]
                - [video1_part2, t, 0, video2, t, 1145.2]
                - [video2, t, 0, imu_data, t, 0]

        The mapping should be an undirected connected graph without any cycles. An exception is thrown if the graph is
        not connected, i.e. if one or more mappings are missing. Cycles (e.g. adding
        [video1_part1, t, 0, imu_data, t, 0] to the example) are not detected and may lead to inconsistent data.

        Currently, only one sync mapping can be provided for each data set and the rate has to be known. In the future,
        support for estimating the rate based on multiple measurements may be added.
        """

        if isinstance(syncInfo, Struct):
            syncInfo = syncInfo.data
        self.syncInfo = syncInfo
        self._nameToIdx = {n: i for i, n in enumerate(syncInfo['data'].keys())}
        self._mapA = None  # matrices with index mappings; ind2 = a12*ind1 + b12
        self._mapB = None

    def _process(self):
        N = len(self._nameToIdx)
        self._mapA = np.full((N, N), np.nan)
        self._mapB = np.full((N, N), np.nan)

        for sync in self.syncInfo['sync']:
            # fill A and B matrix with the given sync pairs
            name1, mode1, val1, name2, mode2, val2 = sync
            rate1 = self.syncInfo['data'][name1]['rate']
            rate2 = self.syncInfo['data'][name2]['rate']

            i = self._nameToIdx[name1]
            j = self._nameToIdx[name2]

            if mode1 == 'ind':
                ind1 = val1
            else:
                assert mode1 == 't'
                ind1 = val1*rate1

            if mode2 == 'ind':
                ind2 = val2
            else:
                assert mode2 == 't'
                ind2 = val2*rate2

            a_ij = float(rate2)/float(rate1)
            b_ij = float(ind2) - a_ij*float(ind1)

            self._mapA[i, j] = a_ij
            self._mapB[i, j] = b_ij
            self._mapA[j, i] = 1/a_ij  # inverse mapping
            self._mapB[j, i] = -b_ij/a_ij

        # try to fill remaining mappings
        while True:
            foundNew = False
            for i, k in np.ndindex(self._mapA.shape):
                if i == k or not np.isnan(self._mapA[i, k]):
                    continue
                for j in range(N):
                    if j == i or j == k:
                        continue
                    if np.isnan(self._mapA[i, j]) or np.isnan(self._mapA[j, k]):
                        continue

                    # there is a way from i to j and a way from j to k, add the concatenation to get from i to k
                    self._mapA[i, k] = self._mapA[i, j] * self._mapA[j, k]
                    self._mapB[i, k] = self._mapA[j, k] * self._mapB[i, j] + self._mapB[j, k]
                    foundNew = True
                    break

            if not foundNew:
                break

        np.fill_diagonal(self._mapA, 1)
        np.fill_diagonal(self._mapB, 0)

        if np.isnan(self._mapA).any():
            print(self._mapA)
            print(self._mapB)
            print(self._nameToIdx)
            raise RuntimeError('sync mapping is incomplete')
        return True

    def map(self, nameFrom, modeFrom, valFrom, nameTo, modeTo):
        """
        Maps a timestamp/index from one data stream to a timestamp/index of another data stream.

        Example call to get the index in imu_data corresponding to 25 s of video1_part2:
        sync.map('video1_part2', 't', 25.0, 'imu_data', 'ind')
        """
        assert modeFrom in ('t', 'ind')
        assert modeTo in ('t', 'ind')
        assert nameFrom in self._nameToIdx, f'{nameFrom!r} not in {list(self._nameToIdx.keys())}'
        assert nameTo in self._nameToIdx, f'{nameTo!r} not in {list(self._nameToIdx.keys())}'
        if self._mapA is None:
            self._process()

        if nameFrom == nameTo:
            rate = self.syncInfo['data'][nameFrom]['rate']
            if modeFrom == 'ind' and modeTo == 't':
                return valFrom/rate
            elif modeFrom == 't' and modeTo == 'ind':
                return valFrom*rate
            else:
                assert modeFrom == modeTo
                return valFrom

        if modeFrom == 'ind':
            indFrom = valFrom
        else:
            assert modeFrom == 't'
            indFrom = valFrom * self.syncInfo['data'][nameFrom]['rate']

        i = self._nameToIdx[nameFrom]
        j = self._nameToIdx[nameTo]

        indTo = self._mapA[i, j] * indFrom + self._mapB[i, j]

        if modeTo == 'ind':
            return indTo
        else:
            assert modeTo == 't'
            return indTo / self.syncInfo['data'][nameTo]['rate']

    def timeToInd(self, name, t):
        return self.map(name, 't', t, name, 'ind')

    def indToTime(self, name, ind):
        return self.map(name, 'ind', ind, name, 't')

    def resample(self, nameFrom, signal, nameTo, N, method='auto'):
        """
        Resamples data (``signal`` sampled in ``nameFrom``) to match data sampled in a different data stream
        (``nameTo``, consisting of ``N`` samples). By default, slerp is used when the input is in shape (M, 4)
        and linear interpolation is used otherwise.

        :param nameFrom: name of the original data stream
        :param signal: data, shape (M, P)
        :param nameTo: name of the target data stream
        :param N: number of samples of the target data
        :param method: 'vecInterp'/'quatInterp'/'auto'
        :return: (N, P) resampled signal
        """
        assert signal.ndim == 2
        if method == 'auto':
            if signal.shape[1] == 4:
                method = 'quatInterp'
            else:
                method = 'vecInterp'
        assert method in ('vecInterp', 'quatInterp')

        ind = self.map(nameTo, 'ind', np.arange(N), nameFrom, 'ind')

        if method == 'vecInterp':
            return vecInterp(signal, ind)
        elif method == 'quatInterp':
            return quatInterp(signal, ind)
        else:
            raise RuntimeError(f'invalid method "{method}"')


def _rmsecorrelate(longer, shorter):
    outN = longer.shape[0] - shorter.shape[0] + 1
    overlapN = shorter.shape[0]
    out = np.empty(outN)
    for shift in range(outN):
        out[shift] = -rms(shorter - longer[shift:shift+overlapN])
    return out


def _syncOptImuOffsetViaGyr(imuGyr, imuRate, optQuat, optRate, syncRate=1000.0, cut=0.15, fc=10, correlate='rmse',
                            debug=False):
    """
    Syncs IMU and optical data by cross correlation of rotation rates.

    The shorter signal is cut even more (by default to 70%) to increase the search range if the IMU recording does not
    strictly lie within the optical recording or vice versa.

    :param imuGyr: IMU gyroscope data in rad/s
    :param imuRate: IMU sampling rate
    :param optQuat: quaternion derived from optical data
    :param optRate: sampling rate of optical data
    :param syncRate: sampling rate used for syncing
    :param cut: amount of data to cut away at start/end of the shorter signal (default: 15 %, i.e. 70 % of the shorter
        signal are used)
    :param fc: cutoff frequency for gyr low pass filters (<= 0 to disable)
    :param correlate: correlation method ('rmse' or 'linear')
    :param debug: returns additional debugging information
    :return: time offset (i.e. when the optical recording started in IMU time)
    """
    assert imuGyr.ndim == 2 and imuGyr.shape[1] == 3
    assert optQuat.ndim == 2 and optQuat.shape[1] == 4

    imuGyrInterp = vecInterp(imuGyr, np.arange(0, imuGyr.shape[0], imuRate / syncRate))
    optQuatInterp = quatInterp(optQuat, np.arange(0, optQuat.shape[0], optRate / syncRate))
    optGyrInterp = quatToGyrStrapdown(optQuatInterp, syncRate)

    # low pass filter the rotation rates with a cutoff frequency of 10 Hz to increase robustness
    if fc > 0:
        [b, a] = scipy.signal.butter(2, fc / (syncRate / 2))
        validInd = ~np.any(np.isnan(optGyrInterp), axis=1)
        optGyrInterp[validInd] = scipy.signal.filtfilt(b, a, optGyrInterp[validInd], axis=0)
        validInd = ~np.any(np.isnan(imuGyrInterp), axis=1)
        imuGyrInterp[validInd] = scipy.signal.filtfilt(b, a, imuGyrInterp[validInd], axis=0)

    optGyrNorm = np.linalg.norm(optGyrInterp, axis=1)
    imuGyrNorm = np.linalg.norm(imuGyrInterp, axis=1)

    if correlate == 'linear':
        corrfn = np.correlate
        optGyrNorm[np.isnan(optGyrNorm)] = 0
        imuGyrNorm[np.isnan(imuGyrNorm)] = 0
    elif correlate == 'rmse':
        corrfn = _rmsecorrelate
    else:
        raise ValueError('invalid correlation method: '+correlate)
    if imuGyrInterp.shape[0] < optGyrInterp.shape[0]:
        cutN = int(cut*imuGyrInterp.shape[0])
        xcorr = corrfn(optGyrNorm, imuGyrNorm[cutN:-cutN])
        maxind = np.argmax(xcorr)
        offset = -maxind + cutN
    else:
        cutN = int(cut*optGyrInterp.shape[0])
        xcorr = corrfn(imuGyrNorm, optGyrNorm[cutN:-cutN])
        maxind = np.argmax(xcorr)
        offset = maxind - cutN

    debugData = None
    if debug:
        debugData = dict(
            offset=offset,
            syncRate=syncRate,
            xcorr=xcorr,
            maxind=maxind,
            optGyrNorm=optGyrNorm,
            imuGyrNorm=imuGyrNorm,
        )

    return offset / syncRate, debugData


def _optimizeClockDriftUsingGyr(optGyr, imuGyr, imuInd, fast=False, x=None):
    optGyrNorm = np.linalg.norm(optGyr, axis=1)

    def costFn(x):
        imuGyrInterp = vecInterp(imuGyr, imuInd + np.linspace(x[0], x[1], imuInd.shape[0]))
        imuGyrInterp = imuGyrInterp - x[2:]
        imuGyrNorm = np.linalg.norm(imuGyrInterp, axis=1)

        return rms(imuGyrNorm - optGyrNorm)

    if x is not None:
        return costFn(x), x

    optResult = None
    log = []
    startVals = [0] if fast else [-100, -10, -1, 0, 1, 10, 100]
    for start1 in startVals:
        for start2 in startVals:
            init = np.array([start1, start2, 0, 0, 0], float)
            res = scipy.optimize.minimize(costFn, init, method='BFGS')
            log.append([start1, start2, res.x[0], res.x[1], res.x[2:], res.fun])
            # print('start: [{start1}, {start2}], x: {    x}, cost: {fun}'.format(start1=start1, start2=start2, x=res.x,
            #                                                                     fun=res.fun))
            if optResult is None or optResult.fun > res.fun:
                optResult = res

    shiftStart, shiftEnd, *gyrBias = optResult.x
    return shiftStart, shiftEnd, gyrBias, log


def syncOptImu(opt_quat, opt_rate, imu_gyr, imu_rate, params, debug=False, plot=False):
    """
    Determines time offsets between data recorded using opticial motion capture and IMU data.
    The offsets are determined based on angular rates. For the optical data, the angular rates are derived from a
    quaternion. Two synchronization steps are performed:

    - First, a fixed offset between the data is determined via cross correlation (imu_offsetonly).
    - Based on the result of this, both a time offset and an adjusted IMU sampling rate is adjusted to account for
      clock drift. This is done by minimizing the RMSE of the gyr norms.

    See Appendix B of https://doi.org/10.3390/data6070072 for more information about this algorithm.

    :param opt_quat: orientation quaternion from optical mocap system, Nx4
    :param opt_rate: sampling rate of opt_quat in Hz
    :param imu_gyr: gyroscope measurements from IMU in rad/s, Nx3
    :param imu_rate: sampling rate of imu_gyr in Hz
    :param params: optional parameters, defaults: syncRate=1000.0, cut=0.15, fc=10.0, correlate='rmse', fast=False
    :param debug: enables debug return value
    :param plot: enables debug plots
    :return: syncInfo dictionary that can be used by :class:`qmt.SyncMapper`
    """

    defaults = dict(syncRate=1000.0, cut=0.15, fc=10.0, correlate='rmse', fast=False)
    params = setDefaults(params, defaults)

    fc = params['fc']
    correlate = params['correlate']
    offset, shiftDebug = _syncOptImuOffsetViaGyr(imu_gyr, imu_rate, opt_quat, opt_rate, syncRate=params['syncRate'],
                                                 cut=params['cut'], fc=fc, correlate=correlate, debug=debug or plot)

    syncInfo = {
        'data': {
            'opt': {
                'rate': opt_rate,
            },
            'imu_offsetonly': {
                'rate': imu_rate
            },
        },
        'sync': [
            ['opt', 't', 0, 'imu_offsetonly', 't', offset]
        ],
    }

    sync = SyncMapper(syncInfo)

    opt_gyr = quatToGyrStrapdown(opt_quat, opt_rate)
    imuInd = sync.map('opt', 'ind', np.arange(opt_quat.shape[0]), 'imu_offsetonly', 'ind')

    # low pass filter the rotation rates with a cutoff frequency of 10 Hz to increase robustness
    if fc > 0:
        [b, a] = scipy.signal.butter(2, fc / (opt_rate / 2))
        validInd = ~np.any(np.isnan(opt_gyr), axis=1)
        opt_gyr[validInd] = scipy.signal.filtfilt(b, a, opt_gyr[validInd], axis=0)

        imu_gyr = imu_gyr.copy()
        [b, a] = scipy.signal.butter(2, fc / (imu_rate / 2))
        validInd = ~np.any(np.isnan(imu_gyr), axis=1)
        imu_gyr[validInd] = scipy.signal.filtfilt(b, a, imu_gyr[validInd], axis=0)

    shiftStart, shiftEnd, _, log = _optimizeClockDriftUsingGyr(opt_gyr, imu_gyr, imuInd, fast=params['fast'])
    newRate = (imuInd[-1] + shiftEnd - (imuInd[0] + shiftStart)) / (opt_quat.shape[0] - 1) * opt_rate

    syncInfo['data']['imu'] = {'rate': newRate.item()}
    syncInfo['sync'].append(['opt', 'ind', 0, 'imu', 'ind', imuInd[0] + shiftStart])

    # sync = SyncMapper(syncInfo)
    # imuInd2 = sync.map('opt', 'ind', np.arange(opt_quat.shape[0]), 'imu', 'ind')
    # print(shiftStart, imuInd2[0] - imuInd[0], shiftStart - (imuInd2[0] - imuInd[0]))
    # print(shiftEnd, imuInd2[-1] - imuInd[-1], shiftEnd - (imuInd2[-1] - imuInd[-1]))

    if debug or plot:
        debugData = dict(
            syncInfo=syncInfo,
            shift=shiftDebug,
            opt_gyr=opt_gyr,
            imu_gyr=imu_gyr,
            opt_rate=opt_rate,
            imu_rate=imu_rate,
            log=log,
        )
        if plot:
            syncOptImu_debugPlot(debugData, plot)
        if debug:
            return syncInfo, debugData

    return syncInfo


def syncOptImu_debugPlot(debug, figs=None):
    with AutoFigure(figs[0] if isinstance(figs, (list, tuple)) else None) as fig:
        fig.suptitle(AutoFigure.title('syncOptImu (1)'))
        from matplotlib import pyplot as plt
        offset = debug['shift']['offset']
        xcorr = debug['shift']['xcorr']
        maxind = debug['shift']['maxind']
        syncRate = debug['shift']['syncRate']
        optGyrNorm = debug['shift']['optGyrNorm']
        imuGyrNorm = debug['shift']['imuGyrNorm']
        plt.subplot(221)
        plt.plot(xcorr, '-')
        plt.stem([maxind], [xcorr[maxind]], 'r', bottom=np.min(xcorr), use_line_collection=True)
        plt.grid()
        plt.title('cross correlation')
        plt.subplot(222)
        N = int(syncRate/10)
        plt.plot(xcorr[maxind-N:maxind+N], '*-')
        plt.stem([100], [xcorr[maxind]], 'r', bottom=np.min(xcorr[maxind-N:maxind+N]), use_line_collection=True)
        plt.grid()
        plt.title('cross correlation peak, 0.2 seconds zoom')
        plt.subplot(223)
        plt.plot(np.arange(optGyrNorm.shape[0]), optGyrNorm, label='opt')
        plt.plot(np.arange(imuGyrNorm.shape[0]) - offset, imuGyrNorm, label='imu')
        plt.legend()
        plt.grid()
        plt.title('filtered gyr norm')
        plt.subplot(224)
        plt.plot(np.arange(optGyrNorm.shape[0]), optGyrNorm, label='opt')
        plt.plot(np.arange(imuGyrNorm.shape[0]) - offset, imuGyrNorm, label='imu')
        N = int(2*syncRate)
        plt.xlim(optGyrNorm.shape[0]/2-N, optGyrNorm.shape[0]/2+N)
        plt.legend()
        plt.grid()
        plt.title('filtered gyr norm, 4 second zoom')
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    with AutoFigure(figs[1] if isinstance(figs, (list, tuple)) else None) as fig:
        plt.suptitle(AutoFigure.title('syncOptImu (2)'))
        opt_gyr = debug['opt_gyr']
        imu_gyr = debug['imu_gyr']
        opt_rate = debug['opt_rate']
        log = debug['log']
        sync = SyncMapper(debug['syncInfo'])
        optGyrNorm = np.linalg.norm(opt_gyr, axis=1)
        imuIndNooffset = sync.map('opt', 'ind', np.arange(opt_gyr.shape[0]), 'imu_offsetonly', 'ind')
        gyr_nooffset = vecInterp(imu_gyr, imuIndNooffset)
        gyrNorm_nooffset = np.linalg.norm(gyr_nooffset, axis=1)
        imuIndNodrift = sync.map('opt', 'ind', np.arange(opt_gyr.shape[0]), 'imu', 'ind')
        gyr_nodrift = vecInterp(imu_gyr, imuIndNodrift)
        gyrNorm_nodrift = np.linalg.norm(gyr_nodrift, axis=1)

        t = timeVec(N=optGyrNorm.shape[0], rate=opt_rate)
        plt.subplot(221)
        plt.plot([entry[2] for entry in log], '.', label='shiftStart')
        plt.plot([entry[3] for entry in log], '.', label='shiftEnd')
        plt.legend()
        plt.grid()
        plt.title('shifts (in samples)')
        plt.subplot(222)
        cost = [entry[-1] for entry in log]
        plt.plot(cost, 'C3x', label='cost')
        extendYlim(plt.gca(), 0, 2 * np.min(cost))
        plt.legend(loc='upper left')
        plt.grid()
        plt.gca().twinx()
        plt.plot([entry[4] for entry in log], '.', label='bias')
        plt.legend(loc='upper right')
        plt.title('cost function value and gyr bias estimates')
        plt.subplot(223)
        plt.plot(t, optGyrNorm, label='opt')
        plt.plot(t, gyrNorm_nooffset, label='no offset')
        plt.plot(t, gyrNorm_nodrift, label='no drift')
        plt.legend()
        plt.grid()
        plt.title('filtered gyr norm')
        plt.subplot(224)
        plt.plot(t, optGyrNorm, label='opt')
        plt.plot(t, gyrNorm_nooffset, label='no offset')
        plt.plot(t, gyrNorm_nodrift, label='no drift')
        plt.xlim(t[-1] / 2 - 2, t[-1] / 2 + 2)
        plt.legend()
        plt.grid()
        plt.title('filtered gyr norm, 4 second zoom')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
