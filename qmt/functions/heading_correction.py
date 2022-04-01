# SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
# SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
#
# SPDX-License-Identifier: MIT
import numpy as np
import scipy.signal as signal
from scipy.linalg import lstsq
from scipy import interpolate

import qmt
from qmt.utils.misc import setDefaults
from qmt.utils.plot import AutoFigure


def headingCorrection(gyr1, gyr2, quat1, quat2, t, joint, jointInfo, estSettings=None, verbose=False,
                      debug=False, plot=False):
    """
    This function corrects the heading of a kinematic chain of two segments whose orientations are estimated without
    the use of magnetometers. It corrects the heading of the second segment in a way that its orientation is estimated
    in the reference of the first segment. It uses kinematic constraints for rotational joints to estimate the heading
    offset of both segments based on the limited set of possible relative orientations induced by the rotational joint.
    There are methods for 1D, 2D and 3D joints, based on different constraints.

    Equivalent Matlab function: :mat:func:`+qmt.headingCorrection`.

    :param gyr1: Nx3 array of angular velocities in rad/s
    :param gyr2: Nx3 array of angular velocities in rad/s
    :param quat1: Nx4 array of orientation quaternions
    :param quat2: Nx4 array of orientation quaternions
    :param t: Nx1 vector of the equidistant sampled time signal
    :param joint: Dofx3 array containing the axes of the joint.
    :param jointInfo: Dictionary containing additional joint information. Only needed for 3D-joints. Keys:

                   - **convention**: String of the chosen euler angles convention, e.g. 'xyz'
                   - **angle_ranges**: 3x2 array with the minimum and maximum value for each joint angle in radians

    :param verbose: show all logs in function. Disabled by Default.
    :param estSettings: structure containing the needed settings for the estimation:

           - **window_time**: width of the window over which the estimation should be performed in seconds
           - **estimation_rate**: rate in Hz at which the estimation should be performed. typically below 1Hz is
             sufficient
           - **data_rate**: rate at which new data is fed into the estimation. Is used to down sample the data for a
             faster estimation. Typically values around 5Hz are sufficient
           - **alignment**: String of the chosen type of alignment of the estimation window. It describes how the
             samples around the current sample at which the estimation takes place are chosen. Possible values:
             'backward', 'center', 'forward'
           - **enable_stillness**: boolean value if the stillness detection should be enabled
           - **stillness_time**: time in seconds which should pass for the stillness detection to be triggered
           - **stillness_threshold**: threshold for the angular velocity under which the body is assumed to be at rest.
             In rad/s
           - **tauDelta**: time constant for the filter in seconds
           - **tauBias**: time constant for the filter in seconds
           - **delta_range**: array of values that are supposed to be tested for the method for 3D joints.
           - **constraint**: type of contraint used for the estimation.

               - Constraints for 1D: (proj, euler, euler_2d, 1d_corr).
               - Constraints for 2D: (euler, gyro, combined).
               - Constraints for 3D: (default)

           - **optimizer_steps**: Number of Gauss-Newton steps during optimization.

    :param debug: enables debug output
    :param plot: enables debug plot

    :return:
         - **quat2Corr**: Corrected orientation of the second segment in the reference frame of the first segment
         - **delta**: value of the estimated heading offset
         - **deltaFilt**: value of the filtered heading offset
         - **rating**: value of the certainty of the estimation
         - **state**: state of the estimation. 1: regular, 2: startup, 3: stillness
         - **debug**: dict with debug values (only if debug==True)
    """

    #  Input Checking
    joint = np.array(joint).copy()
    if len(joint.shape) < 2:
        assert joint.shape == (3,), 'Wrong dimension of joint matrix'
        dof = 1
    else:
        N = joint.shape[0]
        assert N <= 3, 'Wrong dimension of joint matrix'
        assert joint.shape == (N, 3), 'Wrong dimension of joint matrix'
        dof = N

    # Parameter loading
    # if no settings are given load a set of standard settings
    defaults = dict(
        useRomConstraints=False,
        windowTime=8.0,
        estimationRate=1.0,
        dataRate=5.0,
        tauDelta=5.0,
        tauBias=5.0,
        ratingMin=0.4,
        alignment='backward',
        enableStillness=True,
        optimizerSteps=5,
        stillnessTime=3.0,
        stillnessThreshold=np.deg2rad(4),
        stillnessRating=1,
        startRating=1,
        deltaRange=np.deg2rad(np.linspace(0, 359, 360)),
        angleRanges=np.ones((3, 1)) * np.array([[-np.pi, np.pi]]),
        convention='xyz',
        constraint=None,
    )

    estSettings = setDefaults(estSettings, defaults)
    if estSettings['constraint'] is None:
        del estSettings['constraint']  # default values are later handled depending on DoF

    useRomConstraints = estSettings['useRomConstraints']
    windowTime = estSettings['windowTime']
    estimationRate = estSettings['estimationRate']
    dataRate = estSettings['dataRate']
    tauDelta = estSettings['tauDelta']
    tauBias = estSettings['tauBias']
    ratingMin = estSettings['ratingMin']
    alignment = estSettings['alignment']
    enableStillness = estSettings['enableStillness']
    optimizerSteps = estSettings['optimizerSteps']
    stillnessTime = estSettings['stillnessTime']
    stillnessThreshold = estSettings['stillnessThreshold']
    stillnessRating = estSettings['stillnessRating']
    startRating = estSettings['startRating']
    deltaRange = estSettings['deltaRange']
    angleRanges = np.asarray(estSettings['angleRanges'], float)
    convention = estSettings['convention']

    if dof == 3:
        angleRanges = np.asarray(jointInfo['angle_ranges'], float)
        convention = jointInfo['convention']
        constraint = estSettings.get('constraint', 'default')
    elif dof == 2:
        constraint = estSettings.get('constraint', 'euler')
    else:
        constraint = estSettings.get('constraint', 'euler_1d')

    if verbose:
        print('parameters: ')
        for key, val in estSettings.items():
            print(f'{key}: {val}')

    # Determine data rate from one of the time signals.Assumption: constant
    timeDiff = np.diff(t)
    rate = 1 / timeDiff[1]
    assert np.allclose(timeDiff, 1 / rate)  # make sure the sampling time is constant to avoid surprises
    assert np.isclose(rate % dataRate, 0) or np.isclose(rate % dataRate, dataRate), f'rate should be divisible by ' \
                                                                                    f'given dataRate: {dataRate}'

    windowSteps = round(windowTime * rate)
    estimationSteps = round(1 / estimationRate * rate)
    dataSteps = round(1 / dataRate * rate)
    stillnessSteps = round(stillnessTime * rate)

    # number of total time steps in the time series
    N = len(t)

    if alignment == 'backward':
        starts = np.arange(windowSteps, N, estimationSteps, dtype=int)
    elif alignment == 'center':
        starts = np.arange(windowSteps / 2, N - windowSteps / 2, estimationSteps, dtype=int)
    elif alignment == 'forward':
        starts = np.arange(1, N - windowSteps, estimationSteps, dtype=int)
    else:
        raise ValueError('Wrong alignment type')

    # to have a smoother start up,
    # add estimations at each data step in the beginning until a complete time window has passed

    regularStart = starts[0]  # start of the regular estimation without smooth startup

    starts = np.concatenate((np.arange(dataSteps, starts[0], dataSteps, dtype=int), starts))
    estimations = len(starts)  # number of performed estimations

    # Initialize the result vectors
    delta = np.zeros((estimations, 1))  # delta is the heading offset between the first and second segment
    rating = np.zeros((estimations, 1))  # the rating indicates the quality of the estimation
    stateOut = np.zeros((estimations, 1))
    cost = np.zeros((estimations, 1))
    # Initialize the filtering time constants
    tauDelta = np.ones((estimations, 1)) * tauDelta
    tauBias = np.ones((estimations, 1)) * tauBias
    stillnessCorrector = stillnessCorrection()

    # Estimation
    for k in range(1, estimations):
        # set default values to variables
        stillnessTrigger = False
        state = 'none'

        # get the current index in the data
        index = starts[k]
        # check whether the smooth  startup is active
        # If so, the used data, for the estimation is from index 1 to the current index
        if index < regularStart:
            state = 'startup'
            indexStart = 0
            indexEnd = index
        else:
            # during regular estimation, the start and end index are determined based on
            # the current index and the chosen aligment type
            if alignment == 'center':
                indexStart = index - int(windowSteps / 2)
                indexEnd = index + int(windowSteps / 2)
            elif alignment == 'forward':
                indexStart = index
                indexEnd = index + windowSteps
            else:  # alignment == 'backward'
                indexStart = index - windowSteps
                indexEnd = index

        #  Stillness detection
        #      Check if the two segments are at rest. If they are at rest, the
        #      current value of delta can be calculated from the last estimated
        #      relative orientation. This ensures that both segments do not move in
        #      a resting phase. However, this does not ensure correct estimation
        #      since it is only based on the last estimated orientation
        #
        #      only do the stillness detection of the setting is set to true and do
        #      not do stillness detection during startup

        if enableStillness and state != 'startup':
            # check whether the number of passed samples is larger than the detection peroiod for the stillness
            # detection
            if index > stillnessSteps:
                stillness = checkStillness(gyr1[index - stillnessSteps:index + 1, :],
                                           gyr2[index - stillnessSteps:index + 1, :], stillnessThreshold)
                if stillness:
                    if stateOut[k-1] == 1:  # regular
                        stillnessTrigger = True
                    state = 'stillness'
                else:
                    state = 'regular'
            else:
                state = 'regular'
        else:
            if state != 'startup':
                state = 'regular'

        # Stillness correction
        if state == 'stillness':
            delta[k, :] = stillnessCorrector.stillnessCorrection(quat1[index, :], quat2[index, :], delta[k - 1, :],
                                                                 stillnessTrigger)

        # Estimation
        # the estimation will only be performed in startup and regular state,
        # not in stillness state
        if state == 'startup' or state == 'regular':
            # % for convenience extract the necessary data windows from both
            # % segments
            q1 = quat1[indexStart:indexEnd + 1:dataSteps]
            q2 = quat2[indexStart:indexEnd + 1:dataSteps]
            g1 = gyr1[indexStart:indexEnd + 1:dataSteps]
            g2 = gyr2[indexStart:indexEnd + 1:dataSteps]
            time = t[indexStart:indexEnd + 1:dataSteps]

            # execute a dedicated estimation algorithm for each type of joint.
            # for more detailed explanation look in the descriptions of the
            # corresponding method
            if dof == 1:
                delta[k, :], rating[k, :], cost[k, :] = estimateDelta1d(q1, q2, joint, delta[k - 1, :], constraint,
                                                                        optimizerSteps)
            elif dof == 2:
                delta[k, :], rating[k, :], cost[k, :] = estimateDelta2d(q1, q2, g1, g2, time, joint, delta[k - 1, :],
                                                                        constraint, optimizerSteps)
            else:
                delta[k, :], rating[k, :], cost[k, :] = estimateDelta3d(q1, q2, angleRanges, convention, deltaRange,
                                                                        delta[k - 1, :])

            # during startup estimation a second estimation is performed from a
            # different starting value to ensure that the global minimum is found

            if state == 'startup':
                if dof == 1:
                    delta2, _, cost2 = estimateDelta1d(q1, q2, joint, delta[k - 1, :] + np.pi, constraint,
                                                       optimizerSteps)
                elif dof == 2:
                    delta2, _, cost2 = estimateDelta2d(q1, q2, g1, g2, time, joint, delta[k - 1, :] + np.pi,
                                                       constraint, optimizerSteps)
                else:
                    delta2, _, cost2 = estimateDelta3d(q1, q2, angleRanges, convention, deltaRange,
                                                       delta[k - 1, :] + np.pi)
                if useRomConstraints:
                    costRom = romCost(angleRanges, convention, q1, q2, delta[k, :])
                    costRom2 = romCost(angleRanges, convention, q1, q2, delta2)
                    if costRom2 < costRom:
                        delta[k] = delta2
                elif cost2 < cost[k, :]:
                    delta[k] = delta2

        # adapt the rating in the two special states startup and stillness and set it to 1 in order
        # to enable fast convergence
        if state == 'startup':
            rating[k, :] = startRating
            tauDelta[k, :] = 0.4
            tauBias[k, :] = 1
        if state == 'stillness':
            rating[k, :] = stillnessRating
        if state == 'regular':
            stateOut[k, :] = 1
        elif state == 'startup':
            stateOut[k, :] = 2
        elif state == 'stillness':
            stateOut[k, :] = 3
        else:
            raise ValueError('Wrong state')

    # Filtering
    # use a filter to smooth the trajectory of the estimated delta.
    # The filter tries to eliminate phase lag introduced by low pass filtering by estimating the slope of
    # the current trajectory. Furthermore, if rating < rating_min,
    # the filter only extrapolates the estimated slope and dismisses new estimates until rating >= rating_min

    # shape(Nx1) for delta, rating tauDelta, tauBias inputs
    deltaFilt, bias = headingFilter(delta, rating, stateOut, estimationRate, tauDelta, tauBias, ratingMin, windowTime,
                                    alignment)

    if debug or plot:
        # uninterpolated debug data
        uninterpolated = {'delta': delta, 'deltaFilt': deltaFilt, 'rating': rating, 'stateOut': stateOut}

    # Interpolation
    # Interpolate the data to the given time signal
    fDelta = interpolate.interp1d(t[starts], np.unwrap(delta.ravel()), kind='linear', fill_value='extrapolate')
    fDeltaFilt = interpolate.interp1d(t[starts], deltaFilt.ravel(), kind='linear', fill_value='extrapolate')
    fRating = interpolate.interp1d(t[starts], rating.ravel(), kind='linear', fill_value='extrapolate')
    fStateOut = interpolate.interp1d(t[starts], stateOut.ravel(), kind='linear', fill_value='extrapolate')
    delta = fDelta(t)
    deltaFilt = fDeltaFilt(t)
    rating = fRating(t)
    stateOut = fStateOut(t)

    # Heading correction
    quat2Corr = qmt.qmult(qmt.quatFromAngleAxis(deltaFilt, [0, 0, 1]), quat2)

    if debug or plot:
        debugData = qmt.Struct(
            uninterpolated=uninterpolated,
            delta=delta,
            delta_filt=deltaFilt,
            rating=rating,
            state_out=stateOut,
            fdelta=fDelta,
            fdelta_filt=fDeltaFilt,
            frating=fRating,
            fstate_out=fStateOut,
            starts=starts,
            bias=bias,
            params=dict(
                windowTime=windowTime,
                estimationRate=estimationRate,
                dataRate=dataRate,
                tauDelta=tauDelta,
                tauBias=tauBias,
                ratingMin=ratingMin,
                alignment=alignment,
                enableStillness=enableStillness,
                optimizerSteps=optimizerSteps,
                stillnessTime=stillnessTime,
                stillnessThreshold=stillnessThreshold,
                stillnessRating=stillnessRating,
                startRating=startRating,
                deltaRange=deltaRange,
                constraint=constraint,
                angleRanges=angleRanges,
                convention=convention,
                joint=joint,
            )
        )
        if plot:
            headingCorrection_debugPlot(debugData, plot)
        if debug:
            return quat2Corr, delta, deltaFilt, rating, stateOut, debugData

    return quat2Corr, delta, deltaFilt, rating, stateOut


def headingCorrection_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex=True)
        fig.suptitle(AutoFigure.title('headingCorrection'))

        ax1.plot(np.rad2deg(debug['delta'])[:, np.newaxis])
        ax1.set_ylabel('[°]')
        ax1.set_xlabel('index')
        ax1.set_title(f'delta, {debug["delta"].shape} {debug["delta"].dtype}')

        ax2.plot(np.rad2deg(debug['delta_filt'])[:, np.newaxis])
        ax2.set_ylabel('[°]')
        ax2.set_xlabel('index')
        ax2.set_title(f'delta_filtered, {debug["delta_filt"].shape} {debug["delta_filt"].dtype}')

        ax3.plot(debug['rating'][:, np.newaxis])
        ax3.set_title(f'rating, {debug["rating"].shape} {debug["rating"].dtype}')

        ax4.plot(debug['state_out'][:, np.newaxis])
        ax4.set_ylabel('[1: regular, 2: startup, 3: stillness]')
        ax4.set_title(f'state_out, {debug["state_out"].shape} {debug["state_out"].dtype}')

        for ax in (ax1, ax2, ax3, ax4):
            ax.grid()
        fig.tight_layout()


# % check whether the mean of both gyroscope signals is smaller than the
# % defined threshold over the complete period
def checkStillness(gyr1, gyr2, still_threshold):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/heading_corection.m

    if np.mean(np.linalg.norm(gyr1, axis=1)) < still_threshold and \
            np.mean(np.linalg.norm(gyr2, axis=1)) < still_threshold:
        stillness = True
    else:
        stillness = False
    return stillness


class stillnessCorrection:
    # based on Matlab implementation in matlab/lib/HeadingCorrection/heading_corection.m
    def __init__(self):
        # % initialize memory variable
        self.quatRelRef = None
        self.deltaStill = None

    def stillnessCorrection(self, quat1, quat2, lastDelta, stillnessTrigger):

        # % set the reference relative orientation which should be held during
        # % stillness phase. It is reset after each rising edge of the stillness
        # % detection.
        if self.quatRelRef is None:
            self.quatRelRef = qmt.qmult(qmt.qmult(qmt.qinv(quat1), qmt.quatFromAngleAxis(
                lastDelta, [0, 0, 1])), quat2)
            self.deltaStill = lastDelta
        else:
            if stillnessTrigger:
                self.quatRelRef = qmt.qmult(
                    qmt.qmult(qmt.qinv(quat1), qmt.quatFromAngleAxis(lastDelta, [0, 0, 1])),
                    quat2)
                self.deltaStill = lastDelta
        qE2E1 = qmt.qmult(qmt.qmult(quat1, self.quatRelRef), qmt.qinv(quat2))
        qRel = qmt.qrel(qmt.quatFromAngleAxis(self.deltaStill, [0, 0, 1]), qE2E1)
        # deltaInc = 2 * np.arctan2(np.dot(qRel[1:], np.array([0, 0, 1])), qRel[0])
        deltaInc = 2 * np.arctan2(np.dot(qRel[0, 1:], np.array([0, 0, 1])), qRel[0, 0])
        delta = self.deltaStill + deltaInc

        return delta


def headingFilter(delta, rating, state, estimationRate, tauBias, tauDelta, minRating, windowTime, alignment):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/heading_corection.m

    N = delta.shape[0]
    assert delta.shape == (N, 1)
    assert rating.shape == (N, 1)

    Ts = 1 / estimationRate
    out = np.zeros((N, 1))
    bias = np.zeros((N, 1))

    windowSize = windowTime * estimationRate

    kBias = 1 - np.exp(-Ts * np.log(2) / tauBias)
    kDelta = 1 - np.exp(-Ts * np.log(2) / tauDelta)

    if isinstance(tauDelta, (int, float)):
        kDelta = np.ones((N, 1)) * kDelta
    if isinstance(tauBias, (int, float)):
        kBias = np.ones((N, 1)) * kBias

    rating = rating.copy()
    rating[rating < minRating] = 0
    out[0, :] = delta[0, :]

    biasClip = np.deg2rad(2) * Ts  # limit bias estimate to 2°/s
    j = 0
    for i in range(1, N):
        if state[i] == 2:  # startup
            kBiasEff = 0.0
            kDeltaEff = max(rating[i, :] * kDelta[i, :], 1 / (i + 1))
        else:
            kBiasEff = rating[i, :] * kBias[i, :]
            kDeltaEff = max(rating[i, :] * kDelta[i, :], 1 / (j + 1))
            j += 1

        deltaDiff = np.clip(qmt.wrapToPi(delta[i, :] - delta[i - 1, :]), -biasClip, biasClip)
        bias[i, :] = np.clip(bias[i - 1, :] + kBiasEff * (deltaDiff - bias[i - 1, :]), -biasClip, biasClip)
        out[i, :] = out[i - 1, :] + bias[i, :] + kDeltaEff * (qmt.wrapToPi(delta[i, :] - out[i - 1, :]) - bias[i, :])
    if alignment == 'backward':
        delta_out = out + windowSize / 2 * bias
    elif alignment == 'forward':
        delta_out = out - windowSize / 2 * bias
    else:  # center
        delta_out = out

    return delta_out, bias


def estimateDelta1d(quat1, quat2, joint, deltaStart, constraint, optimizationSteps):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_1D.m
    j = joint
    delta = deltaStart
    cost = 0.0

    for _ in range(optimizationSteps):
        deltaInc, cost = gaussNewtonStep(quat1, quat2, j, j, delta, constraint)
        delta = delta + deltaInc

    delta = qmt.wrapTo2Pi(delta)

    #  Rating calculation
    v1 = qmt.rotate(quat1, j)
    v2 = qmt.rotate(quat2, j)
    weight = np.sqrt(v1[:, 0] ** 2 + v1[:, 1] ** 2) * np.sqrt(v2[:, 0] ** 2 + v2[:, 1] ** 2)
    rating = qmt.rms(weight)

    return delta, rating, cost


def gaussNewtonStep(quat1, quat2, jB1, jB2, delta, constraint):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_1D.m
    errors, jacobi = errorAndJac1D(quat1, quat2, jB1, jB2, delta, constraint)
    deltaParams = -1 * lstsq(np.atleast_2d(jacobi @ jacobi), np.atleast_2d(jacobi @ errors))[0]
    deltaParams = np.squeeze(deltaParams)
    totalError = np.linalg.norm(errors)

    return deltaParams, totalError


def errorAndJac1D(q1, q2, jB1, jB2, delta, constraint):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_1D.m
    if constraint == 'proj':
        errors, jacobi = getErrorAndJacProj(delta, q1, q2, jB1)
    elif constraint == '1d_corr':
        errors, jacobi = getErrorAndJac1dCorr(delta, q1, q2, jB1, jB2)
    elif constraint == 'euler_1d':
        errors, jacobi = getErrorAndJacEuler1D(delta, q1, q2, jB1, jB2)
    elif constraint == 'euler_2d':
        errors, jacobi = getErrorAndJacEuler2D(delta, q1, q2, jB1, jB2)
    else:
        raise ValueError('Wrong constrain')
    return errors, jacobi


def getErrorAndJacEuler1D(delta, q1, q2, j, jAlt):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_1D.m
    errors = CostEuler1D(delta, q1, q2, j, jAlt)
    eps = 1e-8
    errors_eps = CostEuler1D(delta + eps, q1, q2, j, jAlt)
    jacobi = (errors_eps - errors) / eps

    return errors, jacobi


def CostEuler1D(delta, q1, q2, jB1, jB2):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_1D.m
    qE2E1 = qmt.quatFromAngleAxis(delta, [0, 0, 1])
    qRot1 = qmt.quatFromAngleAxis(np.arccos(np.array([0, 0, 1]) @ jB1), np.cross(np.array([0, 0, 1]), jB1))
    qRot2 = qmt.quatFromAngleAxis(np.arccos(np.array([0, 0, 1]) @ jB2), np.cross(np.array([0, 0, 1]), jB2))

    qB1E1 = qmt.qmult(q1, qRot1)
    qB2E2 = qmt.qmult(q2, qRot2)
    qB2B1 = qmt.qmult(qmt.qmult(qmt.qinv(qB1E1), qE2E1), qB2E2)

    angles = qmt.eulerAngles(qB2B1, 'zxy', True)
    secondAngle = angles[:, 1]
    thirdAngle = angles[:, 2]

    error = getWeight(q1, q2, jB1, jB2) * np.sqrt(secondAngle ** 2 + thirdAngle ** 2)

    return error


def getWeight(q1, q2, jB1, jB2):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_1D.m
    v1 = qmt.rotate(q1, jB1)
    v2 = qmt.rotate(q2, jB2)
    weight = np.sqrt(v1[:, 0] ** 2 + v1[:, 1] ** 2) * np.sqrt(v2[:, 0] ** 2 + v2[:, 1] ** 2)
    return weight


def getErrorAndJacProj(delta, q1, q2, j):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_1D.m
    errors = CostProj(delta, q1, q2, j)
    eps = 1e-8
    errorsEps = CostProj(delta + eps, q1, q2, j)
    jacobi = (errorsEps - errors) / eps
    return errors, jacobi


def CostProj(delta, q1, q2, j):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_1D.m
    qE2E1 = qmt.quatFromAngleAxis(delta, [0, 0, 1])
    qhat = qmt.qmult(qmt.qinv(q1), qmt.qmult(qE2E1, q2))

    alpha = 2 * np.arctan2(qhat[:, 1:] @ j, qhat[:, 0])
    qProj = qmt.quatFromAngleAxis(alpha, j)
    qRes = qmt.qmult(qmt.qinv(qProj), qhat)

    v1 = qmt.rotate(q1, j)
    v2 = qmt.rotate(q2, j)
    # weight = sqrt(v1(:, 1).^ 2 + v1(:, 2).^ 2).*sqrt(v2(:, 1).^ 2 + v2(:, 2).^ 2);
    weight = np.sqrt(v1[:, 0] ** 2 + v1[:, 1] ** 2) * np.sqrt(v2[:, 0] ** 2 + v2[:, 1] ** 2)
    error = weight * 2. * np.arccos(qRes[:, 0])
    return error


def getErrorAndJac1dCorr(delta, q1, q2, jB1, jB2):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_1D.m
    v1 = qmt.rotate(q1, jB1)
    v2 = qmt.rotate(q2, jB2)

    weight = getWeight(q1, q2, jB1, jB2)

    error = qmt.wrapToPi(np.arctan2(v2[:, 1], v2[:, 0]) - np.arctan2(v1[:, 1], v1[:, 0]) + delta) * weight
    jac = weight
    return error, jac


def getErrorAndJacEuler2D(delta, q1, q2, j1, j2):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_1D.m
    # % Calculate the error
    qE2E1 = qmt.quatFromAngleAxis(delta, [0, 0, 1])
    qB2S2 = qmt.quatFromAngleAxis(np.arccos(np.dot([0, 1, 0], j2)), np.cross([0, 1, 0], j2))
    qB1S1 = qmt.quatFromAngleAxis(np.arccos(np.dot([0, 0, 1], j1)), np.cross([0, 0, 1], j1))

    qE1B1 = qmt.qinv(qmt.qmult(q1, qB1S1))
    qE2B1 = qmt.qmult(qE1B1, qE2E1)
    qS2B1 = qmt.qmult(qE2B1, q2)
    qB2B1 = qmt.qmult(qS2B1, qB2S2)

    arcsinArg = 2 * (qB2B1[:, 1] * qB2B1[:, 0] + qB2B1[:, 2] * qB2B1[:, 3])
    secondAngle = np.arcsin(np.clip(arcsinArg, -1, 1))

    error = secondAngle

    #  Jacobian
    qB2E2 = qmt.qmult(q2, qB2S2)
    dQ3Ba = np.zeros((1, 4))
    dQ3Ba[:, 0] = -0.5 * np.sin(delta / 2)
    dQ3Ba[:, 3] = 0.5 * np.cos(delta / 2)
    dQ3Ba = qmt.normalized(dQ3Ba)
    dQB = qmt.qmult(qmt.qmult(qE1B1, dQ3Ba), qB2E2)
    dQ = dQB

    jac = 2 * (dQ[:, 1] * qB2B1[:, 0] + qB2B1[:, 1] * dQ[:, 0] +
               dQ[:, 2] * qB2B1[:, 3] + qB2B1[:, 2] * dQ[:, 3])
    jac = jac / np.sqrt(1 - arcsinArg ** 2)

    return error, jac


#  2D functions
def errorAndJac2D(q1, q2, gyr1, gyr2, t, j1, j2, deltaStart, constraint):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/estimate_delta_2d.m
    if constraint == 'euler':
        errors, jacobi = getJacEulerCons(deltaStart[0], q1, q2, j1, j2)
    elif constraint == 'gyro':
        errors, jacobi = getJacGyroCons(j1, j2, q1, q2, gyr1, gyr2, deltaStart)
    elif constraint == 'combined':
        errors, jacobi = getJacCombCons(j1, j2, q1, q2, gyr1, gyr2, deltaStart)
    else:
        raise ValueError(f'Wrong constraint: {constraint}')
    return errors, jacobi


def getJacEulerCons(delta, q1, q2, j1, j2):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_2D.m
    # % Calculate the error
    qE2E1 = qmt.quatFromAngleAxis(delta, [0, 0, 1])
    qB2S2 = qmt.quatFromAngleAxis(np.arccos(np.dot([0, 1, 0], j2)), np.cross([0, 1, 0], j2))
    qB1S1 = qmt.quatFromAngleAxis(np.arccos(np.dot([0, 0, 1], j1)), np.cross([0, 0, 1], j1))

    qE1B1 = qmt.qinv(qmt.qmult(q1, qB1S1))
    qE2B1 = qmt.qmult(qE1B1, qE2E1)
    qS2B1 = qmt.qmult(qE2B1, q2)
    qB2B1 = qmt.qmult(qS2B1, qB2S2)

    arcsinArg = 2 * (qB2B1[:, 1] * qB2B1[:, 0] + qB2B1[:, 2] * qB2B1[:, 3])
    secondAngle = np.arcsin(np.clip(arcsinArg, -1, 1))
    error = get2DStaticWeight([j1, j2], q1, q2) * secondAngle

    #  Jacobian
    qB2E2 = qmt.qmult(q2, qB2S2)

    dQ3Ba = np.zeros((1, 4))
    dQ3Ba[:, 0] = -0.5 * np.sin(delta / 2)
    dQ3Ba[:, 3] = 0.5 * np.cos(delta / 2)
    dQ3Ba = qmt.normalized(dQ3Ba)
    dQB = qmt.qmult(qmt.qmult(qE1B1, dQ3Ba), qB2E2)
    dQ = dQB
    #
    jac = 2 * (dQ[:, 1] * qB2B1[:, 0] + qB2B1[:, 1] * dQ[:, 0] +
               dQ[:, 2] * qB2B1[:, 3] + qB2B1[:, 2] * dQ[:, 3])
    jac = jac / np.sqrt(1 - arcsinArg ** 2)

    return error, jac


def get2DStaticWeight(jointAxes, quat1, quat2):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_2D.m
    axis1 = jointAxes[0]
    axis2 = jointAxes[1]
    j1 = qmt.rotate(quat1, axis1)
    j2 = qmt.rotate(quat2, axis2)
    weight = np.sqrt(j1[:, 0] ** 2 + j1[:, 1] ** 2) * np.sqrt(j2[:, 0] ** 2 + j2[:, 1] ** 2)
    return weight


def estimateDelta2d(quat1, quat2, gyr1, gyr2, time, joint, deltaStart, constraint, optimizationSteps):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_2D.m
    j1 = joint[0, :]
    j2 = joint[1, :]
    delta = deltaStart
    cost = 0

    for _ in range(optimizationSteps):
        deltaInc, cost = gaussNewtonStep2d(quat1, quat2, gyr1, gyr2, time, j1, j2, delta, constraint)
        delta = delta + deltaInc

    if constraint == 'euler_lin':
        delta = delta[0] * time + delta[1]
    delta = qmt.wrapTo2Pi(delta)
    # %% Rating calculation
    j1Global = qmt.rotate(quat1, j1)
    j2Global = qmt.rotate(quat2, j2)
    rating = qmt.rms(np.sqrt(j1Global[:, 0] ** 2 + j1Global[:, 1] ** 2)
                     * np.sqrt(j2Global[:, 0] ** 2 + j2Global[:, 1] ** 2))
    return delta, rating, cost


def gaussNewtonStep2d(quat1, quat2, gyr1, gyr2, time, j1, j2, delta, constraint):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_2D.m
    errors, jacobi = errorAndJac2D(quat1, quat2, gyr1, gyr2, time, j1, j2, delta, constraint)
    deltaParams = -1 * np.linalg.lstsq(np.atleast_2d(jacobi @ jacobi), np.atleast_2d(jacobi @ errors), rcond=None)[0]
    deltaParams = np.squeeze(deltaParams)
    totalError = np.linalg.norm(errors)
    return deltaParams, totalError


def getJacGyroCons(j1, j2, q1, q2, gyr1, gyr2, delta):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_2D.m
    steps = q1.shape[0]
    eZ = np.zeros((steps, 3)) + np.array([0, 0, 1])
    quatE2E1 = np.array([np.cos(delta / 2), 0, 0, np.sin(delta / 2)])

    j1E1 = qmt.rotate(q1, j1)
    j2E2 = qmt.rotate(q2, j2)
    j2E1 = qmt.rotate(quatE2E1, j2E2)
    w1E1 = qmt.rotate(q1, gyr1)
    w2E2 = qmt.rotate(q2, gyr2)
    w2E1 = qmt.rotate(quatE2E1, w2E2)

    jN = np.cross(j1E1, j2E1)
    wD = w1E1 - w2E1
    errors = np.sum(wD * jN, axis=1)

    # % Calculate one Row of the Jacobian
    dj2B = -j2E2 * np.sin(delta) + np.cross(eZ, j2E2, axis=1) * np.cos(delta) \
        + eZ * (np.sum(eZ * j2E2, axis=1) * np.sin(delta))[:, np.newaxis]
    dwdB = -1 * (-w2E2 * np.sin(delta) + np.cross(eZ, w2E2, axis=1) * np.cos(delta)
                 + eZ * (np.sum(eZ, w2E2, axis=1) * np.sin(delta))[:, np.newaxis])
    dB = np.sum(dwdB * np.cross(j1E1, j2E1, axis=1), axis=1) + np.sum(wD * np.cross(j1E1, dj2B, axis=1), axis=1)
    jac = dB

    return errors, jac


def getJacCombCons(j1, j2, q1, q2, gyr1, gyr2, delta):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/errorAndJac_2D.m
    steps = q1.shape[0]
    eE = np.zeros((steps, 3)) + np.array([0, 0, 1])
    quatE2E1 = np.array([np.cos(delta / 2), 0, 0, np.sin(delta / 2)], dtype=float)
    j1E1 = qmt.rotate(q1, j1)
    j2E2 = qmt.rotate(q2, j2)
    j2E1 = qmt.rotate(quatE2E1, j2E2)
    w1E1 = qmt.rotate(q1, gyr1)
    w2E2 = qmt.rotate(q2, gyr2)
    w2E1 = qmt.rotate(quatE2E1, w2E2)

    jN = np.cross(j1E1, j2E1, axis=1)
    wD = w1E1 - w2E1
    errorsGyro = np.sum(wD * jN, axis=1)

    # % Calculate one Row of the Jacobian
    dj2B = -j2E2 * np.sin(delta) + np.cross(eE, j2E2, axis=1) * np.cos(delta) + \
        eE * (np.sum(eE * j2E2, axis=1) * np.sin(delta))[:, np.newaxis]
    dwdB = -(-w2E2 * np.sin(delta) + np.cross(eE, w2E2, axis=1) * np.cos(delta) +
             eE * (np.sum(eE * w2E2, axis=1) * np.sin(delta))[:, np.newaxis])
    dB = np.sum(dwdB * np.cross(j1E1, j2E1, axis=1), axis=1) + np.sum(wD * np.cross(j1E1, dj2B, axis=1), axis=1)
    jacGyro = dB

    # % Calculate the error
    qE2E1 = qmt.quatFromAngleAxis(delta, [0, 0, 1])
    qB2S2 = qmt.quatFromAngleAxis(np.arccos(np.dot([0, 1, 0], j2)), np.cross([0, 1, 0], j2))
    qB1S1 = qmt.quatFromAngleAxis(np.arccos(np.dot([0, 0, 1], j1)), np.cross([0, 0, 1], j1))

    qE1B1 = qmt.qinv(qmt.qmult(q1, qB1S1))
    qE2B1 = qmt.qmult(qE1B1, qE2E1)
    qS2B1 = qmt.qmult(qE2B1, q2)
    qB2B1 = qmt.qmult(qS2B1, qB2S2)

    arcsinArg = 2 * (qB2B1[:, 1] * qB2B1[:, 0] + qB2B1[:, 2] * qB2B1[:, 3])
    secondAngle = np.arcsin(np.clip(arcsinArg, -1, 1))

    errorsEuler = get2DStaticWeight([j1, j2], q1, q2) * secondAngle

    # % Jacobian
    qB2E2 = qmt.qmult(q2, qB2S2)

    dQ3Ba = np.zeros((1, 4))
    dQ3Ba[:, 0] = -0.5 * np.sin(delta / 2)
    dQ3Ba[:, 3] = 0.5 * np.cos(delta / 2)
    dQ3Ba = qmt.normalized(dQ3Ba)
    dQB = qmt.qmult(qmt.qmult(qE1B1, dQ3Ba), qB2E2)
    dQ = dQB

    jac = 2 * (dQ[:, 1] * qB2B1[:, 0] + qB2B1[:, 1] * dQ[:, 0] +
               dQ[:, 2] * qB2B1[:, 3] + qB2B1[:, 2] * dQ[:, 3])
    jacEuler = jac / np.sqrt(1 - arcsinArg ** 2)

    errors = errorsGyro + errorsEuler
    jac = jacEuler + jacGyro

    return errors, jac


#  3-D joint
def estimateDelta3d(quat1, quat2, angleRanges, convention, deltaRange, deltaStart):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/estimate_delta_3d.m
    # generate the distribution of values for delta that produce a valid relative orientation
    deltaProbability = getPossibleAngles(quat1, quat2, angleRanges[0, :], angleRanges[1, :], angleRanges[2, :],
                                         convention, deltaRange, 'max')

    # get the maximum
    maxVal = max(deltaProbability)

    distributionLastDelta = quat1.shape[0] / np.pi * abs(qmt.wrapToPi(deltaRange - deltaStart))

    deltaProbabilityMin = getPossibleAngles(quat1, quat2, angleRanges[0, :], angleRanges[1, :], angleRanges[2, :],
                                            convention, deltaRange, 'min')
    distribNewMin = deltaProbabilityMin + distributionLastDelta

    # % get the minimum
    minVal = min(distribNewMin)

    delta = np.mean(np.unwrap(deltaRange[distribNewMin == minVal]))
    cost = minVal

    delta = qmt.wrapTo2Pi(delta)

    # %% Rating calculation
    # % calculate the standard deviation of the distribution in order to quantify the quality of the estimation
    _, stdConstraint = stdProbAngles(deltaRange[deltaProbability > maxVal / 2],
                                     deltaProbability[deltaProbability > maxVal / 2])
    #  % scale the standard deviation to a range of [0.1]
    rating = map_(stdConstraint, 0, np.deg2rad(20), 1, 0)
    #  % clip the window rating to values of [0,1]
    rating = min(max(rating, 0), 1)
    return delta, rating, cost


def getPossibleAngles(quat1, quat2, angle1Range, angle2Range, angle3Range, convention, deltaRange, mode):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/estimate_delta_3d.m
    qHand = quat1[~np.isnan(quat1[:, 0]), :]
    qMeta = quat2[~np.isnan(quat2[:, 0]), :]
    assert qHand.shape == qMeta.shape
    # qMeta: Mx4
    # qHand: Mx4
    qE1E2 = qmt.quatFromAngleAxis(deltaRange, [0, 0, 1])
    qE1E2Broadcast, qMetaBroadcast, qHandBroadcast = np.broadcast_arrays(qE1E2[None], qMeta[:, None], qHand[:, None])
    N = qE1E2.shape[0]
    M = qMeta.shape[0]
    qMetaCorr = qmt.qmult(qE1E2Broadcast.reshape(M * N, 4), qMetaBroadcast.reshape(M * N, 4)).reshape(M, N, 4)
    # Mx360x4
    qRel = qmt.qrel(qHandBroadcast.reshape(M * N, 4), qMetaCorr.reshape(M * N, 4)).reshape(M, N, 4)
    angles = qmt.eulerAngles(qRel.reshape(M * N, 4), convention, True).reshape(M * N, 3)
    angleDiff = getAngleDiff(angles.reshape(M * N, 3), angle1Range, angle2Range, angle3Range)
    angleDiff[angleDiff > 0] = 1
    angleDiff = angleDiff.astype(bool).reshape(M, N)
    if mode == 'min':
        possibleAngles = angleDiff
    else:
        possibleAngles = ~angleDiff
    out = np.sum(possibleAngles, axis=0)
    return out


def getAngleDiff(angles, angle1Range, angle2Range, angle3Range):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/estimate_delta_3d.m
    d1 = angle1Range[np.newaxis, :] - angles[:, 0][:, np.newaxis]
    out = abs(np.minimum(-np.sign(d1[:, 0] * d1[:, 1]) * np.min(np.abs(d1), axis=1), 0))

    d2 = angle2Range[np.newaxis, :] - angles[:, 1][:, np.newaxis]
    out = out + abs(np.minimum(-np.sign(d2[:, 0] * d2[:, 1]) * np.min(np.abs(d2), axis=1), 0))

    d3 = angle3Range[np.newaxis, :] - angles[:, 2][:, np.newaxis]
    out = out + abs(np.minimum(-np.sign(d3[:, 0] * d3[:, 1]) * np.min(np.abs(d3), axis=1), 0))

    return out


def stdProbAngles(val=None, prob=None):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/estimate_delta_3d.m
    val = np.array(val)

    if val is None or len(val) < 2:
        mean = 0
        stdDev = 0

        return mean, stdDev

    area = sum(prob) * (val[1] - val[0])
    mean = angularMean(val, prob)
    temp = sum(qmt.wrapToPi(val - mean) ** 2 * prob / area)
    temp = temp * (val[1] - val[0])
    stdDev = np.sqrt(temp)
    return mean, stdDev


def angularMean(angles, probability):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/estimate_delta_3d.m
    N = len(angles)
    mean = np.arctan2(1 / N * sum(probability * np.sin(angles)), 1 / N * sum(probability * np.cos(angles)))
    mean = qmt.wrapTo2Pi(mean)
    return mean


def map_(val, inMin, inMax, outMin, outMax):
    # based on Matlab implementation in matlab/lib/HeadingCorrection/estimate_delta_3d.m
    out = (val - inMin) * (outMax - outMin) / (inMax - inMin) + outMin
    return out


def movMean1D(x, windowSize):
    windowSize = np.ones(windowSize) / windowSize
    x = np.array(x, dtype=float)
    xFilt = signal.correlate(x, windowSize, mode='same')
    return xFilt


def romCost(ROM, convention, q1, q2, delta):
    qParent = q1[~np.isnan(q1[:, 0]), :]
    qChild = q2[~np.isnan(q2[:, 0]), :]

    qE2E1 = qmt.quatFromAngleAxis(delta, [0, 0, 1])

    qChildCorr = qmt.qmult(qE2E1, qChild)
    qRel = qmt.qrel(qParent, qChildCorr)
    angles = qmt.eulerAngles(qRel, convention, True)

    N = angles.shape[0]
    angleRang1 = ROM[0, :]
    angleRang2 = ROM[1, :]
    angleRang3 = ROM[2, :]

    rating = np.zeros((N, 3))

    d1 = angleRang1[np.newaxis, :] - angles[:, [0]]
    rating[:, 0] = np.sign(d1[:, 0] * d1[:, 1])
    d2 = angleRang2[np.newaxis, :] - angles[:, [1]]
    rating[:, 1] = np.sign(d2[:, 0] * d2[:, 1])
    d3 = angleRang3[np.newaxis, :] - angles[:, [2]]
    rating[:, 2] = np.sign(d3[:, 0] * d3[:, 1])

    rating = np.sum(rating)

    return rating


def removeHeadingDriftForStraightWalk(t, quat, winlength, stepthreshold, movmeanwidth, plot=False, debug=False):
    """
    A function that takes a quaternion with slow heading drift and removes that heading drift by assumimg that
    the heading remains constant except for a few large steps (turns) by multiples of pi.
    The function removes the heading steps, then low-pass filters heading and uses it to correct the given quaternion.

    :param t: time vector with a fixed sampling rate
    :param quat: Nx4 input quaternion array
    :param winlength: winlength is the time in which the turns happen (choose e.g. 5 sec).
    :param stepthreshold: stepthreshold defines which heading steps are ignored (not removed).
    :param movmeanwidth: movmeanwidth is the width of the moving average low-pass filter.
    :param plot: enables debug plot
    :param debug: enables debug output
    :return:
        - output: Nx3 or 1x3 vector output array
        - debug: dict with debug values (only if debug==True)
    """

    timediff = np.diff(t)
    rate = 1 / timediff[1]
    assert np.allclose(timediff, 1 / rate)

    qrel = qmt.qmult(quat, qmt.qinv(quat[199, :]))
    h = qmt.eulerAngles(qrel, 'zyx', intrinsic=True)
    h = np.unwrap(h, axis=0)
    h = h[:, 0]
    hsteps1 = np.zeros(h.shape)
    hsteps2 = np.zeros(h.shape)
    hsteps3 = np.zeros(h.shape)
    for i in range(winlength, (h.shape[0] - winlength)):
        hsteps1[i] = np.mean(h[i:i + winlength]) - np.mean(h[i - winlength:i])
    hsteps1 = hsteps1 - np.mean(hsteps1[np.abs(hsteps1) < np.pi / 4])

    for i in range(winlength, (h.shape[0] - winlength)):
        if np.abs(hsteps1[i]) < stepthreshold:
            hsteps2[i] = 0
        else:
            hsteps2[i] = np.round(hsteps1[i] / np.pi) * np.pi
        if hsteps2[i] == 0 and hsteps2[i - 1] != 0:
            # (nonzero rows, cols)
            startind = np.nonzero(hsteps2[:i] == 0)[0][-1]
            ind = np.argmax(np.abs(hsteps1[startind:i]))
            hsteps3[startind + ind] = hsteps2[startind + ind]

    hsteps4 = np.cumsum(hsteps3)
    hstepsfree = h - hsteps4
    drift = movMean1D(hstepsfree, 1000)
    hnice = np.zeros(h.shape)
    driInd = np.abs(hstepsfree - drift) > np.pi / 6
    hnice[driInd] = drift[driInd]
    hnice[~driInd] = hstepsfree[~driInd]
    drift = movMean1D(hnice, movmeanwidth)
    corrquat = np.concatenate((np.cos(-drift / 2)[:, None],
                               np.sin(-drift / 2)[:, None] * np.array([0, 0, 1])), axis=1)
    quatCorrected = qmt.qmult(corrquat, quat)

    qrelCorrected = qmt.qmult(quatCorrected, qmt.qinv(quatCorrected[200, :]))
    hCorrected = np.unwrap(qmt.eulerAngles(qrelCorrected, 'zyx', 1), 30.0 * np.pi)
    hOld = np.unwrap(qmt.eulerAngles(qrel, 'zyx', 1), 3.0 * np.pi)

    if debug or plot:
        debugData = dict(h=h,
                         hsteps1=hsteps1,
                         hsteps2=hsteps2,
                         hsteps3=hsteps3,
                         hsteps4=hsteps4,
                         hstepsfree=hstepsfree,
                         drift=drift,
                         hnice=hnice,
                         t=t,
                         h_corrected=hCorrected[:, 0],
                         h_old=hOld[:, 0])
        if plot:
            removeHeadingDriftForStraightWalk_debugPlot(debugData, plot)
        if debug:
            return quatCorrected, debugData

    return quatCorrected


def removeHeadingDriftForStraightWalk_debugPlot(debug, fig=None):
    with AutoFigure(fig) as fig:
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)
        fig.suptitle('removeHeadDrift debug plot')

        ax1.plot(debug['t'], np.rad2deg(debug['h']), '.-')
        ax1.plot(debug['t'], np.rad2deg(debug['hsteps1']), '.-')
        ax1.plot(debug['t'], np.rad2deg(debug['hsteps2']), '.-')
        ax1.plot(debug['t'], np.rad2deg(debug['hsteps3']), '.-')
        ax1.legend(['h0', 'h1', 'h2', 'h3'])
        ax1.set_ylabel('[°]')
        ax1.set_title('h0~h3')

        ax2.plot(debug['t'], np.rad2deg(debug['h']), '.-')
        ax2.plot(debug['t'], np.rad2deg(debug['hstepsfree']), '.-')
        ax2.plot(debug['t'], np.rad2deg(debug['drift']), '.-')
        ax2.legend(['h', 'hstepsfree', 'drift'])
        ax2.set_title('h , hstepsfree, drift')
        ax2.set_ylabel('[°]')

        ax3.plot(debug['t'], np.rad2deg(debug['hstepsfree']), '.-')
        ax3.plot(debug['t'], np.rad2deg(debug['hnice']), '.-')
        ax3.plot(debug['t'], np.rad2deg(debug['drift']), '.-')
        ax3.legend(['hstepsfree', 'hnice', 'drift'])
        ax3.set_title('hstepsfree, hnice, drift')
        ax3.set_ylabel('[°]')

        ax4.plot(debug['t'], np.rad2deg(debug['h_old']), '.-')
        ax4.plot(debug['t'], np.rad2deg(debug['h_corrected']), '.-')
        ax4.legend(['h_old', 'h_corrected'])
        ax4.set_title('eulerAngle z-axis')
        ax4.set_ylabel('[°]')

        for ax in (ax1, ax2, ax3, ax4):
            ax.grid()
