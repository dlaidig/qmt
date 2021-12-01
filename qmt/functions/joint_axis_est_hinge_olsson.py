# SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
# SPDX-FileCopyrightText: 2021 Fredrik Olsson <fredrik.olsson@it.uu.se>
#
# SPDX-License-Identifier: LicenseRef-Unspecified
# (based on Matlab implementation available at https://github.com/frols88/sensor-to-segment)
import numpy as np

from qmt.utils.struct import Struct
from qmt.utils.plot import AutoFigure


def jointAxisEstHingeOlsson(acc1, acc2, gyr1, gyr2, estSettings=None, debug=False, plot=False):
    """
    This function estimates the 1 DoF joint axes of a kinematic chain of two segments in the local
    coordinates of the sensors attached to each segment respectively.

    Ported to Python based on https://www.mdpi.com/1424-8220/20/12/3534 and
    the Matlab implementation from Fredrik Olsson available at
    https://github.com/frols88/sensor-to-segment.

    Equivalent Matlab function: :mat:func:`+qmt.jointAxisEstHingeOlsson`.

    :param acc1: Nx3 array of angular velocities in m/s^2
    :param acc2: Nx3 array of angular velocities in m/s^2
    :param gyr1: Nx3 array of angular velocities in rad/s
    :param gyr2: Nx3 array of angular velocities in rad/s
    :param estSettings: Dictionary containing settings for estimation. If no settings are given, the default settings
        will be used. Available options:

        - **w0**: Parameter that sets the relative weighting of gyroscope to accelerometer residual, w0 = wg/wa. Default
          value: 50.
        - **wa**: Accelerometer residual weight.
        - **wg**: Gyroscope residual weight
        - **useSampleSelection**: Boolean flag to use sample selection. Default value: False.
        - **dataSize**: Maximum number of samples that will be kept after sample selection. Default value: 1000.
        - **winSize**: Window size for computing the average angular rate energy, should be an odd integer. Default
          value: 21.
        - **angRateEnergyThreshold**: Theshold for the angular rate energy.  Default value: 1.
        - **x0**: Initial state of joint axes in 2 sensors coordinates. Default value: [0, 0, 0, 0] in spherical
          coordinate.

    :param debug: enables debug output
    :param plot: enables debug plot
    :return:
        - **jhat1**: 3x1 array, estimated joint axis in imu1 frame in cartesian coordinate.
        - **jhat2**: 3x1 array, estimated joint axis in imu2 frame in cartesian coordinate.
        - debug: dict with debug values (only if debug==True).
    """

    # check inputs
    Na = np.max(acc1.shape)
    Ng = np.max(gyr2.shape)
    assert acc1.shape == (Na, 3) and acc2.shape == (Na, 3), 'Incorrect input shape of acc.'
    assert gyr1.shape == (Ng, 3) and gyr1.shape == (Ng, 3), 'Incorrect input shape of acc.'

    # In data Struct  acc/gyr stack column-wise instead
    imu1 = Struct(acc=acc1.T, gyr=gyr1.T)
    imu2 = Struct(acc=acc2.T, gyr=gyr2.T)

    if estSettings is None:
        estSettings = Struct()
    elif not isinstance(estSettings, (dict, Struct)):
        raise TypeError('Wrong data type of estSetting')
    else:
        estSettings = Struct(estSettings)

    # %% Sample selection parameters
    #  Boolean flag to use sample selection
    useSampleSelection = estSettings.get('useSampleSelection', False)
    #  Maximum number of samples that will be kept after sample selection
    dataSize = estSettings.get('dataSize', 2000)
    #  Window size for computing the average angular rate energy, should be an odd integer
    winSize = estSettings.get('winSize', 21)
    #  Threshold for the angular rate energy
    angRateEnergyThreshold = estSettings.get('angRateEnergyThreshold', 1)

    sampleSelectionVars = Struct()
    if useSampleSelection:
        sampleSelectionVars['useSampleSelection'] = useSampleSelection
        sampleSelectionVars['dataSize'] = dataSize
        sampleSelectionVars['winSize'] = winSize
        sampleSelectionVars['angRateEnergyThreshold'] = angRateEnergyThreshold
        sampleSelectionVars['deltaGyr'] = []
        sampleSelectionVars['gyrSamples'] = []
        sampleSelectionVars['accSamples'] = []
        sampleSelectionVars['accScore'] = []
        sampleSelectionVars['angRateEnergy'] = []
        # imu1, imu2, sampleSelectionVars = jointAxisSampleSelection(imu1, imu2, sampleSelectionVars)
        # return imu1, imu2, sampleSelectionVars
        gyr, acc, sampleSelectionVars = jointAxisSampleSelection(imu1, imu2, sampleSelectionVars)
        imu1['gyr'] = gyr[0:3, :]
        imu1['acc'] = acc[0:3, :]
        imu2['gyr'] = gyr[3:6, :]
        imu2['acc'] = acc[3:6, :]
    # %%Identification
    # Initial estimate
    estSettings['x0'] = estSettings.get('x0', np.array([0, 0, 0, 0]).reshape(-1, 1))

    # % Parameter that sets the relative weighting of gyroscope to accelerometer residual, w0 = wg/wa
    weights = {}
    w0 = estSettings.get('w0', 50.0)
    weights['w0'] = w0
    # Accelerometer residual weight
    weights['wa'] = estSettings.get('wa', 1 / np.sqrt(w0))
    # Gyroscope residual weight
    weights['wg'] = estSettings.get('wg', np.sqrt(w0))
    estSettings['weights'] = weights

    jhat, xhat, optimVarsAxis = jointAxisIdent(imu1, imu2, estSettings)

    jhat1 = jhat[0:3, :]
    jhat2 = jhat[3:, :]
    if debug or plot:
        debugData = dict(
            acc1=acc1,
            gyr1=gyr1,
            jhat=jhat,
            xhat=xhat,
            optimVarsAxis=optimVarsAxis,
            sampleSelectionVars=sampleSelectionVars,
        )
        if plot:
            jointAxisEstHingeOlsson_debugPlot(debugData, plot)
        if debug:
            return jhat1, jhat2, debugData
    return jhat1, jhat2


def jointAxisEstHingeOlsson_debugPlot(debug, figs=None):
    useSampleSelection = debug['sampleSelectionVars'].get('useSampleSelection', False)
    xtraj = debug['optimVarsAxis']['xtraj']
    loss = debug['optimVarsAxis']['ftraj']
    xtraj = xtraj.T
    x1 = xtraj[:, 0:2]
    x2 = xtraj[:, 2:]
    j1 = spherical2Vector(x1)
    j2 = spherical2Vector(x2)
    from matplotlib import pyplot as plt
    with AutoFigure(figs[0] if isinstance(figs, (list, tuple)) else None) as fig:
        fig.suptitle(AutoFigure.title('jointAxisEstHingeOlsson'))

        plt.subplot(321)
        plt.plot(loss, '.-')
        plt.grid()
        plt.title(f'loss value, {loss.shape}')

        plt.subplot(323)
        plt.plot(j1, '.-')
        plt.grid()
        plt.title(f'jhat1, {j1.shape}')

        plt.subplot(325)
        plt.plot(j2, '.-')
        plt.grid()
        plt.title(f'jhat2, {j2.shape}')

        plt.subplot(122, projection='3d')
        plt.plot(j1[:, 0], j1[:, 1], j1[:, 2], '-x', label='jhat1')
        plt.plot(j2[:, 0], j2[:, 1], j2[:, 2], '-x', label='jhat2')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylabel("z")
        plt.grid()
        plt.legend()
        plt.title(f'joint axis trajectory, {j1.shape}')

        fig.tight_layout()

    if useSampleSelection:
        with AutoFigure(figs[1] if isinstance(figs, (list, tuple)) else None) as fig:
            fig.suptitle(AutoFigure.title('jointAxisEstHingeOlsson sample selection'))
            Ng = max(debug['gyr1'].shape)
            Na = max(debug['acc1'].shape)
            gyrInd = debug['sampleSelectionVars']['gyrSamples']
            accInd = debug['sampleSelectionVars']['accSamples']

            plt.subplot(211)
            plt.plot(np.arange(Ng), np.arange(Ng), linestyle='solid')
            plt.plot(gyrInd, gyrInd, '.')
            plt.title(f'gyr samples, {gyrInd.shape} of {debug["gyr1"].shape}')
            plt.grid()
            plt.subplot(212)
            plt.plot(np.arange(Na), np.arange(Na), linestyle='solid')
            plt.plot(accInd, accInd, '.')
            plt.title(f'acc samples, {accInd.shape} of {debug["acc1"].shape}')
            plt.grid()

            fig.tight_layout()


def jointAxisIdent(imu1, imu2, settings):
    # based on Matlab implementation in matlab/lib/Sensor2SegmentCalibration/jointAxisIdent.m

    # % Use default settings if no settings struct is provided
    # Initialize as uniformly random unit vectors
    x0 = -np.pi + 2 * np.pi * np.random.rand(4, 1)
    if settings is None:
        settings = Struct()
    residuals = settings.get('residuals', [1, 2])
    weights = settings.get('weights', {'wa': 1, 'wg': 1})
    loss = settings.get('loss', lambda e: lossFunctions(e, 'squared'))
    optOptions = settings.get('optOptions', optimOptions(settings))
    x0 = settings.get('x0', x0)
    if x0.ndim < 2:
        x0 = x0.reshape(-1, 1)

    # %% Optimization
    # % Define cost function
    def costFunc(x):
        return jointAxisCost(x, imu1, imu2, residuals, weights, loss)

    xhat1, optimVars1 = optimGaussNewton(x0, costFunc, optOptions)
    # xhat1 : 4x1

    # % Convert from spherical coordinates to unit vectors

    nhat = np.array([
        np.cos(xhat1[0, :]) * np.cos(xhat1[1, :]),
        np.cos(xhat1[0, :]) * np.sin(xhat1[1, :]),
        np.sin(xhat1[0, :]),

        np.cos(xhat1[2, :]) * np.cos(xhat1[3, :]),
        np.cos(xhat1[2, :]) * np.sin(xhat1[3, :]),
        np.sin(xhat1[2, :])
    ])

    xhat1 = np.concatenate((
        vector2Spherical(nhat[0:3, :].reshape(-1, 3)).reshape(-1, 1),
        vector2Spherical(nhat[3:6, :].reshape(-1, 3)).reshape(-1, 1)),
        axis=0)

    x0 = np.concatenate((
        vector2Spherical(nhat[0:3, :].reshape(-1, 3)).reshape(-1, 1),
        vector2Spherical(-nhat[3:6, :].reshape(-1, 3)).reshape(-1, 1)),
        axis=0)

    xhat2, optimVars2 = optimGaussNewton(x0, costFunc, optOptions)

    if optimVars2['f'] < optimVars1['f']:
        xhat = xhat2
        optimVars = optimVars2
        optimVars['flip'] = optimVars1
    else:
        xhat = xhat1
        optimVars = optimVars1
        optimVars['flip'] = optimVars2

    nhat = np.array([
        np.cos(xhat[0, :]) * np.cos(xhat[1, :]),
        np.cos(xhat[0, :]) * np.sin(xhat[1, :]),
        np.sin(xhat[0, :]),
        np.cos(xhat[2, :]) * np.cos(xhat[3, :]),
        np.cos(xhat[2, :]) * np.sin(xhat[3, :]),
        np.sin(xhat[2, :]),
    ])
    xhat = np.concatenate((
        vector2Spherical(nhat[0:3].reshape(-1, 3)).reshape(-1, 1),
        vector2Spherical(nhat[3:6].reshape(-1, 3)).reshape(-1, 1)), axis=0)

    return nhat, xhat, optimVars


def lossFunctions(e, type_, param=None):
    # based on Matlab implementation in matlab/lib/Sensor2SegmentCalibration/lossFunctions.m
    if type_ == 'squared':
        loss = np.linalg.norm(e, axis=1, keepdims=True) ** 2
        dlde = 2 * e
    elif type_ == 'huber':
        if param is None:
            delta = 1
        else:
            delta = param[0]
        ind = np.linalg.norm(e, ord=1, axis=1) < delta
        loss = np.linalg.norm(e, axis=1, keepdims=True) ** 2 / 2
        dlde = e
        loss[:, ~ind] = delta * (np.linalg.norm(e, ord=1, axis=1) - delta / 2)
        dlde[:, ~ind] = delta * np.sign(e[:, ~ind])
    elif type_ == 'absolute':
        loss = np.linalg.norm(e, ord=1, axis=1)
        dlde = np.sign(e)
    else:
        raise ValueError("Type of loss function undefined")
    return loss, dlde


def jointAxisCost(x, imu1, imu2, residuals, weights, loss):
    # based on Matlab implementation in matlab/lib/Sensor2SegmentCalibration/jointAxisCost.m

    # Load data variables from imus struct
    # Note: shape of gyr/acc 3xN
    gyr1 = imu1['gyr']
    Ng = gyr1.shape[1]
    acc1 = imu1['acc']
    Na = acc1.shape[1]

    # Initialize
    if imu2 is None:
        imu2 = Struct()
        imu2['gyr'] = np.zeros((3, Ng))
        imu2['acc'] = np.zeros((3, Na))
    gyr2 = imu2['gyr']
    acc2 = imu2['acc']

    if residuals is None:
        residuals = [1, 2]

    if weights is None:
        wg = 1.0
        wa = 1.0
    else:
        assert isinstance(weights['wg'], (int, float)) or \
               ((isinstance(weights['wg'], np.ndarray) and len(weights['wg'].shape))), 'Invalid weights type wg.'
        wg = weights['wg']
        assert isinstance(weights['wa'], (int, float)) or \
               ((isinstance(weights['wa'], np.ndarray) and len(weights['wa'].shape))), 'Invalid weights type wa.'
        wa = weights['wa']

    if loss is None:
        def loss(err): lossFunctions(err, 'squared')

    Nr = len(residuals)
    N = Ng + Na
    # Residuals
    e = np.zeros((N, 1))
    # Gradient
    g = np.zeros((4, 1))
    # cost function value
    f = 0
    # Jacobian
    J = np.zeros((N, 4))

    # % Current estimated normal vectors
    x1 = np.zeros((2, 1))
    x2 = np.zeros((2, 1))
    if max(x.shape) > 4:
        n1 = x[0:3, :]
        n2 = x[3:6, :]
        x1[0, :] = np.arcsin(n1[2, :])
        x1[1, :] = np.arccos(n1[0, :] / np.cos(x1[0, :]))
        x2[0, :] = np.arcsin(n2[2, :])
        x2[1, :] = np.arccos(n2[0, :] / np.cos(x2[0, :]))
    else:
        # in spherical, (2,)
        x1 = x[0:2, 0]
        x2 = x[2:4, 0]
        # in cartesian, 3x1. j
        n1 = np.array([np.cos(x1[0]) * np.cos(x1[1]), np.cos(x1[0]) * np.sin(x1[1]), np.sin(x1[0])]).reshape(-1, 1)
        n2 = np.array([np.cos(x2[0]) * np.cos(x2[1]), np.cos(x2[0]) * np.sin(x2[1]), np.sin(x2[0])]).reshape(-1, 1)

    # % Partial derivatives of normal vectors n w.r.t. spherical coordinates x
    # 2x3,  ∂j/∂x
    dn1dx1 = np.array([[-np.sin(x1[0]) * np.cos(x1[1]), -np.sin(x1[0]) * np.sin(x1[1]), np.cos(x1[0])],
                       [-np.cos(x1[0]) * np.sin(x1[1]), np.cos(x1[0]) * np.cos(x1[1]), 0]])

    dn2dx2 = np.array([[-np.sin(x2[0]) * np.cos(x2[1]), -np.sin(x2[0]) * np.sin(x2[1]), np.cos(x2[0])],
                       [-np.cos(x2[0]) * np.sin(x2[1]), np.cos(x2[0]) * np.cos(x2[1]), 0]])

    # Evaluate cost function and Jacobian
    for j in range(Nr):
        if residuals[j] == 1:

            # ||j X gyr||,  (N,)
            ng1 = np.linalg.norm(np.cross(gyr1, n1, axis=0), axis=0)
            ng2 = np.linalg.norm(np.cross(gyr2, n2, axis=0), axis=0)

            ind = np.logical_or(np.logical_or(ng1 == 0, ng2 == 0),
                                np.logical_or(np.isnan(ng1), np.isnan(ng2)))

            # 1xN * 3xN * N,
            # degdn: ∂e_w/∂j
            # why wg ????
            degdn1 = wg * np.cross(np.cross(gyr1, n1, axis=0), gyr1, axis=0) / ng1.reshape(1, -1)
            degdn2 = -wg * np.cross(np.cross(gyr2, n2, axis=0), gyr2, axis=0) / ng2.reshape(1, -1)

            # residual gyr
            # why no wg: weight?
            eg = (ng1 - ng2).reshape(1, -1)

            # ind = ng1 == 0 or ng2 == 0 or (np.isnan(ng1) or np.isnan(ng1))
            degdn1[:, ind] = np.zeros((3, 1))
            degdn2[:, ind] = np.zeros((3, 1))
            eg[:, ind] = 0

            # Ngx4 = [[2x3 * 3xN],[2x3 * 3xN]].T
            J[0:Ng, :] = np.concatenate((np.matmul(dn1dx1, degdn1),
                                         np.matmul(dn2dx2, degdn2)), axis=0).T

            # Ngx1, # residual gyr
            e[0:Ng, :] = wg * eg.T
            # Nx1, Nx1
            l, dlde = loss(e[0:Ng, :])

            f = f + np.sum(l)
            g = g + np.sum(dlde * J[0:Ng, :], axis=0, keepdims=True).T

        elif residuals[j] == 2:
            # acc shape 3xN
            # ea1 N,
            ea1 = np.sum(acc1 * n1, axis=0) - np.sum(acc2 * n2, axis=0)
            # reshape to 1xN
            ea1 = ea1.reshape(1, -1)

            # ∂e_a /∂j
            # wa: Nx1, acc: 3xN = 3xN
            dea1dn1 = wa * acc1
            dea1dn2 = -wa * acc2
            # dea1dn1 3xN
            ind = np.logical_or(np.isnan(acc1).any(axis=0), np.isnan(acc2).any(axis=0))
            dea1dn1[:, ind] = np.zeros((3, 1))
            dea1dn2[:, ind] = np.zeros((3, 1))
            ea1[:, ind] = 0

            # dn1dx1: 2x3, dea1dn1: 3xN
            # J: Nx4
            J[Ng:, :] = np.concatenate((np.matmul(dn1dx1, dea1dn1),
                                        np.matmul(dn2dx2, dea1dn2)), axis=0).T

            e[Ng:, :] = wa * ea1.T
            l, dlde = loss(e[Ng:, :])

            f = f + np.sum(l)
            # dlde: Nx1
            # J: 2Nx4
            g = g + np.sum(dlde * J[Ng:, :], axis=0, keepdims=True).T
    P = np.linalg.inv(np.matmul(J.T, J)) * (f / (Na + Ng - 4))
    return f, g, e, J, P


def optimGaussNewton(x, costFunc, options=None):
    # based on Matlab implementation in matlab/lib/Sensor2SegmentCalibration/optimGaussNewton.m

    if options is None:
        options = optimOptions()
    tol = options['tol']
    maxSteps = options['maxSteps']
    alpha = options['alpha']
    beta = options['beta']
    quiet = options['quiet']
    incMax = 5

    f_prev = 0
    diff = tol + 1
    step = 0
    xtraj = np.zeros((x.shape[0], maxSteps + 1))
    xtraj[:, [step]] = x
    fMins = np.zeros((incMax, 1))
    xMins = np.zeros((x.shape[0], incMax))
    cInc = 0
    optimVars = Struct()
    ftraj = []
    #  Gauss-Newton optimization
    while step < maxSteps and diff > tol:
        # Evaluate cost function, Jacobian and residual
        f, g, e, J, P = costFunc(x)

        # Save initial parameters and cost function value
        if step == 0:
            optimVars['f0'] = f
            optimVars['x0'] = x

        # Backtracking line search
        # Initial step size
        length = 1
        # Search direction
        dx = np.matmul(np.linalg.pinv(J), e)
        # dx = np.matmul(np.matmul(np.linalg.pinv(np.matmul(J.T, J)), J.T), e)
        f_next, _, _, _, _ = costFunc(x - length * dx)

        while (f_next > f + alpha * length * np.matmul(J, dx)).all():
            length = beta * length
            f_next, _, _, _, _ = costFunc(x - length * dx)
        # Handle increased fval
        if f_next > f_prev and step > 0:
            fMins[cInc, :] = f_prev
            xMins[:, [cInc]] = x
            cInc = cInc + 1

        # Update
        x = x - length * dx
        step = step + 1
        if x.shape[1] > 1:
            x = x[:, [0]]
        xtraj[:, [step]] = x
        if step > 1:
            diff = np.linalg.norm(f_prev - f_next)
        f_prev = f_next
        ftraj.append(f_prev)
        if cInc >= incMax:
            fMins = np.concatenate((fMins, np.atleast_2d(f_next)), axis=0)
            xMins = np.concatenate((xMins, x), axis=1)
            minInd = fMins == np.min(fMins)
            x = xMins[:, minInd.squeeze()]
            if not quiet:
                print('Gauss-Newton. Cost function increased, picking found minimum.')

        if not quiet:
            print('Gauss-Newton. Step ' + str(step) + '. f = ' + str(f_next) + ' .')
            if step > maxSteps:
                print('Gauss-Newton. Maximum iterations reached.')
            elif diff <= tol:
                print('Gauss-Newton. Cost function update less than tolerance.')
    xtraj[:, step:] = np.nan * np.ones((x.shape[0], 1)) * np.ones((1, xtraj[:, step:].shape[1]))
    if not quiet:
        print('Gauss-Newton. Stopped after ' + str(step) + ' iterations.')

    f, g, e, J, P = costFunc(x)
    ftraj.append(f)
    ftraj = np.array(ftraj).reshape(-1, 1)
    optimVars['xtraj'] = xtraj
    optimVars['f'] = f
    optimVars['Hessian'] = np.matmul(J.T, J)
    optimVars['costFunc'] = costFunc
    optimVars['ftraj'] = ftraj
    optimVars['x'] = x

    return x, optimVars


def optimGradientDescent(x, costFunc, options=None):
    # based on Matlab implementation in matlab/lib/Sensor2SegmentCalibration/optimGradientDescent.m
    if options is None:
        options = optimOptions()
    tol = options['.tol']
    maxSteps = options['maxSteps']
    alpha = options['alpha']
    beta = options['beta']
    f_prev = 0
    diff = 10
    step = 0
    optimVars = Struct()
    xtraj = np.zeros((x.shape[0], maxSteps + 1))
    xtraj[:, [step]] = x
    ftraj = []

    # Gradient descent optimization
    while step < maxSteps and diff > tol:
        # Evaluate cost function and gradient
        f, g, e, J, P = costFunc(x)
        # % Backtracking line search
        length = 1
        f_next, _, _, _, _ = costFunc(x - length * g)
        while f_next > f - alpha * length * np.linalg.norm(g) ** 2:
            length = beta * length
            f_next, _, _, _, _ = costFunc(x - length * g)

        # % Update
        x = x - length * g
        step = step + 1
        xtraj[:, [step]] = x
        if step > 1:
            diff = np.linalg.norm(f_prev - f_next)
        f_prev = f_next
        ftraj.append(f_prev)
        # Print cost function value
        print(f'Gradient descent. Step: ,{step}, f = {f_next}.')
        if step > maxSteps:
            print('Gradient descent. Maximum iterations reached.')
        elif diff <= tol:
            print('Gradient descent. Cost function update less than tolerance.')
    xtraj[:, step:] = np.nan * np.ones((x.shape[0], 1)) * np.ones((1, xtraj[:, step:].shape[0]))
    print(f'Gauss-Newton. Stopped after:  {step} iterations.')

    f, g, e, J, P = costFunc(x)
    ftraj.append(f)
    ftraj = np.array(ftraj).reshape(-1, 1)
    optimVars['xtraj'] = xtraj
    optimVars['f'] = f
    optimVars['Hessian'] = np.matmul(J.T, J)
    optimVars['costFunc'] = costFunc
    optimVars['ftraj'] = ftraj

    return x, optimVars


def optimOptions(optionsInput=None):
    # based on Matlab implementation in matlab/lib/Sensor2SegmentCalibration/optimOptions.m
    # Default options
    defaults = {'tol': 1e-5,
                'maxSteps': 300 - 1,
                'alpha': 0.4,
                'beta': 0.5,
                'quiet': False}
    # Minimum tolerance in cost function update
    # Maximum number of steps allowed
    # Line search parameter 0 < alpha < 0.5
    # Line search parameter 0 < beta < 1
    # % Quiet printing
    options = {}
    options.update(defaults)
    if optionsInput is not None:
        options.update(optionsInput)

        tol = options.get('tol')
        maxSteps = options.get('maxSteps')
        alpha = options.get('alpha')
        beta = options.get('beta')
        quiet = options.get('quiet')

        if isinstance(tol, np.ndarray) and (len(tol.shape) < 2 and tol > 0):
            pass
        elif isinstance(tol, (int, float)) and tol > 0:
            pass
        else:
            raise ValueError('Optimization options: tol has to be scalar and > 0.')

        if isinstance(maxSteps, np.ndarray) and (len(maxSteps.shape) < 2 and maxSteps > 1):
            pass
        elif isinstance(maxSteps, (int, float)) and maxSteps > 1:
            pass
        else:
            raise ValueError('Optimization options: maxSteps has to be scalar and > 1.')

        if isinstance(alpha, np.ndarray) and (len(alpha.shape) < 2 and 0 < alpha < 0.5):
            pass
        elif isinstance(alpha, (int, float)) and 0 < alpha < 0.5:
            pass
        else:
            raise ValueError('Optimization options: 0 < alpha < 0.5.')

        if isinstance(beta, np.ndarray) and (len(beta.shape) < 2 and 0 < beta < 1):
            pass
        elif isinstance(beta, (int, float)) and 0 < beta < 1:
            pass
        else:
            raise ValueError('Optimization options: 0 < beta < 0.5.')

        if not isinstance(quiet, bool):
            raise ValueError('Optimization option: quiet not boolean')

    return options


def vector2Spherical(vec, outR=False, debug=False, plot=False):
    """
    Convert vector in cartesian coordinate into spherical coordinate
    :param vec: Input vector in cartesian coordinate, in shape Nx3.
    :param outR: Boolean. If set to False, only (θ, φ) will be returned, otherwise (θ, φ), r will be returned.
    :param debug: enables returning debug data
    :param plot: enables the debug plot
    :return: vector in spherical coordinate,
    """
    v = np.array(vec).copy()

    # v = qmt.normalized(v)
    is1D = len(v.shape) < 2
    if is1D:
        assert v.shape == (3,)
        assert np.linalg.norm(v) == 1, "input vector is not unit vector"
        x = np.array([np.arcsin(v[2]), np.arctan2(v[1], v[0])])
        r = np.linalg.norm(vec)
        # axis = None
    else:
        N = v.shape[0]
        assert v.shape == (N, 3), 'Invalid input shape.'
        assert (np.isclose(np.linalg.norm(v, axis=1), 1)).all(), "input vectors must be unit vectors"
        N = v.shape[0]
        x = np.zeros((N, 2))

        x[:, 0] = np.arcsin(v[:, 2])
        x[:, 1] = np.arctan2(v[:, 1], v[:, 0])

    if debug or plot:
        debugData = dict(
            x=x,
            # r=r,
            v=v,
            # axis=axis,
        )
        if plot:
            vector2Spherical_debugPlot(debugData, True)
        if debug:
            return x, debugData
    if outR:
        return x, r
    return x


def vector2Spherical_debugPlot(debug, fig):

    with AutoFigure(fig) as fig:
        v = debug['v'].reshape(-1, 3)
        x = debug['x'].reshape(-1, 2)
        (ax1, ax2) = fig.subplots(1, 2, sharex=False)
        ax1.plot(v, '.-')
        ax1.legend('xyz')
        ax1.set_title(f'vector, {v.shape} {v.dtype}')

        ax2.plot(x, '.-')
        ax2.legend('θφ')
        ax2.set_title(f'spherical, {x.shape} {x.dtype}')

        for ax in (ax1, ax2):
            ax.grid()


def spherical2Vector(x, r=1, debug=False, plot=False):
    """
    Convert vector in spherical coordinate into cartesian coordinate.
    :param x: Input (θ, φ) in spherical coordinate, in shape Nx2.
    :param r: Input number r in spherical coordinate, default value: r=1.
    :param debug: enables returning debug data
    :param plot: enables the debug plot
    :return: vector in spherical coordinate,

    """
    x = np.array(x)
    is1D = len(x.shape) < 2
    if is1D:
        assert x.shape == (2,)
        v = r * np.array([np.cos(x[0]) * np.cos(x[1]), np.cos(x[0]) * np.sin(x[1]), np.sin(x[0])])
    else:
        N = max(x.shape)
        if x.shape == (N, 2) or x.shape == (1, 2):
            v = np.array([
                np.cos(x[:, 0]) * np.cos(x[:, 1]),
                np.cos(x[:, 0]) * np.sin(x[:, 1]),
                np.sin(x[:, 0])
            ]).T * r
        #     keep shape of input Nx3
        else:
            raise ValueError('Invalid input shape')

        if debug or plot:
            debugData = dict(
                v=v,
                x=x,
            )
            if plot:
                spherical2Vector_debugPlot(debugData, plot)
            if debug:
                return x, debugData
    return v


def spherical2Vector_debugPlot(debug, fig):

    with AutoFigure(fig) as fig:
        v = debug['v'].reshape(-1, 3)
        x = debug['x'].reshape(-1, 2)
        (ax1, ax2) = fig.subplots(1, 2, sharex=False)
        ax1.plot(v, '.-')
        ax1.legend('xyz')
        ax1.set_title(f'vector, {v.shape} {v.dtype}')

        ax2.plot(x, '.-')
        ax2.legend('θφ')
        ax2.set_title(f'spherical, {x.shape} {x.dtype}')

        for ax in (ax1, ax2):
            ax.grid()


def jointAxisSampleSelection(imu1, imu2, sampleSelectionVars):
    # based on Matlab implementation in matlab/lib/Sensor2SegmentCalibration/jointAxisSampleSelection.m
    gyrNew = np.concatenate((imu1['gyr'], imu2['gyr']), axis=0)
    accNew = np.concatenate((imu1['acc'], imu2['acc']), axis=0)

    M = gyrNew.shape[1]
    n = sampleSelectionVars['winSize']
    N = sampleSelectionVars['dataSize']
    angRateEnergyThreshold = sampleSelectionVars.get('angRateEnergyThreshold', 1)

    # Remove irregular new measurements (set to NaN)
    diffGyr = np.diff(gyrNew, axis=1, prepend=0)
    indGyr = np.logical_or(np.linalg.norm(diffGyr[0:3, :], axis=0) < 3 * np.finfo(float).eps,
                           np.linalg.norm(diffGyr[3:6, :], axis=0) < 3 * np.finfo(float).eps)
    gyrNew[0:3, indGyr] = np.nan * np.ones((3, 1))

    diffAcc = np.diff(accNew, axis=1, prepend=0)
    indACC = np.logical_or(np.linalg.norm(diffAcc[0:3, :], axis=0) < 3 * np.finfo(float).eps,
                           np.linalg.norm(diffAcc[3:6, :], axis=0) < 3 * np.finfo(float).eps)
    accNew[3:6, indACC] = np.nan * np.ones((3, 1))

    # Gyr magnitude difference
    deltaGyr = np.zeros((M, 1))
    deltaGyrFilt = np.zeros((M, 1))
    absGyr = np.zeros((M, 1))

    absGyr = (np.linalg.norm(gyrNew[0:3, :], axis=0) + np.linalg.norm(gyrNew[3:6, :], axis=0)).reshape(absGyr.shape)
    deltaGyr = (np.linalg.norm(gyrNew[0:3, :], axis=0) - np.linalg.norm(gyrNew[3:6, :], axis=0)).reshape(deltaGyr.shape)

    nn = int((n - 1) / 2)
    k1 = nn
    k2 = M - nn - 1

    for k in range(M):
        if k1 <= k <= k2:
            dgk = deltaGyr[k - nn:k + nn + 1]
            adgk = np.abs(dgk)
            kmin = np.where(adgk == adgk.min())[0]
            if kmin.size == 0:
                deltaGyrFilt[k, :] = 0
            else:
                deltaGyrFilt[k, :] = dgk[kmin[0], :]
            if np.isnan(deltaGyrFilt[k, :]):
                deltaGyrFilt[k, :] = 0
        else:
            deltaGyrFilt[k, :] = 0

    gyr = gyrNew.copy()
    gyrSamples = np.arange(0, M).reshape(-1, 1)
    deltaGyr = deltaGyrFilt.copy()

    # Remove NaN measurements
    notNaN = ~np.isnan(gyr[1, :])
    # Note: gyr shape 6xN
    # deltaGyr shape Nx1, gyrSamples: N
    gyr = gyr[:, notNaN]
    gyrSamples = gyrSamples[notNaN, :]
    deltaGyr = deltaGyr[notNaN, :]

    notNaN = ~np.isnan(gyr[4, :])
    gyr = gyr[:, notNaN]
    gyrSamples = gyrSamples[notNaN, :]
    deltaGyr = deltaGyr[notNaN, :]

    # Pick gyro samples
    gyrSort = np.argsort(-deltaGyr, axis=None, kind='mergesort')
    deltaGyr = -np.sort(-deltaGyr, axis=None, kind='mergesort').reshape(deltaGyr.shape)

    gyr = gyr[:, gyrSort]
    gyrSamples = gyrSamples[gyrSort, :]

    if gyr.shape[1] > N:
        gyr = np.concatenate((gyr[:, 0:int(N / 2)], gyr[:, -1 - int(N / 2) + 1:]), axis=1)
        deltaGyr = np.concatenate((deltaGyr[0:int(N / 2), :], deltaGyr[-1 - int(N / 2) + 1:, :]), axis=0)
        gyrSamples = np.concatenate((gyrSamples[0:int(N / 2), :], gyrSamples[-1 - int(N / 2) + 1:, :]), axis=0)

    # % Detect low angular rate
    angRateEnergy = np.zeros((M, 1))
    gyr1energy = np.linalg.norm(gyrNew[0:3, :], axis=0) ** 2
    gyr2energy = np.linalg.norm(gyrNew[3:6, :], axis=0) ** 2
    for k in range(M):
        if k1 <= k <= k2:
            g1k = gyr1energy[k - nn:k + nn + 1]
            g2k = gyr2energy[k - nn:k + nn + 1]
            angRateEnergy[k, :] = min(np.mean(g1k[~np.isnan(g1k)]), np.mean(g2k[~np.isnan(g2k)]))
        else:
            angRateEnergy[k, :] = np.nan

    acc = accNew.copy()
    accSamples = np.arange(0, M).reshape(-1, 1)
    angRateEnergy = angRateEnergy.copy()
    # Remove NaN measurements
    notNaN = ~np.isnan(acc[1, :])
    # Note: gyr shape 6xN
    # deltaGyr shape Nx1, gyrSamples: N
    acc = acc[:, notNaN]
    accSamples = accSamples[notNaN, :]
    angRateEnergy = angRateEnergy[notNaN, :]

    notNaN = ~np.isnan(acc[4, :])
    acc = acc[:, notNaN]
    accSamples = accSamples[notNaN, :]
    angRateEnergy = angRateEnergy[notNaN, :]
    notNaN = ~np.isnan(angRateEnergy).squeeze()
    acc = acc[:, notNaN]
    accSamples = accSamples[notNaN, :]
    angRateEnergy = angRateEnergy[notNaN, :]

    # % Compute score and sort
    accScore = angRateEnergy.copy()
    accSort = np.argsort(accScore, axis=None, kind='mergesort')
    acc = acc[:, accSort]
    accSamples = accSamples[accSort, :]
    accScore = accScore[accSort, :]
    angRateEnergy = angRateEnergy[accSort, :]

    # Remove samples with too high energy
    if acc.shape[1] > N:
        ind = angRateEnergy > angRateEnergyThreshold
        ind = ind.squeeze()
        acc = acc[:, ~ind]
        angRateEnergy = angRateEnergy[~ind, :]
        accSamples = accSamples[~ind, :]

    # Singular value decomposition
    from scipy.sparse.linalg import svds
    while acc.shape[1] > N:
        A = np.concatenate((acc[0:3, :].T, -acc[3:6, :].T), axis=1)
        U, s, V = svds(A, k=2)
        # scipy.sparse.linalg.svds in different oder
        # V: shape (k, 6)
        sInd = np.argsort(-s)
        s = -np.sort(-s)
        V = V[sInd, :].T
        nRemove = int(min(np.floor(s[1] / s[0] * acc.shape[1]), acc.shape[1] - N - 1) + 1)
        v1 = V[:, [0]]
        Anorm = np.linalg.norm(A, axis=1)
        remove = np.abs(np.matmul(A, v1)).squeeze() / Anorm
        remove = np.where(remove > 0.5)[0]
        if max(remove.shape) > nRemove:
            remove = remove[-1 - nRemove + 1:]
        acc = np.delete(acc, remove, axis=1)

        accSamples = np.delete(accSamples, remove, axis=0)
        accScore = np.delete(accScore, remove, axis=0)
        angRateEnergy = np.delete(angRateEnergy, remove, axis=0)
    # Save variables
    sampleSelectionVars['deltaGyr'] = deltaGyr
    sampleSelectionVars['gyrSamples'] = gyrSamples
    sampleSelectionVars['accSamples'] = accSamples
    sampleSelectionVars['accScore'] = accScore
    sampleSelectionVars['angRateEnergy'] = angRateEnergy

    # imu1['gyrOri'] = imu1['gyr']
    # imu1['accOri'] = imu1['acc']
    # imu1['gyr'] = gyr[0:3, :]
    # imu1['acc'] = acc[0:3, :]
    # imu2['gyrOri'] = imu2['gyr']
    # imu2['accOri'] = imu2['acc']
    # imu2['gyr'] = gyr[3:6, :]
    # imu2['acc'] = acc[3:6, :]

    # return imu1, imu2, sampleSelectionVars
    return gyr, acc, sampleSelectionVars
