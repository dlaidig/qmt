% SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function  [jhat1, jhat2, debug] = jointAxisEstHingeOlsson(acc1, acc2, gyr1, gyr2, estSettings)
    % This function estimates the 1 DoF joint axes of a kinematic chain of two segments. The axes are returned in the
    % local coordinates of the sensors attached to each segment.
    %
    % Equivalent Python function: :func:`qmt.jointAxisEstHingeOlsson`.
    %
    % :param acc1: Nx3 array of angular velocities in m/s^2
    % :param acc2: Nx3 array of angular velocities in m/s^2
    % :param gyr1: Nx3 array of angular velocities in rad/s
    % :param gyr2: Nx3 array of angular velocities in rad/s
    % :param estSettings: Dictionary containing settings for estimation. If no settings are given, the default settings will be used. Available options:
    %
    %     - **w0**: Parameter that sets the relative weighting of gyroscope to accelerometer residual, w0 = wg/wa. Default value: 50.
    %     - **useSampleSelection**: Boolean flag to use sample selection. Default value: False.
    %     - **dataSize**: Maximum number of samples that will be kept after sample selection. Default value: 1000.
    %     - **winSize**: Window size for computing the average angular rate energy, should be an odd integer. Default value: 21.
    %     - **angRateEnergyThreshold**: Theshold for the angular rate energy.  Default value: 1.
    %     - **x0**: Initial estimate value. Default value: [0, 0, 0, 0].
    %     - **wa**: Accelerometer residual weight.
    %     - **wg**: Gyroscope residual weight
    %
    % :return:
    %     - **jhat1**: 33x1 array, estimated joint axis in imu1 frame in cartesian coordinate.
    %     - **jhat2**: 3x1 array, estimated joint axis in imu2 frame in cartesian coordinate.
    %     - debug: dict with debug values.

    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'lib', 'JointAxisEstHingeOlsson'))

    if nargin<5 || isempty(estSettings)
        estSettings = struct.empty();
    end

    if class(estSettings)== 'containers.Map'
        settingStruct = struct();
        for k = keys(estSettings)
            settingStruct.(k{1}) = estSettings(k{1});
        end
        estSettings = settingStruct;
    end

    useSampleSelection = 0;
    dataSize = 2000;
    winSize = 21;
    angRateEnergyThreshold = 1;
    w0=50;

    if isfield(estSettings,'useSampleSelection')
        useSampleSelection = estSettings.useSampleSelection;
    end
    if isfield(estSettings,'dataSize')
        dataSize = estSettings.dataSize;
    end
    if isfield(estSettings,'winSize')
        winSize = estSettings.winSize;
    end
    if isfield(estSettings,'angRateEnergyThreshold')
        angRateEnergyThreshold = estSettings.angRateEnergyThreshold;
    end
    if isfield(estSettings,'wa')
        wa = estSettings.wa;
    end
    if isfield(estSettings,'wg')
        wg = estSettings.wg;
    end
    if isfield(estSettings,'w0')
        w0 = estSettings.w0;
    end
    if size(acc1, 2) == 3
        acc1 = acc1';
    end
    if size(acc2, 2) == 3
        acc2 = acc2';
    end
    if size(gyr1, 2) == 3
        gyr1 = gyr1';
    end
    if size(gyr2, 2) == 3
        gyr2 = gyr2';
    end
    acc = [acc1;acc2];
    gyr = [gyr1;gyr2];

    %% Sample selection
    if useSampleSelection
        sampleSelectionVars.dataSize = dataSize;
        sampleSelectionVars.winSize = winSize;
        sampleSelectionVars.angRateEnergyThreshold = angRateEnergyThreshold;
        sampleSelectionVars.deltaGyr = [];
        sampleSelectionVars.gyrSamples = [];
        sampleSelectionVars.accSamples = [];
        sampleSelectionVars.accScore = [];
        sampleSelectionVars.angRateEnergy = [];
        [gyr, acc, sampleSelectionVars] = jointAxisSampleSelection([], [], gyr, acc, 1:length(acc), sampleSelectionVars);
        debug.sampleSelectionVars = sampleSelectionVars;
        debug.gyr = gyr;
        debug.acc = acc;
    end

    if (~isfield(estSettings, 'wa') || ~isfield(estSettings, 'wg')) && ~isfield(estSettings,'w0')
        estSettings.w0 = 50;
        estSettings.wg = sqrt(setting.w0);
        estSettings.wa = 1/sqrt(setting.w0);
    end

    if ~isfield(estSettings,'x0')
        estSettings.('x0') = [0 0 0 0]';
    end

    imu1 = struct('acc',acc(1:3,:),'gyr',gyr(1:3,:));
    imu2 = struct('acc',acc(4:6,:),'gyr',gyr(4:6,:));

    [jhat,xhat,optimVarsAxis] = jointAxisIdent(imu1,imu2,estSettings);

    jhat1 = jhat(1:3);
    jhat2 = jhat(4:6);
    debug.xhat = xhat;
    debug.optimVarsAxis = optimVarsAxis;

end
