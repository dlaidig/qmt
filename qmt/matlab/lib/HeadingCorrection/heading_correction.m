% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [quat2_corr,delta,delta_filt,rating,state_out, debug_data] = heading_correction(data1,data2,joint,joint_info,est_settings, debug)
%% Description
%
% This function corrects the heading of a kinematic chain of two segments
% whose orientations are estimated without the use of magnetometers. It
% corrects the heading of the second segment in a way that its orientation
% is estimated in the reference of the first segment. It uses kinematic
% constraints for rotational joints to estimate the heading offset of both
% segments based on the limited set of possible relative orientations
% induced by the rotational joint.
% There are methods for 1D, 2D and 3D joints, based on different
% constraints.
%
% Outputs:
% quat2_corr: Corrected orientation of the second segment in the reference
%             frame of the first segment
% delta:      value of the estimated heading offset
% delta_filt: value of the filtered heading offset
% rating:     value of the certainty of the estimation
% state:      state of the estimation. 1: regular, 2: startup, 3: stillness
%
% Inputs:
% data1         Structure containing the data of segment 1 containing the
%               following fields:
%               - gyr: Nx3 vector of angular velocities in rad/s
%               - quat: Nx4 vector of orientation quaternions
%               - time: Nx1 vector of the equidistant sampled time signal
% data2         Structure containing the data of segment 2 containing the
%               following fields:
%               - gyr: Nx3 vector of angular velocities in rad/s
%               - quat: Nx4 vector of orientation quaternions
%               - time: Nx1 vector of the equidistant sampled time signal
% joint         dofx3 array containing the axes of the joint
% joint_info    structure containing additional joint information. only
%               needed for 3D-joints. Fields:
%               - convention: String of the chosen euler angles convention,
%                 e.g. 'xyz'
%               - angle_ranges: 3x2 array with the minimum and maximum
%                 value for each joint angle in radians
% est_settings  structure containing the needed settings for the
%               estimation. Fields:
%               - window_time: width of the window over which the
%               estimation should be performed in seconds
%               - estimation_rate: rate in Hz at which the estimation should be
%               performed. typically below 1Hz is sufficient
%               - data_rate: rate at which new data is fed into the
%               estimation. Is used to downsample the data for a faster
%               estimation. Typically values around 5Hz are sufficient
%               - alignment: String of the chosen type of alignment of the
%               estimation window. It describes how the samples around the
%               current sample at which the estimation takes place are
%               chosen. Possible values: 'backward', 'center', 'forward'
%               - enable_stillness: boolean value if the stillness
%               detection should be enabled
%               - stillness_time: time in seconds which should pass for the
%               stillness detection to be triggered
%               - stillness_threshold: threshold for the angular velocity
%               under which the body is assumed to be at rest. In rad/s
%               - tau_delta: time constant for the filter in seconds
%               - tau_bias: time constant for the filter in seconds
%               - delta_range: array of values that are supposed to be
%               tested for the method for 3D joints.
%               - constraint: type of contraint used for the estimation.
%               Constraints for 1D: (proj, euler_1d, euler_2d, icorr).
%               Constraints for 2D: (euler,euler_lin, gyro, combined).
%               Constraints for 3D: (default)
%               - optimizer_steps: Number of Gau?-Newton steps during
%               optimization.
    %% Input checking
    % if no settings are given load a set of standard settings
    if(nargin<5 || isempty(est_settings))
        disp('Default params used for heading correction');
        use_default_params = true;
        est_settings = Struct()
    else
        use_default_params = false;
    end

    % check if both data structures contain all needed fields
    if(not(isfield(data1,'gyr')) && not(isfield(data1,'g')))
        error('Data1 does not contain gyrsoscope data!');
    end
    if(not(isfield(data2,'gyr')) && not(isfield(data2,'g')))
        error('Data2 does not contain gyrsoscope data!');
    end
    if(not(isfield(data1,'quat')))
        error('Data1 does not contain orientation (quat)!');
    end
    if(not(isfield(data2,'quat')))
        error('Data2 does not contain orientation (quat)!');
    end
    if(not(isfield(data1,'time')))
        error('Data1 does not contain time signal (time)!');
    end
    if(not(isfield(data2,'time')))
        error('Data2 does not contain time signal (time)!');
    end

    %% Parameter loading

    % Determine the degrees of freedom of the joint, this determines the type
    % of estimation that will be used

    dof = size(joint,1);
    if(dof > size(joint,2))
        error('Wrong dimension of joint matrix');
    end
    % Estimation parameters
    window_time = 8;
    estimation_rate = 1;
    data_rate = 5;
    tau_delta = 5;
    tau_bias = 5;
    rating_min = 0.4;
    alignment = 'backward';
    enable_stillness = true;
    optimizer_steps = 5;
    stillness_time = 3;
    stillness_threshold = deg2rad(4);
    delta_range = [0:deg2rad(1):deg2rad(359)];

    switch dof
        case 1
            constraint = 'euler_1d';
        case 2
            constraint = 'euler';
        case 3
            constraint = 'default';
    end

    if(~use_default_params)
        if isfield(est_settings, 'windowTime')
            window_time = est_settings.windowTime;
        end
        if isfield(est_settings, 'estimationRate')
            estimation_rate = est_settings.estimationRate;
        end
        if isfield(est_settings, 'dataRate')
            data_rate = est_settings.dataRate;
        end
        if isfield(est_settings, 'tauDelta')
             tau_delta = est_settings.tauDelta;
        end
        if isfield(est_settings, 'tauBias')
            tau_bias = est_settings.tauBias;
        end
        if isfield(est_settings, 'ratingMin')
            rating_min = est_settings.ratingMin;
        end
        if isfield(est_settings, 'alignment')
            alignment = est_settings.alignment;
        end
        if isfield(est_settings, 'enableStillness')
            enable_stillness = est_settings.enableStillness;
        end
        if isfield(est_settings, 'optimizerSteps')
            optimizer_steps = est_settings.optimizerSteps;
        end
        if isfield(est_settings, 'stillnessTime')
            stillness_time = est_settings.stillnessTime;
        end
        if isfield(est_settings, 'stillnessThreshold')
            stillness_threshold = est_settings.stillnessThreshold;
        end
        if isfield(est_settings, 'deltaRange')
            delta_range = est_settings.deltaRange;
        end
        if isfield(est_settings, 'constraint')
            constraint = est_settings.constraint;
        end
    end

    %% Preprocessing
    % Determine data rate from one of the time signals. Assumption: constant
    % rate
    dt = data1.time(2) - data1.time(1);
    rate = 1/dt;

    % Load the joint info data for three-dimensional joints
    if(dof == 3)
        angle_ranges = joint_info.angle_ranges;
        convention = joint_info.convention;
    end
    debug_data=struct();

    % Calculate the window steps (Nw), estimation steps (Ne) und data steps
    % from the settings and the actual time series
    window_steps = window_time * rate;
    estimation_steps = 1/estimation_rate * rate;
    data_steps = 1/data_rate * rate;
    stillness_steps = stillness_time * rate;

    window_steps = round(window_steps);
    estimation_steps = round(estimation_steps);
    data_steps = round(data_steps);
    stillness_steps = round(stillness_steps);


    %
    N = length(data1.time); % number of total time steps in the time series

    % Calculate the time instants at which an estimation is performed. This
    % depends on the number of steps in the window, the rate at which
    % estimations are performed and the aligment of the window
    switch alignment
        case 'backward'
            starts = window_steps+1:estimation_steps:(N-1);
        case 'center'
            starts = window_steps/2+1:estimation_steps:(N-window_steps/2-1);
        case 'forward'
            starts = 2:estimation_steps:(N-window_steps);
        otherwise
            error('Wrong alignment type');
    end

    starts = round(starts);

    % to have a smoother start up, add estimations at each data step in the
    % beginning until a complete time window has passed
    regular_start = starts(1); % start of the regular estimation without smooth startup
    starts = [data_steps+1:data_steps:(starts(1)-data_steps),starts];
    estimations = length(starts); % number of performed estimations

    % Initialize the result vectors
    delta = zeros(estimations,1); % delta is the heading offset between the first and second segment
    rating = zeros(estimations,1); %the rating indicates the quality of the estimation
    state_out = zeros(estimations,1);

    % Initialize the filtering time constants
    tau_delta = ones(estimations,1)*tau_delta;
    tau_bias = ones(estimations,1)*tau_bias;

    % initialize state variable
    %% Estimation
    for k = 2:estimations
        % set default values to variables
        stillness_trigger = false;
        state = 'none';

        % get the current index in the data
        index = starts(k);

        % check whether the smooth startup is active. If so, the used data
        % for the estimation is from index 1 to the current index
        if index < regular_start
            state = 'startup';
            index_start = 1;
            index_end = index;
        else % during regular estimation, the start and end index are determined based on the current index and the chosen aligment type
            switch alignment
                case 'center'
                    index_start = index - window_steps/2;
                    index_end = index +window_steps/2;
                case 'forward'
                    index_start = index;
                    index_end = index + window_steps;
                case 'backward'
                    index_start = index-window_steps;
                    index_end = index;
            end
        end

        % Stillness detection
        % Check if the two segments are at rest. If they are at rest, the
        % current value of delta can be calculated from the last estimated
        % relative orientation. This ensures that both segments do not move in
        % a resting phase. However, this does not ensure correct estimation
        % since it is only based on the last estimated orientation

        % only do the stillness detection of the setting is set to true and do
        % not do stillness detection during startup
        if(enable_stillness && not(strcmp(state,'startup')))
            % check whether the number of passed samples is larger than the
            % detection period for the stillness detection
            if(index>stillness_steps)
                stillness = check_stillness(data1.gyr(index-stillness_steps:index,:),data2.gyr(index-stillness_steps:index,:),stillness_threshold);
                if(stillness)
                    if(state_out(k-1) == 1)  % regular
                        stillness_trigger = true;
                    end
                    state = 'stillness';
                else
                    state = 'regular';
                end
            else
                state = 'regular';
            end
        else
            if(not(strcmp(state,'startup')))
                state = 'regular';
            end
        end
        % Stillness correction
        if(strcmp(state,'stillness'))
            delta(k) = stillness_correction(data1.quat(index,:),data2.quat(index,:),delta(k-1),stillness_trigger);
        end


        %% Estimation
        % the estimation will only be performed in startup and regular state,
        % not in stillness state

        if(strcmp(state,'startup') || strcmp(state,'regular'))

            % for convenience extract the necessary data windows from both
            % segments
            quat1 = data1.quat(index_start:data_steps:index_end,:);
            quat2 = data2.quat(index_start:data_steps:index_end,:);
            gyr1 = data1.gyr(index_start:data_steps:index_end,:);
            gyr2 = data2.gyr(index_start:data_steps:index_end,:);
            time = data1.time(index_start:data_steps:index_end);

            % execute a dedicated estimation algorithm for each type of joint.
            % for more detailed explanation look in the descriptions of the
            % corresponding method
            switch dof
                case 1
                    [delta(k),rating(k),cost(k)] = estimate_delta_1d(quat1,quat2,joint,delta(k-1),constraint,optimizer_steps);
                case 2
                    [delta(k),rating(k),cost(k)] = estimate_delta_2d(quat1,quat2,gyr1,gyr2,time,joint,delta(k-1),constraint,optimizer_steps);
                case 3
                    [delta(k),rating(k),cost(k)] = estimate_delta_3d(quat1,quat2,angle_ranges,convention,delta_range,delta(k-1));
            end

            % during startup estimation a second estimation is performed from a
            % different starting value to ensure that the global minimum is
            % found
            if(strcmp(state,'startup'))
                switch dof
                    case 1
                        [delta_2,~,cost_2] = estimate_delta_1d(quat1,quat2,joint,delta(k-1)+pi,constraint,optimizer_steps);
                    case 2
                        [delta_2,~,cost_2] = estimate_delta_2d(quat1,quat2,gyr1,gyr2,time,joint,delta(k-1)+pi,constraint,optimizer_steps);
                    case 3
                        [delta_2,~,cost_2] = estimate_delta_3d(quat1,quat2,angle_ranges,convention,delta_range,delta(k-1)+pi);
                end
                if(cost_2<cost(k))
                    delta(k) = delta_2;
                end

            end
        end

        % adapt the rating in the two special states startup and stillness and set it to 1 in order
        % to enable fast convergence
        if(strcmp(state,'startup'))
            rating(k) = 1;
            tau_delta(k) = 0.4;
            tau_bias(k) = 1;
        end
        if(strcmp(state,'stillness'))
            rating(k) = 1;
        end

        if(strcmp(state,'regular'))
            state_out(k) = 1;
        elseif(strcmp(state,'startup'))
            state_out(k) = 2;
        elseif(strcmp(state,'stillness'))
            state_out(k) = 3;
        else
            error('Wrong state');
        end

    end


    %% Filtering
    % use a filter to smooth the trajectory of the estimated delta. The filter tries to eliminate phase lag introduced by low pass filtering by estimating the slope of the current trajectory. Furthermore, if
    % rating < rating_min, the filter only extrapolates the estimated slope and
    % dismisses new estimates until rating >= rating_min
    [delta_filt, bias]  = headingFilter(delta,rating,state_out,estimation_rate,tau_delta,tau_bias,rating_min,window_time,alignment);

    if debug
        debug_data.uninterpolated.delta = delta;
        debug_data.uninterpolated.deltaFilt = delta_filt;
        debug_data.uninterpolated.rating = rating;
        debug_data.uninterpolated.stateOut = state_out;
        debug_data.uninterpolated.cost = cost;
    end

    %% Interpolation
    % Interpolate the data to the given time signal
    delta       = interp1(data1.time(starts),unwrap(delta),data1.time,'linear','extrap');
    delta_filt  = interp1(data1.time(starts),delta_filt,data1.time,'linear','extrap');
    rating      = interp1(data1.time(starts),rating,data1.time,'linear', 'extrap');
    state_out   = interp1(data1.time(starts),state_out,data1.time,'linear', 'extrap');
    %% Heading correction
    quat2_corr = quaternionMultiply(getQuat(delta_filt,[0 0 1]),data2.quat);

    if debug
        debug_data.Interpolation.delta = delta;
        debug_data.Interpolation.deltaFilt = delta_filt;
        debug_data.Interpolation.rating = rating;
        debug_data.Interpolation.stateOut = state_out;
        debug_data.bias = bias;
        debug_data.starts=starts;

        debug_data.params.window_time = window_time;
        debug_data.params.estimationRate = estimation_rate;
        debug_data.params.dataRate = data_rate;
        debug_data.params.tauDelta = tau_delta;
        debug_data.params.tauBias = tau_bias;
        debug_data.params.ratingMin = rating_min;
        debug_data.params.alignment = alignment;
        debug_data.params.enableStillness = enable_stillness;
        debug_data.params.optimizerSteps = optimizer_steps;
        debug_data.params.stillnessTime = stillness_time;
        debug_data.params.stillnessThreshold = stillness_threshold;
        debug_data.params.deltaRange = delta_range;
        debug_data.params.joint = joint;
        debug_data.params.constraint = constraint;
    end

end


%% check_stillness()
function stillness = check_stillness(gyr1,gyr2,still_threshold)

% check whether the mean of both gyroscope signals is smaller than the
% defined threshold over the complete period
if(mean(vecnorm2(gyr1)) < still_threshold && mean(vecnorm2(gyr2)) < still_threshold)
    stillness = true;
else
    stillness = false;
end
end

%% stillness_correction()
function delta = stillness_correction(quat1,quat2,last_delta,stillness_trigger)

% initialize memory variable
persistent quat_rel_ref;
persistent delta_still;

% set the reference relative orientation which should be held during
% stillness phase. It is reset after each rising edge of the stillness
% detection
if(isempty(quat_rel_ref))
    quat_rel_ref = quaternionMultiply(quaternionMultiply(quaternionInvert(quat1),getQuat(last_delta,[0 0 1])),quat2);
    delta_still = last_delta;
else
    if(stillness_trigger)
        quat_rel_ref = quaternionMultiply(quaternionMultiply(quaternionInvert(quat1),getQuat(last_delta,[0 0 1])),quat2);
        delta_still = last_delta;
    end
end

% calculate the value of delta that would lead to the reference orientation
q_e2_e1 = quaternionMultiply(quaternionMultiply(quat1,quat_rel_ref),quaternionInvert(quat2));
q_rel = relativeQuaternion(getQuat(delta_still,[0 0 1]),q_e2_e1);
delta_inc = 2*atan2(dot(q_rel(2:4),[0,0,1]),q_rel(1));
delta = delta_still+ delta_inc;

end

%% headingFilter
function [delta_out, bias] = headingFilter(delta, rating, state, estimation_rate,tau_bias,tau_delta,minRating,window_time, alignment)

% tau_bias = 8; % tuning parameter: time constant for bias filter %15
% tau_delta = 15; % tuning parameter: time constant for heading filter %30
%minRating = 0.4; % tuning parameter: extrapolating if rating < minRating

delta = unwrap(delta);
N = length(delta);
Ts = 1/estimation_rate;
out = zeros(N, 1);
delta_out = zeros(N, 1);
bias = zeros(N, 1);

window_size = window_time*estimation_rate;

k_bias = 1-exp(-Ts*log(2)./tau_bias);
k_delta = 1-exp(-Ts*log(2)./tau_delta);
if(length(tau_delta)==1)
    k_delta = repmat(k_delta,length(rating),1);
end
if(length(tau_bias)==1)
    k_bias = repmat(k_bias,length(rating),1);
end

rating(rating<minRating) = 0;
out(1) = delta(1);

biasClip = deg2rad(2) * Ts;

j = 0;
for i=2:N
    if(state(i) == 2)  % startup
        kBiasEff = 0.0;
        kDeltaEff = max(rating(i) * k_delta(i), 1 / i);
    else
        kBiasEff = rating(i) * k_bias(i);
        kDeltaEff = max(rating(i) * k_delta(i), 1 / (j + 1));
        j = j + 1;
    end

    deltaDiff = clip(wrapToPi(delta(i) - delta(i-1)), -biasClip, biasClip);
    bias(i) = clip(bias(i-1) + kBiasEff * (deltaDiff - bias(i-1)), -biasClip, biasClip);
    out(i) = out(i-1) + bias(i) + kDeltaEff * (wrapToPi(delta(i) - out(i-1)) - bias(i));

    if strcmp(alignment, 'backward')
        delta_out(i) = out(i) + window_size/2 * bias(i);
    elseif strcmp(alignment, 'forward')
        delta_out(i) = out(i) - window_size/2 * bias(i);
    else  % center
        delta_out(i) = out(i);
    end
end
end
