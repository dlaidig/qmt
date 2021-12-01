function [nhat,xhat,optimVars] = jointAxisIdent(imu1,imu2,settings)
%% Plane of movement identification
% DESCRIPTION:
% Identify the joint axis

%% Initialzie
if ~isfield(imu1,'acc') && ~isfield(imu1,'gyr')
    error('Both acc and gyr are missing from first IMU.')
end

% Use default settings if no settings struct is provided
residuals = [1 2]; % Active residuals
weights.wa = 1;
weights.wg = 1;
loss = @(e) lossFunctions(e,'squared');
optOptions = optimOptions(); % Optimization options
%x0 = -pi + 2*pi*rand(4,1); % Initialize as uniformly random unit vectors
x0 = zeros(4, 1); % Initialize as uniformly random unit vectors
if nargin > 2
    if isfield(settings,'residuals')
        residuals = settings.residuals;
    end
    if isfield(settings,'loss')
        loss = settings.loss;
    end
    if isfield(settings,'optOptions')
        optOptions = settings.optOptions;
    end
    if isfield(settings,'x0')
        x0 = settings.x0;
    end
    if isfield(settings,'weights')
        weights = settings.weights;
    end
    if isfield(settings,'wa')
        weights.wa=settings.wa;
    end
    if isfield(settings,'wg')
        weights.wg=settings.wg;
    end
end

%% Optimization
% Define cost function
costFunc = @(x) jointAxisCost(x,imu1,imu2,residuals,weights,loss);
[xhat1,optimVars1] = optimGaussNewton(x0,costFunc,optOptions);

% Convert from spherical coordinates to unit vectors
nhat = [[cos(xhat1(1))*cos(xhat1(2)) cos(xhat1(1))*sin(xhat1(2)) sin(xhat1(1))]'; ...
        [cos(xhat1(3))*cos(xhat1(4)) cos(xhat1(3))*sin(xhat1(4)) sin(xhat1(3))]'];

xhat1 = [vector2spherical(nhat(1:3));vector2spherical(nhat(4:6))];
x0 = [vector2spherical(nhat(1:3));vector2spherical(-nhat(4:6))];

[xhat2,optimVars2] = optimGaussNewton(x0,costFunc,optOptions);

if optimVars2.f < optimVars1.f
    xhat = xhat2;
    optimVars = optimVars2;
else
    xhat = xhat1;
    optimVars = optimVars1;
end

nhat = [[cos(xhat(1))*cos(xhat(2)) cos(xhat(1))*sin(xhat(2)) sin(xhat(1))]'; ...
        [cos(xhat(3))*cos(xhat(4)) cos(xhat(3))*sin(xhat(4)) sin(xhat(3))]'];
xhat = [vector2spherical(nhat(1:3));vector2spherical(nhat(4:6))];
