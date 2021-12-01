function [f,g,e,J,P] = jointAxisCost(x,imu1,imu2,residuals,weights,loss)
%% Initialize
if ~exist('imu2','var')
    imu2 = [];
end

% Load data variables from imus struct
if ~isfield(imu1,'acc') && ~isfield(imu1,'gyr')
    error('Both acc and gyr are missing from first IMU.')
end
if isfield(imu1,'gyr')
    gyr1 = imu1.gyr;
    Ng = size(gyr1,2);
end
if isfield(imu1,'acc')
    acc1 = imu1.acc;
    Na = size(acc1,2);
end
if isfield(imu2,'acc')
    acc2 = imu2.acc;
else
    acc2 = zeros(3,Na);
end
if isfield(imu2,'gyr')
    gyr2 = imu2.gyr;
else
    gyr2 = zeros(3,Ng);
end

% Set active residuals
if ~exist('residuals','var')
    residuals = [1 2];
end

% Select weights
if ~exist('weights','var')
    weights.wg = ones(Na,1);
    weights.wa = ones(Ng,1);
end
if isfield(weights,'wg') && isscalar(weights.wg)
    wg = repmat(weights.wg,[Ng 1]);
else
    wg = weights.wg;
end
if isfield(weights,'wa') && isscalar(weights.wa)
    wa = repmat(weights.wa,[Na 1]);
else
    wa = weights.wa;
end

% Choose square loss function as default unless other is specified by loss
if ~exist('loss','var')
    loss = @(e) lossFunctions(e,'squared');
end

% Initiate result variables
Nr = length(residuals);
N = Ng+Na;
e = zeros(N,1); % Residuals
g = zeros(4,1); % Gradient
f = 0; % Cost function value
J = zeros(N,4); % Jacobian

% Current estimated normal vectors
if length(x) > 4
    n1 = x(1:3);
    n2 = x(4:6);
    x1(1) = asin(n1(3));
    x1(2) = acos(n1(1)/cos(x1(1)));
    x2(1) = asin(n2(3));
    x2(2) = acos(n2(1)/cos(x2(1)));
else
    x1 = x(1:2,1);
    x2  = x(3:4,1);
    n1 = [cos(x1(1))*cos(x1(2)) cos(x1(1))*sin(x1(2)) sin(x1(1))]';
    n2 = [cos(x2(1))*cos(x2(2)) cos(x2(1))*sin(x2(2)) sin(x2(1))]';
end

% Partial derivatives of normal vectors n w.r.t. spherical coordinates x
dn1dx1 = [-sin(x1(1))*cos(x1(2)) -sin(x1(1))*sin(x1(2)) cos(x1(1));
          -cos(x1(1))*sin(x1(2))  cos(x1(1))*cos(x1(2)) 0];
dn2dx2 = [-sin(x2(1))*cos(x2(2)) -sin(x2(1))*sin(x2(2)) cos(x2(1));
          -cos(x2(1))*sin(x2(2))  cos(x2(1))*cos(x2(2)) 0];

%% Evaluate cost function and Jacobian
jk = 1;
for j = 1:Nr
    switch residuals(j)
        case 1
%             test_g_l = 0;
%             test_g_dlde = 0;
%             test_gg = 0;
            for k = 1:Ng
                % Current measurements
%               g1/g2   3x1 
                g1 = gyr1(:,k);
                g2 = gyr2(:,k);
%               ng1: 1
                ng1 = norm(cross(g1,n1),2);
                ng2 = norm(cross(g2,n2),2);
                if ng1 == 0 || ng2 == 0 || any(isnan([ng1 ng2]))
                    degdn1 = zeros(3,1);
                    degdn2 = zeros(3,1);
                    eg = 0;
                else
%                     degdn1: 3x1
                    degdn1 = wg(k)*cross(cross(g1,n1),g1)/ng1;
                    degdn2 = -wg(k)*cross(cross(g2,n2),g2)/ng2;
                    eg = ng1 - ng2;
                end
%                 dn1dx1: 2x3 ---> 2x3 x 3x1
%               [2x1' 2x1'] = 1x2 1x2 = [1x4]
                J(jk,:) = [(dn1dx1*degdn1)' (dn2dx2*degdn2)'];
                e(jk) = wg(k)*eg;
                [l,dlde] = loss(e(jk));
                f = f + l;
                g = g + dlde*J(jk,:)';
                jk = jk + 1;
%                 test_g_l = test_g_l + l;
%                 test_g_dlde = test_g_dlde + dlde;
%                 test_gg = test_gg + dlde*J(jk,:)';
            end
        case 2
%             test_l = 0;
%             test_dlde = 0;
%             test_g = 0;
            for k = 1:Na
                a1 = acc1(:,k);
                a2 = acc2(:,k);
                ea1 = dot(a1,n1) - dot(a2,n2);
                dea1dn1 = wa(k)*a1;
                dea1dn2 = -wa(k)*a2;
                if any(isnan([a1;a2]))
                    ea1 = 0;
                    dea1dn1 = zeros(3,1);
                    dea1dn2 = zeros(3,1);
                end
                J(jk,:) = [(dn1dx1*dea1dn1)' (dn2dx2*dea1dn2)'];
                e(jk) = wa(k)*ea1;
                [l,dlde] = loss(e(jk));
                f = f + l;
                g = g + dlde*J(jk,:)';
                jk = jk + 1;
                
%                 test_l = test_l + l;
%                 test_dlde = test_dlde + dlde;
%                 test_g = test_g + dlde*J(jk,:)';
            end
    end
end

P = inv(J'*J)*(f/(Na+Ng-4));