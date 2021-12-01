% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

%%
function [delta,rating,cost] = estimate_delta_1d(quat1,quat2,joint,delta_start,constraint,optimization_steps)

j = joint; % the joint is only one axis, so assign it to j
delta = delta_start;

%% Delta estimation
for i = 1:optimization_steps
    [delta_inc,cost]  = gaussNewton_step(quat1,quat2,j,j,delta,constraint);
    delta               = delta + delta_inc;
end
delta = wrapTo2Pi(delta);

%% Rating calculation
v1 = quaternionRotate(quat1,j);
v2 = quaternionRotate(quat2,j);
weight = sqrt(v1(:,1).^2 + v1(:,2).^2) .* sqrt(v2(:,1).^2 + v2(:,2).^2);
rating = rms(weight);
end


%% gaussNewton_step
function [deltaParams,totalError] = gaussNewton_step(quat1,quat2,j_b1,j_b2,delta,constraint)
[errors,jacobi] = errorAndJac_1D(quat1,quat2,j_b1,j_b2,delta,constraint);
deltaParams = -(mldivide((jacobi'*jacobi),jacobi'*errors));
totalError = norm(errors);
end

