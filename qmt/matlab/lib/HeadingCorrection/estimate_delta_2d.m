% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [delta,rating,cost] = estimate_delta_2d(quat1,quat2,gyr1,gyr2,time,joint,delta_start,constraint,optimization_steps)

j1 = joint(1,:);
j2 = joint(2,:);
delta = delta_start;

%% Delta estimation
for i = 1:optimization_steps
    [delta_inc,cost]  = gaussNewton_step(quat1,quat2,gyr1,gyr2,time,j1,j2,delta,constraint);
    delta               = delta + delta_inc;
end

if(strcmp(constraint,'euler_lin'))
    delta = delta(1) * time + delta(2);
end

delta = wrapTo2Pi(delta);

%% Rating calculation
j1_global = quaternionRotate(quat1,j1);
j2_global = quaternionRotate(quat2,j2);
rating = rms(sqrt(j1_global(:,1).^2 + j1_global(:,2).^2) .* sqrt(j2_global(:,1).^2 + j2_global(:,2).^2));
end

%% gaussNewton_step
function [deltaParams,totalError] = gaussNewton_step(quat1,quat2,gyr1,gyr2,time,j1,j2,delta,constraint)
[errors,jacobi] = errorAndJac_2D(quat1,quat2,gyr1,gyr2,time,j1,j2,delta,constraint);
deltaParams = -(mldivide((jacobi'*jacobi),jacobi'*errors));
totalError = norm(errors);
end