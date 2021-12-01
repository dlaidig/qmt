% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [errors,jacobi] = errorAndJac_2D(q1,q2,gyr1,gyr2,t,j1,j2,delta_start,constraint)

switch constraint
    case 'euler'
        [errors,jacobi] = getJac_euler_cons(delta_start(1),q1,q2,j1,j2);
    case 'euler_lin'
        [errors,jacobi] = getJac_euler_lin(q1,q2,j1,j2,delta_start(1),delta_start(2),t);
    case 'gyro'
        [errors,jacobi] = getJac_gyro_cons(j1,j2,q1,q2,gyr1,gyr2,delta_start);
    case 'combined'
        [errors,jacobi] = getJac_comb_cons(j1,j2,q1,q2,gyr1,gyr2,delta_start);
    otherwise
        error('Wrong constraint');
end
end


%% getJac_euler_cons
function [error,jac] = getJac_euler_cons(delta,q1,q2,j1,j2)
% Calculate the error
q_e2_e1  = getQuat(delta,[0 0 1]);
q_b2_s2 = getQuat(acos(dot([0, 1, 0], j2)), cross([0, 1, 0], j2));
q_b1_s1 = getQuat(acos(dot([0, 0, 1], j1)), cross([0, 0, 1], j1));

q_e1_b1 = quaternionInvert(quaternionMultiply(q1, q_b1_s1));
q_e2_b1 = quaternionMultiply(q_e1_b1,q_e2_e1);
q_s2_b1 = quaternionMultiply(q_e2_b1,q2);
q_b2_b1 = quaternionMultiply(q_s2_b1,q_b2_s2);

arcsin_arg = 2*(q_b2_b1(:,2).*q_b2_b1(:,1) + q_b2_b1(:,3) .* q_b2_b1(:,4));
second_angle = asin(clip(arcsin_arg,-1,1));

error = get2D_static_weight({j1,j2},q1,q2).*(second_angle); 

% Jacobian
q_b2_e2 = quaternionMultiply(q2,q_b2_s2);

d_q3_ba = zeros(1,4);
d_q3_ba(1) = -0.5*sin(delta/2);
d_q3_ba(4) = 0.5*cos(delta/2);

% need normalization?
d_q3_ba = normalize(d_q3_ba);

d_q_b = quaternionMultiply(quaternionMultiply(q_e1_b1,d_q3_ba),q_b2_e2);
d_q = d_q_b;

jac = 2*(d_q(:,2) .* q_b2_b1(:,1) + q_b2_b1(:,2).*d_q(:,1)+d_q(:,3).*q_b2_b1(:,4) + q_b2_b1(:,3) .* d_q(:,4));
jac = jac./sqrt(1 - arcsin_arg.^2);
end

%% getJac_euler_lin
function [error,jacRow] = getJac_euler_lin(q1,q2,j1,j2,a,b,t)
steps = size(t,1);
delta = a*t + b;

% Calculate the error
q_e2_e1  = getQuat(delta,[0 0 1]);
q_b2_s2 = getQuat(acos(dot([0, 1, 0], j2)), cross([0, 1, 0], j2));
q_b1_s1 = getQuat(acos(dot([0, 0, 1], j1)), cross([0, 0, 1], j1));

q_e1_b1 = quaternionInvert(quaternionMultiply(q1, q_b1_s1));

q_e2_b1 = quaternionMultiply(q_e1_b1,q_e2_e1);
q_s2_b1 = quaternionMultiply(q_e2_b1,q2);
q_b2_b1 = quaternionMultiply(q_s2_b1,q_b2_s2);

arcsin_arg = 2*(q_b2_b1(:,2).*q_b2_b1(:,1) + q_b2_b1(:,3) .* q_b2_b1(:,4));
euler_ca = asin(clip(arcsin_arg,-1,1));

error = (euler_ca); 

% Jacobian

q_b2_e2 = quaternionMultiply(q2,q_b2_s2);

d_q3_ba = zeros(steps,4);
d_q3_ba(:,1) = -0.5*sin(delta/2);
d_q3_ba(:,4) = 0.5*cos(delta/2);
d_q_b = quaternionMultiply(quaternionMultiply(q_e1_b1,d_q3_ba),q_b2_e2);
d_q3_ba = d_q3_ba .* t;
d_q_a = quaternionMultiply(quaternionMultiply(q_e1_b1,d_q3_ba),q_b2_e2);

d_q = [d_q_a',d_q_b'];

jacRow = 2*(d_q(2,:)*q_b2_b1(1) + q_b2_b1(2)*d_q(1,:)+d_q(3,:)*q_b2_b1(4) + q_b2_b1(3) * d_q(4,:));
jacRow = jacRow./sqrt(1 - arcsin_arg^2);

end


%% getJac_gyro_cons
function [errors,jac] = getJac_gyro_cons(j1,j2,q1,q2,gyr1,gyr2,delta)
[steps,~] = size(q1);
e_z = zeros(steps,3) + [0 0 1];
quat_e2_e1 = [cos(delta/2) 0 0 sin(delta/2)];
q2_s2_e1 = quaternionMultiply(quat_e2_e1,q2);
j1_e1 = quaternionRotate(q1,j1);
j2_e2 = quaternionRotate(q2,j2);
j2_e1 = quaternionRotate(quat_e2_e1,j2_e2);
w1_e1 = quaternionRotate(q1, gyr1);
w2_e2 = quaternionRotate(q2,gyr2);
w2_e1 = quaternionRotate(quat_e2_e1,w2_e2);

j_n = cross(j1_e1,j2_e1);
w_d = w1_e1 - w2_e1;
errors = dot(w_d,j_n,2);

% Calculate one Row of the Jacobian

dj2_b = -j2_e2 * sin(delta) + cross(e_z,j2_e2,2)*cos(delta) + ...
    e_z .* (dot(e_z,j2_e2,2)*sin(delta));
dwd_b = -(-w2_e2*sin(delta) + cross(e_z,w2_e2,2)*cos(delta) +...
    e_z.*(dot(e_z,w2_e2,2) * sin(delta)));
d_b = dot(dwd_b,cross(j1_e1,j2_e1,2),2) + dot(w_d,cross(j1_e1,dj2_b,2),2);
jac = d_b;
end


%% getJac_comb_cons
function [errors,jac] = getJac_comb_cons(j1,j2,q1,q2,gyr1,gyr2,delta)
disp(size(q1))
[steps,~] = size(q1);
e_z = zeros(steps,3) + [0 0 1];
quat_e2_e1 = [cos(delta/2) 0 0 sin(delta/2)];
q2_s2_e1 = quaternionMultiply(quat_e2_e1,q2);
j1_e1 = quaternionRotate(q1,j1);
j2_e2 = quaternionRotate(q2,j2);
j2_e1 = quaternionRotate(quat_e2_e1,j2_e2);
w1_e1 = quaternionRotate(q1, gyr1);
w2_e2 = quaternionRotate(q2,gyr2);
w2_e1 = quaternionRotate(quat_e2_e1,w2_e2);

%###### 
%%
j_n = cross(j1_e1,j2_e1,2);
%% dim = 2
%%
w_d = w1_e1 - w2_e1;
errors_gyro = dot(w_d,j_n,2);

% Calculate one Row of the Jacobian

dj2_b = -j2_e2 * sin(delta) + cross(e_z,j2_e2,2)*cos(delta) + ...
    e_z .* (dot(e_z,j2_e2,2)*sin(delta));
dwd_b = -(-w2_e2*sin(delta) + cross(e_z,w2_e2,2)*cos(delta) +...
    e_z.*(dot(e_z,w2_e2,2) * sin(delta)));
d_b = dot(dwd_b,cross(j1_e1,j2_e1,2),2) + dot(w_d,cross(j1_e1,dj2_b,2),2);
jac_gyro = d_b;


% Calculate the error
q_e2_e1  = getQuat(delta,[0 0 1]);
q_b2_s2 = getQuat(acos(dot([0, 1, 0], j2)), cross([0, 1, 0], j2));
q_b1_s1 = getQuat(acos(dot([0, 0, 1], j1)), cross([0, 0, 1], j1));

q_e1_b1 = quaternionInvert(quaternionMultiply(q1, q_b1_s1));
q_e2_b1 = quaternionMultiply(q_e1_b1,q_e2_e1);
q_s2_b1 = quaternionMultiply(q_e2_b1,q2);
q_b2_b1 = quaternionMultiply(q_s2_b1,q_b2_s2);

arcsin_arg = 2*(q_b2_b1(:,2).*q_b2_b1(:,1) + q_b2_b1(:,3) .* q_b2_b1(:,4));
second_angle = asin(clip(arcsin_arg,-1,1));

errors_euler = get2D_static_weight({j1,j2},q1,q2).*(second_angle); 

% Jacobian
q_b2_e2 = quaternionMultiply(q2,q_b2_s2);

d_q3_ba = zeros(1,4);
d_q3_ba(1) = -0.5*sin(delta/2);
d_q3_ba(4) = 0.5*cos(delta/2);
%%
% ####################################
d_q3_ba = normalize(d_q3_ba);
%%
d_q_b = quaternionMultiply(quaternionMultiply(q_e1_b1,d_q3_ba),q_b2_e2);
d_q = d_q_b;

jac = 2*(d_q(:,2) .* q_b2_b1(:,1) + q_b2_b1(:,2).*d_q(:,1)+d_q(:,3).*q_b2_b1(:,4) + q_b2_b1(:,3) .* d_q(:,4));
jac_euler = jac./sqrt(1 - arcsin_arg.^2);

errors = errors_gyro + errors_euler;
jac = jac_euler + jac_gyro;

end



%% getJac_gyro_lin
function [error,jacobiRow] = getJac_gyro_lin(t,a,b,j1,j2,q1,q2,w1_s1,w2_s2)
e_z = [0 0 1];
delta = a*t + b;
quat_e2_e1 = [cos(delta/2) 0 0 sin(delta/2)];
q2_s2_e1 = quaternionMultiply(quat_e2_e1,q2);
j1_e1 = quaternionRotate(q1,j1);
j2_e2 = quaternionRotate(q2,j2);
j2_e1 = quaternionRotate(quat_e2_e1,j2_e2);
w1_e1 = quaternionRotate(q1, w1_s1);
w2_e2 = quaternionRotate(q2,w2_s2);
w2_e1 = quaternionRotate(quat_e2_e1,w2_e2);

j_n = cross(j1_e1,j2_e1);
w_d = w1_e1 - w2_e1;
error = dot(w_d,j_n);

% Calculate one Row of the Jacobian

dj2_b = -j2_e2 * sin(delta) + cross(e_z,j2_e2)*cos(delta) + ...
    e_z * (dot(e_z,j2_e2)*sin(delta));
dwd_b = -(-w2_e2*sin(delta) + cross(e_z,w2_e2)*cos(delta) +...
    e_z*(dot(e_z,w2_e2) * sin(delta)));
d_b = dot(dwd_b,cross(j1_e1,j2_e1)) + dot(w_d,cross(j1_e1,dj2_b));

dj2_a = -j2_e2 * t * sin(delta) + cross(e_z,j2_e2) * t * cos(delta) + ...
    e_z * (dot(e_z,j2_e2) * t * sin(delta));
dwd_a = -(-w2_e2 * t * sin(delta) + cross(e_z,w2_e2)*t*cos(delta)+...
    e_z * (dot(e_z,w2_e2) * t * sin(delta)));
d_a = dot(dwd_a,cross(j1_e1,j2_e1)) + dot(w_d,cross(j1_e1,dj2_a));

jacobiRow = [d_a,d_b];

end

function weight = get2D_static_weight(joint_axes,quat1,quat2)
% 2D Static weight is best, when both projections of the rotation axes are
% of length 1

axis1 = joint_axes{1};
axis2 = joint_axes{2};
j1 = quaternionRotate(quat1,axis1);
j2 = quaternionRotate(quat2,axis2);
weight = sqrt(j1(:,1).^2 + j1(:,2).^2) .* sqrt(j2(:,1).^2 + j2(:,2).^2);
end

