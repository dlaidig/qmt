% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [ja,beta] = test_euler(q1,q2,j1,j2)


x = [1 0 0];
y = [0 1 0];
z = [0 0 1];

j1 = j1/norm(j1);
j2 = j2/norm(j2);

q4 = getQuat(acos(dot(x,j1)),cross(x,j1));
q5 = getQuat(acos(dot(z,j2)),cross(z,j2));

q_b2_b1 = quaternionMultiply(quaternionInvert(q1),q2);
q_b1s_b1 = q4;
q_b2s_b2 = q5;
q_b2s_b1s = quaternionMultiply(quaternionMultiply(quaternionInvert(q_b1s_b1),q_b2_b1),q_b2s_b2);
fixed_angles = getEulerAngles(quaternionMultiply(quaternionInvert(q4),q5),'xyz',true);
angles = getEulerAngles(q_b2s_b1s,'xyz',true);

angles = angles-fixed_angles;
ja(1) = angles(1);
ja(2) = angles(3);
beta = fixed_angles(2);


end