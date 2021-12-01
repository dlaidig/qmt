% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function angles = calculate_joint_angles(q_rel,j)
dof = size(j,1);

if(dof == 1)
    angles = joint_angles_1d(q_rel,j);
elseif(dof == 2)
    angles = joint_angles_2d(q_rel,j);
else
    %warning('not implemented yet');
     angles = joint_angles_3d(q_rel,j);
end


end

function ja = joint_angles_3d(q_rel,j)

conv = conventionFromAxes(j);

ja = getEulerAngles(q_rel,conv,true);

end

function ja = joint_angles_2d(q_rel,j)

x = [1 0 0];
y = [0 1 0];
z = [0 0 1];

j1 = j(1,:)/norm(j(1,:));
j2 = j(2,:)/norm(j(2,:));

q_rot1 = getQuat(acos(dot(x,j1)),cross(x,j1));
q_rot2 = getQuat(acos(dot(z,j2)),cross(z,j2));
q_b1s_b1 = q_rot1;
q_b2s_b2 = q_rot2;
q_b2s_b1s = quaternionMultiply(quaternionMultiply(quaternionInvert(q_b1s_b1),q_rel),q_b2s_b2);
fixed_angles = getEulerAngles(quaternionMultiply(quaternionInvert(q_rot1),q_rot2),'xyz',true);
angles = getEulerAngles(q_b2s_b1s,'xyz',true);

angles = angles-fixed_angles;
ja(:,1) = angles(:,1);
ja(:,2) = angles(:,3);
beta = fixed_angles(:,2);

end

function ja = joint_angles_1d(q_rel,j)
x = [1 0 0];
y = [0 1 0];
z = [0 0 1];

j = j/norm(j);

q_rot1 = getQuat((acos(dot(x,j))),(cross(x,j)));
q_rot2 = getQuat((acos(dot(x,j))),(cross(x,j)));

q_rel_mod = quaternionMultiply(quaternionMultiply(quaternionInvert(q_rot1),q_rel),q_rot2);
angles = getEulerAngles(q_rel_mod,'xyz',true);
ja = angles(:,1);
end