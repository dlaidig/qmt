% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function quat_out = quatInterp(quats,ind)

N = size(quats,1);
M = length(ind);

ind0 = double(clip(uint32(floor(ind)),1,N));
ind1 = double(clip(uint32(ceil(ind)),1,N));

q0 = quats(ind0,:);
q1 = quats(ind1,:);

q_1_0 = relativeQuaternion(q0,q1);

invert_sign_id = q_1_0(:,1) < 0;
q_1_0(invert_sign_id) = -q_1_0(invert_sign_id);

angle = 2*acos(clip(q_1_0(:,1),-1,1));
axis = q_1_0(:,2:end);

direct_ind = angle < 1e-6;

quat_out = zeros(size(q0,1),size(q0,2));

quat_out(direct_ind,:) = q0(direct_ind,:);

interp_ind = ~direct_ind;
t01 = ind-ind0;

temp = t01.*angle';
q_t_0 = getQuat(temp(interp_ind),axis(interp_ind,:));

quat_out(interp_ind,:) = quaternionMultiply(q0(interp_ind,:),q_t_0);
end