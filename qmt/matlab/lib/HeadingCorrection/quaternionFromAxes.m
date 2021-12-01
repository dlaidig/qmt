% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function quat = quaternionFromAxes(x,y,z,exact_axis)

norm_x = vecnorm(x);
norm_y = vecnorm(y);
norm_z = vecnorm(z);

axes = [x;y;z];
norms = vecnorm2(axes);
if(not(sum(norms == 0)==1))
    error('Exactly one axis has to be 0');
end

zero_axis = find(norms==0,1);
A = [1 2 3];
axis_ind = A(A~=zero_axis);
axis1_ind = axis_ind(1);
axis2_ind = axis_ind(2);

R = axes';

angle = angleBetweenAxes(R(:,mod(zero_axis,3)+1),R(:,mod(zero_axis+1,3)+1))*180/pi;

if(not(isempty(exact_axis)))
   
    A = [1 2 3];
    approx_axis = A(A~=exact_axis & A~= zero_axis);
    R(:,approx_axis) = cross(cross(R(:,exact_axis),R(:,approx_axis)),R(:,exact_axis));
    
else

    axis1_adj = cross(cross(R(:,axis2_ind),R(:,axis1_ind)),R(:,axis2_ind));
    axis2_adj = cross(cross(R(:,axis1_ind),R(:,axis2_ind)),R(:,axis1_ind));
    
    R(:,axis1_ind) = 0.5*(normalize(R(:,axis1_ind))+normalize(axis1_adj));
    R(:,axis2_ind) = 0.5*(normalize(R(:,axis2_ind))+normalize(axis2_adj));
    
end

R(:,zero_axis) = cross(R(:,mod(zero_axis,3)+1),R(:,mod(zero_axis+1,3)+1));

for i = 1:3
    R(:,i) = normalize(R(:,i));
end
quat = rotMatToQuat(R);

end