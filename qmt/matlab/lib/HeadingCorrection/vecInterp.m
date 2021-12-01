% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function vec_out = vecInterp(vec,ind)
N = size(vec,1);
ind0 = double(clip(uint32(floor(ind)),1,N));
ind1 = double(clip(uint32(ceil(ind)),1,N));

v0 = vec(ind0,:);
v1 = vec(ind1,:);

v_1_0 = v1 - v0;
t01 = ind-ind0;

vec_out = v0 + t01'.*v_1_0;
end