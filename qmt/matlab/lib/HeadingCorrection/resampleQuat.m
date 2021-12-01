% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = resampleQuat(quat,sample_rate_new,sample_rate_old)

ind = [1:sample_rate_old/sample_rate_new:size(quat,1)];
out = quatInterp(quat,ind);

end