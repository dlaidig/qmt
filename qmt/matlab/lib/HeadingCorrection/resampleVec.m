% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = resampleVec(vec,sample_rate_new,sample_rate_old)

ind = [1:sample_rate_old/sample_rate_new:size(vec,1)];
out = vecInterp(vec,ind);

end