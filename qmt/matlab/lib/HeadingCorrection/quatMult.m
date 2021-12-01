% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = quatMult(quats)

out = [1 0 0 0];

for i = 1:size(quats,1)
   out = quaternionMultiply(out,quats(i,:));
end

end