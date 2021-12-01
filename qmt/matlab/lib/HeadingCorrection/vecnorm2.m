% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = vecnorm2(vec)
out = cellfun(@norm,num2cell(vec,2));
end