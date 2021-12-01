% SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = cell2array(cellArray)

if ~iscell(cellArray)
    error('Input argument must be a cell array');
end
out = {};
for idx_c = 1:numel(cellArray)
    if iscell(cellArray{idx_c})
        out = [out cell2array(cellArray{idx_c})];
    else
        out = [out cellArray(idx_c)];
    end
end
out = reshape(cell2mat(out), [], 3);
end
