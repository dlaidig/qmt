% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function data = unwrap_mod(data,mode,start_index)

if(strcmp(mode,'deg'))
    data(1:start_index,:) = wrapTo180(data(1:start_index,:));
    data(start_index:end,:) = unwrap_d(data(start_index:end,:));
else
    data(1:start_index,:) = wrapToPi(data(1:start_index,:));
    data(start_index:end,:) = unwrap(data(start_index:end,:));
end


end