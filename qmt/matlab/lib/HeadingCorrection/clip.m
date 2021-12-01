% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = clip(in,min,max)
out = in;
out(out>max) = max;
out(out<min) = min;
end