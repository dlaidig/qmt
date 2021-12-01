% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = unwrap_d(in)

out = rad2deg(unwrap(deg2rad(in)));

end