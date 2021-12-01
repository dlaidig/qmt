% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function axis = axisFromQuat(quat)

angle = angleFromQuat(quat);

axis = 1/sin(angle/2) * quat(:,2:4);

end