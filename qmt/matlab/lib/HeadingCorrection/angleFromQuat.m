% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function angle = angleFromQuat(quat)

quat = quat./vecnorm2(quat);
angle = 2*acos(quat(:,1));

end