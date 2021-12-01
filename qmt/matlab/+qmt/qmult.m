% SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function output = qmult(q1, q2)
    % Quaternion multiplication.
    %
    % If two Nx4 matrices are given, they are multiplied row-wise. Alternative one of the inputs can be a single
    % quaternion which is then multiplied to all rows of the other input matrix.
    %
    % At the moment, this function is mostly useful to test the Matlab integration of the qmt toolbox. In Python
    % code, use the equivalent Python function :func:`qmt.qmult`.
    %
    % :param q1: Nx4 or 1x4 quaternion input array
    % :param q2: Nx4 or 1x4 quaternion input array
    % :return:
    %     - output: Nx4 quaternion output array

    output = [];
    output(:,1) = q1(:,1).*q2(:,1) - q1(:,2).*q2(:,2) - q1(:,3).*q2(:,3) - q1(:,4).*q2(:,4);
    output(:,2) = q1(:,1).*q2(:,2) + q1(:,2).*q2(:,1) + q1(:,3).*q2(:,4) - q1(:,4).*q2(:,3);
    output(:,3) = q1(:,1).*q2(:,3) - q1(:,2).*q2(:,4) + q1(:,3).*q2(:,1) + q1(:,4).*q2(:,2);
    output(:,4) = q1(:,1).*q2(:,4) + q1(:,2).*q2(:,3) - q1(:,3).*q2(:,2) + q1(:,4).*q2(:,1);
end
