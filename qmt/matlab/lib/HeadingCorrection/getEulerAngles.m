% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [angles] = getEulerAngles(q,convention,intrinsic)

%EULERANGLES Calculates Euler angles from quaternions
%   convention: Euler angle axis sequence as string, e.g. "zyx".
%   intrinsic: boolean, if true, intrinsic Euler angles (e.g. z-y'-x'') are
%   calculated, if false, extrinsic angles (e.g. z-y-x) are calculated.
%
%   Author: Daniel Laidig <laidig@control.tu-berlin.de>

    if ~ischar(convention) || ~all(size(convention) == [1 3])
        error('eulerAngles:invalidConvention', 'Invalid convention "%s"', convention);
    end
    
    if intrinsic
        convention = fliplr(convention);
    end
    
    a = conventionIdentifierToNum(convention(1));
    b = conventionIdentifierToNum(convention(2));
    c = conventionIdentifierToNum(convention(3));
    d = NaN;
    if a == c
        if a ~= 1 && b ~= 1
            d = 1;
        elseif a ~= 2 && b ~= 2
            d = 2;
        else
            d = 3;
        end
    end

    if b == a || b == c
        error('eulerAngles:invalidAxisOrder', 'Invalid axis order');
    end

    % sign factor depending on the axis order
    if b == mod(a, 3) + 1 % cyclic order
        s = 1;
    else % anti-cyclic order
        s = -1;
    end
    
    angles = zeros([size(q, 1) 3]);
    if a == c % proper Euler angles
        angles(:,1) = atan2(q(:,a+1).*q(:,b+1) - s.*q(:,d+1).*q(:,1), ...
            q(:,b+1).*q(:,1) + s.*q(:,a+1).*q(:,d+1));
        angles(:,2) = acos(clip(q(:,1).^2 + q(:,a+1).^2 - q(:,b+1).^2 - q(:,d+1).^2, -1, 1));
        angles(:,3) = atan2(q(:,a+1).*q(:,b+1) + s.*q(:,d+1).*q(:,1), ...
            q(:,b+1).*q(:,1) - s.*q(:,a+1).*q(:,d+1));
    else % Tait-Bryan
        angles(:,1) = atan2(2.*(q(:,a+1).*q(:,1) + s.*q(:,b+1).*q(:,c+1)), ...
            q(:,1).^2 - q(:,a+1).^2 - q(:,b+1).^2 + q(:,c+1).^2);
        angles(:,2) = asin(clip(2.*(q(:,b+1).*q(:,1) - s.*q(:,a+1).*q(:,c+1)), -1, 1));
        angles(:,3) = atan2(2.*(s.*q(:,a+1).*q(:,b+1) + q(:,c+1).*q(:,1)), ...
            q(:,1).^2 + q(:,a+1).^2 - q(:,b+1).^2 - q(:,c+1).^2);
    end

    if intrinsic
        angles = fliplr(angles);
    end
end

function [n]=conventionIdentifierToNum(c)
        if c == 'i' || c == '1' || c == 'x' || c == 'X'
            n = 1;
        elseif c == 'j' || c == '2' || c == 'y' || c == 'Y'
            n = 2;
        elseif c == 'k' || c == '3' || c == 'z' || c == 'Z'
            n = 3;
        else
            error('eulerAngles:invalidConventionIdentifier', 'Invalid convention identifier "%s"', c);
        end
end

function [out]=clip(value, lower, upper)
    out = max(lower, min(value, upper));
end






