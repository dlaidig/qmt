% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function out = conventionFromAxes(axes,mode_2d)

if(nargin<2)
    mode_2d = 'end';
else
    mode_2d = 'middle';
end

out = '';

if(size(axes,1) < 3)
    if(size(axes,1) == 1)
        if(isequal(axes,[1 0 0]))
            axes(2,:) = [0 1 0];
        elseif(isequal(axes,[0 1 0]))
            axes(2,:) = [0 0 1];
        elseif(isequal(axes,[0 0 1]))
            axes(2,:) = [1 0 0];
        end
        axes(3,:) = abs(cross(axes(1,:),axes(2,:)));
        
    else
        if(strcmp(mode_2d,'end'))
            axes(3,:) = abs(cross(axes(1,:),axes(2,:)));
        else
            axes(3,:) = axes(2,:);
            axes(2,:) = abs(cross(axes(1,:),axes(3,:)));
        end
    end
end

for i = 1:3
    axis = axes(i,:);
    
    if(isequal(axis,[1 0 0]))
        out = [out 'x'];
    elseif (isequal(axis,[0 1 0]))
        out = [out 'y'];
    elseif (isequal(axis,[0 0 1]))
        out = [out 'z'];
    else
        error('Wrong axis');
    end
end


end