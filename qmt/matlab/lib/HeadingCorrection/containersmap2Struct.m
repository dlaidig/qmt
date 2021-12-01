% SPDX-FileCopyrightText: 2021 Bo Yang <b.yang@campus.tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [settingStruct] = containersmap2Struct(setting)
    settingStruct = struct();
     if isa(setting, 'containers.Map')
        for k = keys(setting)
            key = k{1};
            val = setting(k{1});
            if isa(val, 'containers.Map')
                val = containersmap2Struct(val);
                settingStruct.(key) = val;
%            elseif  iscell(val)
%                settingStruct.(key) = cell2array(val)
            elseif isempty(iscell(setting(k{1})))
            else
                settingStruct.(key) = val;
            end
        end
    else
        settingStruct = setting;
    end
end
