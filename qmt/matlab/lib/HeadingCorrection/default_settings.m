% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function settings = default_settings(varargin)

settings.D2.tauDelta           = 2;
settings.D2.tauBias            = 2;
settings.D2.minWeight          = 0.45;
settings.D2.window_time        = 8;
settings.D2.estimation_rate    = 1;
settings.D2.data_rate          = 5;
settings.D2.alignment          = 'center';
settings.D2.constraint         = 'euler';
settings.D2.assumption         = 'cons';
settings.D2.enable_stillness   = true;
settings.D2.gyro_bias_comp     = true;
settings.D2.rom_constraints    = true;
settings.D2.optimizer_steps    = 3;

settings.D1.tauDelta           = 2;
settings.D1.tauBias            = 2;
settings.D1.minWeight          = 0.45;
settings.D1.window_time        = 5;
settings.D1.estimation_rate    = 1;
settings.D1.data_rate          = 5;
settings.D1.alignment          = 'center';
settings.D1.constraint         = 'euler_1d';
settings.D1.assumption         = 'cons';
settings.D1.enable_stillness   = true;
settings.D1.gyro_bias_comp     = true;
settings.D1.rom_constraints    = true;
settings.D1.optimizer_steps    = 2;

settings.D3.tauDelta           = 2;
settings.D3.tauBias            = 2;
settings.D3.minWeight          = 0;
settings.D3.window_time        = 8;
settings.D3.estimation_rate    = 1;
settings.D3.data_rate          = 5;
settings.D3.alignment          = 'backward';
settings.D3.enable_stillness   = true;
settings.D3.gyro_bias_comp     = true;
settings.D3.rom_constraints    = true;
settings.D3.optimizer_steps    = 2;

settings.D1.stillness_time    = 3;
settings.D2.stillness_time    = 3;
settings.D3.stillness_time    = 3;

settings.D1.stillness_threshold    = deg2rad(4);
settings.D2.stillness_threshold    = deg2rad(4);
settings.D3.stillness_threshold    = deg2rad(4);

settings.D3.delta_range = [0:deg2rad(1):deg2rad(359)];


settings.joints             = {'MCP','PIP','DIP'};
settings.fingers            = {'F1','F2','F3','F4','F5'};


end