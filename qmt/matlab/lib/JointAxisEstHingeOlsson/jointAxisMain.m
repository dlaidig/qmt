%% Settings
w0 = 50; % Parameter that sets the relative weighting of gyroscope to accelerometer residual, w0 = wg/wa

% Sample selection parameters
useSampleSelection = 1; % Boolean flag to use sample selection
dataSize = 1000; % Maximum number of samples that will be kept after sample selection
winSize = 21; % Window size for computing the average angular rate energy, should be an odd integer
angRateEnergyThreshold = 1; % Theshold for the angular rate energy

% Add utility files to path
addpath([pwd,'\Utility\']);

%% Load data
% Data that needs to be loaded
% acc - 6xN, rows 1:3 from sensor 1, rows 4:6 from sensor 2
% gyr - 6xN, rows 1:3 from sensor 1, rows 4:6 from sensor 2
data_corr = load('data_corr.mat');
data_corr = data_corr.data_corr;

acc1 = data_corr.upper_leg_right.acc';
acc2 = data_corr.lower_leg_right.acc';

gyr1 = data_corr.upper_leg_right.gyr';
gyr2 = data_corr.lower_leg_right.gyr';

acc = [acc1; acc2];
gyr = [gyr1; gyr2];

%% Sample selection
if useSampleSelection
    sampleSelectionVars.dataSize = dataSize;
    sampleSelectionVars.winSize = winSize;
    sampleSelectionVars.angRateEnergyThreshold = angRateEnergyThreshold;
    sampleSelectionVars.deltaGyr = [];
    sampleSelectionVars.gyrSamples = [];
    sampleSelectionVars.accSamples = [];
    sampleSelectionVars.accScore = [];
    sampleSelectionVars.angRateEnergy = [];  
    [gyr,acc,sampleSelectionVars] = jointAxisSampleSelection([],[],gyr,acc,1:length(acc),sampleSelectionVars);
end

%% Identification
settings.x0 = [0 0 0 0]'; % Initial estimate
settings.wa = 1/sqrt(w0); % Accelerometer residual weight
settings.wg = sqrt(w0); % Gyroscope residual weight

imu1 = struct('acc',acc(1:3,:),'gyr',gyr(1:3,:));
imu2 = struct('acc',acc(4:6,:),'gyr',gyr(4:6,:));
[jhat,xhat,optimVarsAxis] = jointAxisIdent(imu1,imu2,settings);


% %%
% q_b1s1 = getQuat(acos(dot(jhat1,[0 0 1])), cross([0 0 1]', jhat1)');
% q_b2s2 = getQuat(acos(dot(jhat2,[0 0 1])), cross([0 0 1]', jhat2)');
% 
% % quat1/quat2: orientation of imu1/2
% 
% q_b1 = quaternionMultiply(quat1, q_b1s1);
% q_b2 = quaternionMultiply(quat2, q_b2s2);
% segments = {'foot_left','lower_leg_left','upper_leg_left','foot_right','lower_leg_right','upper_leg_right','hip','lower_back','upper_back','head','hand_left','lower_arm_left','upper_arm_left','hand_right','lower_arm_right','upper_arm_right'};
% for i = 1:length(segments)
%     data_corr.(segments{i}).ori_seg_cali.quat = data_corr.(segments{i}).ori_seg_driftfree.quat;
% end
% 
% data_corr.upper_leg_right.ori_seg_cali.quat = q_b1;
% data_corr.lower_leg_right.ori_seg_cali.quat = q_b2;
% 
% info.sampleSelectionVars = sampleSelectionVars;
% info.optimVarsAxis = sampleSelectionVars;
% 
% 
% save([ 'data_corr_cali', '.mat'],'data_corr');
% save([ 'info', '.mat'],'info');
% disp('done');