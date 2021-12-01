% SPDX-FileCopyrightText: 2021 Dustin Lehmann <dustin.lehmann@tu-berlin.de>
%
% SPDX-License-Identifier: MIT

function [delta,rating,cost] = estimate_delta_3d(quat1,quat2,angle_ranges,convention,delta_range,delta_start)

delta_probability = getPossibleAngles(quat1,quat2,angle_ranges(1,:),angle_ranges(2,:),angle_ranges(3,:),convention,delta_range,'max'); % generate the distribution of values for delta that produce a valid relative orientation
max_val = max(delta_probability); % get the maximum
distribution_last_delta = size(quat1,1)/pi*abs(wrapToPi(delta_range - delta_start));

delta_probability_min = getPossibleAngles(quat1,quat2,angle_ranges(1,:),angle_ranges(2,:),angle_ranges(3,:),convention,delta_range,'min');
distrib_new_min = delta_probability_min + distribution_last_delta;
min_val = min(distrib_new_min); % get the minimum
delta = mean(unwrap(delta_range(distrib_new_min == min_val)));
cost = min_val;

delta = wrapTo2Pi(delta);

%% Rating calculation
[~,std_constraint] = std_prob_angles(delta_range(delta_probability>max_val/2),delta_probability(delta_probability>max_val/2)); % calculate the standard deviation of the distribution in order to quantify the quality of the estimation
rating = map(std_constraint,0,deg2rad(20),1,0); % scale the standard deviation to a range of [0.1]
rating = min(max(rating,0),1); % clip the window rating to values of [0,1]
    
end


%% getPossibleAngles
% this function determines the range of values of delta that produce possible
% relative orientations
function out = getPossibleAngles(quat1,quat2,angle1_range,angle2_range,angle3_range,convention,delta_range,mode)
q_hand = quat1(not(isnan(quat1(:,1))),:);
q_meta = quat2(not(isnan(quat1(:,1))),:);

[samples,~] = size(q_meta);
possible_angles = logical(zeros(samples,length(delta_range))); %#ok<LOGL>
for i = 1:samples
    q_e1e2 = getQuat(delta_range',[0 0 1]);
    q_meta_corr = quaternionMultiply(q_e1e2,q_meta(i,:));
    q_rel = relativeQuaternion(q_hand(i,:),q_meta_corr);
    angles = getEulerAngles(q_rel,convention,true);
    angle_diff = getAngleDiff(angles,angle1_range,angle2_range,angle3_range);
    angle_diff(angle_diff > 0) = 1;
    
    if(strcmp(mode,'min'))
    possible_angles(i,:) = angle_diff;
    else
       possible_angles(i,:) = not(angle_diff); 
    end
end
out = sum(possible_angles,1);
end

%% getAngleDiff
function out = getAngleDiff(angles,angle_1_range,angle_2_range,angle_3_range)

d1 = bsxfun(@minus,angle_1_range,angles(:,1));
out = abs(min(-sign(d1(:,1).*d1(:,2)).*min(abs(d1),[],2),0));

d2 = bsxfun(@minus,angle_2_range,angles(:,2));
out = out + abs(min(-sign(d2(:,1).*d2(:,2)).*min(abs(d2),[],2),0));

d3 = bsxfun(@minus,angle_3_range,angles(:,3));
out = out + abs(min(-sign(d3(:,1).*d3(:,2)).*min(abs(d3),[],2),0));
end

%% map
function out = map(val,in_min,in_max,out_min,out_max)
out = (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
end

%% std_prob_angles
function [mean,std_dev] = std_prob_angles(val,prob)

if(isempty(val) || length(val)<2)
    mean = 0;
    std_dev = 0;
    return;
end
N = length(val);
area = sum(prob) * (val(2) - val(1));
mean = angularMean(val,prob);
temp = 0;
for i = 1:N
    temp = temp + (wrapToPi(val(i)-mean))^2 * prob(i) / area;    
end
temp = temp * (val(2) - val(1));
std_dev = sqrt(temp);

end

function mean = angularMean(angles,probability)
N = length(angles);
mean = atan2(1/N*sum(probability.*sin(angles)),1/N*sum(probability.*cos(angles)));
mean = wrapTo2Pi(mean);
end