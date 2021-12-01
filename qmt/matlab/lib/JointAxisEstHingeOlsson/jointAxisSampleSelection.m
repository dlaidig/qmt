function [gyr,acc,sampleSelectionVars] = jointAxisSampleSelection(gyr,acc,gyrNew,accNew,newIndex,sampleSelectionVars)

M = size(gyrNew,2);
n = sampleSelectionVars.winSize;
N = sampleSelectionVars.dataSize;

if isfield(sampleSelectionVars,'angRateEnergyThreshold')
    angRateEnergyThreshold = sampleSelectionVars.angRateEnergyThreshold;
else
    angRateEnergyThreshold = 1; % Default
end

nans=zeros(M,2);

% Remove irregular new measurements (set to NaN)
for k = 2:M
    if norm(gyrNew(1:3,k) - gyrNew(1:3,k-1)) < 3*eps || norm(gyrNew(4:6,k) - gyrNew(4:6,k-1)) < 3*eps
%         gyrNew(:,k) = ones(3,1)*NaN;
        gyrNew(1:3,k) = ones(3,1)*NaN;
        nans(k,1)=k;
    end
    if norm(accNew(1:3,k) - accNew(1:3,k-1)) < 3*eps || norm(accNew(4:6,k) - accNew(4:6,k-1)) < 3*eps
%         accNew(:,k) = ones(3,1)*NaN;
        accNew(4:6,k) = ones(3,1)*NaN;
        nans(k,2)=k;
    end
end

% Gyr magnitude difference
deltaGyr = zeros(M,1);
deltaGyrFilt = zeros(M,1);
absGyr = zeros(M,1);
for k = 1:M
    g1k = gyrNew(1:3,k);
    g2k = gyrNew(4:6,k);
    deltaGyr(k) = norm(g1k) - norm(g2k);
    absGyr(k) = norm(g1k)+norm(g2k);
end
nn = (n-1)/2;
k1 = nn+1;
k2 = M-nn;
for k = 1:M
    if k >= k1 && k <= k2
        dgk = deltaGyr(k-nn:k+nn);
        adgk = abs(dgk);
        kmin = find(adgk == min(adgk));
        if isempty(kmin)
            deltaGyrFilt(k) = 0;
        else
            deltaGyrFilt(k) = dgk(kmin(1));
        end
        if isnan(deltaGyrFilt(k))
           deltaGyrFilt(k) = 0; 
        end
    else
        deltaGyrFilt(k) = 0;
    end 
end
% deltaGyrFilt = deltaGyrFilt; %.*absGyr;
gyr = [gyr gyrNew];
gyrSamples = [sampleSelectionVars.gyrSamples  newIndex];
deltaGyr = [sampleSelectionVars.deltaGyr; deltaGyrFilt];

% Remove NaN measurements
notNaN = ~isnan(gyr(1,:));
gyr = gyr(:,notNaN);
gyrSamples = gyrSamples(notNaN);
deltaGyr = deltaGyr(notNaN);
notNaN = ~isnan(gyr(4,:));
gyr = gyr(:,notNaN);
gyrSamples = gyrSamples(notNaN);
deltaGyr = deltaGyr(notNaN);

% Pick gyro samples
[deltaGyr,gyrSort] = sort(deltaGyr,'descend');
gyr = gyr(:,gyrSort);
gyrSamples = gyrSamples(gyrSort);
if size(gyr,2) > N
    gyr = [gyr(:,1:N/2) gyr(:,end-N/2+1:end)];
    deltaGyr = [deltaGyr(1:N/2); deltaGyr(end-N/2+1:end)];
    gyrSamples = [gyrSamples(1:N/2) gyrSamples(end-N/2+1:end)];
end

% Detect low angular rate
angRateEnergy = zeros(M,1);
gyr1energy = multinorm(gyrNew(1:3,:)).^2;
gyr2energy = multinorm(gyrNew(4:6,:)).^2;
for k = 1:M
    if k >= k1 && k <= k2
        g1k = gyr1energy(k-nn:k+nn);
        g2k = gyr2energy(k-nn:k+nn);
        angRateEnergy(k) = min([mean(g1k(~isnan(g1k))),mean(g2k(~isnan(g2k)))]);
    else
        angRateEnergy(k) = NaN;
    end
end
acc = [acc accNew];
accSamples = [sampleSelectionVars.accSamples newIndex];
angRateEnergy = [sampleSelectionVars.angRateEnergy; angRateEnergy];

% Remove NaN measurements
notNaN = ~isnan(acc(1,:));
acc = acc(:,notNaN);
accSamples = accSamples(notNaN);
angRateEnergy = angRateEnergy(notNaN);
notNaN = ~isnan(acc(4,:));
acc = acc(:,notNaN);
accSamples = accSamples(notNaN);
angRateEnergy = angRateEnergy(notNaN);
notNaN = ~isnan(angRateEnergy);
acc = acc(:,notNaN);
accSamples = accSamples(notNaN);
angRateEnergy = angRateEnergy(notNaN);

% Compute score and sort
accScore = angRateEnergy;
[~,accSort] = sort(accScore,'ascend');
acc = acc(:,accSort);
accSamples = accSamples(accSort);
accScore = accScore(accSort);
angRateEnergy = angRateEnergy(accSort);

% Remove samples with too high energy
% accRemove = find(angRateEnergy > 1);
% acc(:,accRemove) = [];
% angRateEnergy(accRemove) = [];
% accSamples(accRemove) = [];
if size(acc,2) > N
    accRemove = find(angRateEnergy > angRateEnergyThreshold);
    acc(:,accRemove) = [];
    angRateEnergy(accRemove) = [];
    accSamples(accRemove) = [];
end

% Singular value decomposition
it = 0;
while size(acc,2) > N
    A = [acc(1:3,:)' -acc(4:6,:)'];
    [~,S,V] = svds(A,2);
    s = diag(S);
    nRemove = min([floor(s(2)/s(1)*size(acc,2)),size(acc,2)-N-1])+1;

    v1 = V(:,1);
    Anorm = multinorm(A');
    remove = abs(A*v1)./Anorm;
    remove = find(remove > 0.5);
    if length(remove) > nRemove
        remove = remove(end-nRemove+1:end);
    end
%         remove = remove(end);
    acc(:,remove) = [];
    accSamples(remove) = [];
    accScore(remove) = [];
    angRateEnergy(remove) = [];
    disp('+++++++++++++++++');
    disp(it);
    disp(nRemove);
    disp(floor(s(2)/s(1)*size(acc,2)));
    disp(size(acc,2)-N-1);
    disp(s);
    disp('+++++++++++++++++');
    it = it + 1; 
end

% Save variables
sampleSelectionVars.deltaGyr = deltaGyr;
sampleSelectionVars.gyrSamples = gyrSamples;
sampleSelectionVars.accSamples = accSamples;
sampleSelectionVars.accScore = accScore;
sampleSelectionVars.angRateEnergy = angRateEnergy;
