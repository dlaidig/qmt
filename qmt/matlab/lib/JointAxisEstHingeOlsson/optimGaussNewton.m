function [x,optimVars] = optimGaussNewton(x,costFunc,options)
%% Initialize
% Gauss-Newton settings
if nargin < 3
    options = optimOptions();
end
tol = options.tol;
maxSteps = options.maxSteps;
alpha = options.alpha;
beta = options.beta;
quiet = options.quiet;
incMax = 5;

f_prev = 0;
diff = tol+1;
step = 1;
xtraj = zeros(size(x,1),maxSteps+1);
xtraj(:,step) = x;
fMins = zeros(incMax,1);
xMins = zeros(size(x,1),incMax);
cInc = 0;

%% Gauss-Newton optimization
while step < maxSteps && diff > tol
    % Evaluate cost function, Jacobian and residual
    [f,~,e,J,P] = costFunc(x);

    % Save initial parameters and cost function value
    if step == 1
        optimVars.f0 = f;
        optimVars.x0 = x;
    end

    % Backtracking line search
    len = 1; % Initial step size

    %% the original code
    dx = pinv(J)*e; % Search direction

%     %% My assuption
%     dx = pinv(J' * J) * J' * e; % Search direction
    [f_next,~,~] = costFunc(x - len*dx);
    while f_next > f + alpha*len*J*dx
        len = beta*len;
        [f_next,~,~] = costFunc(x - len*dx);
    end

    % Handle increased fval
    if f_next > f_prev && step > 1
        cInc = cInc + 1;
        fMins(cInc) = f_prev;
        xMins(:,cInc) = x;
    end

    % Update
    x = x - len*dx;
    step = step+1;
    if size(x,2) > 1
        x = x(:,1);
    end
    xtraj(:,step) = x;
    if step > 2
        diff = norm(f_prev-f_next);
    end
    f_prev = f_next;

    % Print cost function value
    if ~quiet
        disp(['Gauss-Newton. Step ',num2str(step-1),'. f = ',num2str(f_next),'.'])
    end
    if step > maxSteps && ~quiet
        disp('Gauss-Newton. Maximum iterations reached.')
    elseif diff <= tol && ~quiet
        disp('Gauss-Newton. Cost function update less than tolerance.')
    elseif cInc >= incMax
        if ~quiet
            disp('Gauss-Newton. Cost function increased, picking found minimum.')
        end
        fMins = [fMins; f_next];
        xMins = [xMins x];
        minInd = find(fMins == min(fMins));
        x = xMins(:,minInd);
    end
end
xtraj(:,step:end) = repmat(NaN*ones(size(x,1),1),[1 size(xtraj(:,step:end),2)]);
if ~quiet
    disp(['Gauss-Newton. Stopped after ',num2str(step-1),' iterations.'])
end

% Save optimization variables for analysis
[f,g,e,J] = costFunc(x);
optimVars.xtraj = xtraj;
optimVars.f = f;
optimVars.Hessian = J'*J;
% optimVars.e = e;
optimVars.costFunc = costFunc;
