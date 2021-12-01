function options = optimOptions(varargin)

% Default options
options.tol = 1e-5; % Minimum tolerance in cost function update
options.maxSteps = 300; % Maximum number of steps allowed
options.alpha = 0.4; % Line search parameter 0 < alpha < 0.5
options.beta = 0.5; % Line search parameter 0 < beta < 1
options.quiet = true; % Quiet printing

if nargin >= 2
    for k = 1:2:nargin
        switch varargin{k}
            case 'tol'
                if isnumeric(varargin{k+1}) && isscalar(varargin{k+1}) && varargin{k+1} > 0
                    options.tol = varargin{k+1};
                else
                    error('Optimization options: tol has to be scalar and > 0.')
                end
            case 'maxSteps'
                if isnumeric(varargin{k+1}) && isscalar(varargin{k+1}) && varargin{k+1} > 1
                    options.maxSteps = round(varargin{k+1});
                else
                    error('Optimization options: maxSteps has to be scalar and > 1.')
                end
            case 'alpha'
                if isnumeric(varargin{k+1}) && isscalar(varargin{k+1}) && varargin{k+1} > 0 && varargin{k+1} < 0.5
                    options.alpha = round(varargin{k+1});
                else
                    error('Optimization options: 0 < alpha < 0.5.')
                end
            case 'beta'
                if isnumeric(varargin{k+1}) && isscalar(varargin{k+1}) && varargin{k+1} > 0 && varargin{k+1} < 1
                    options.beta = round(varargin{k+1});
                else
                    error('Optimization options: 0 < beta < 1.')
                end
            case 'quiet'
                if boolean(varargin{k+1})
                    options.quiet = varargin{k+1};
                else
                    error('Optimization option: quiet not boolean')
                end
        end
    end
end
