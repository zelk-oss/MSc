function [x, y, z] = extract_te_variables(driver_series, target_series, m, L)
% EXTRACT_TE_VARIABLES Extract variables for Transfer Entropy computation
%
% Inputs:
%   driver_series - Time series of the driver variable (column vector)
%   target_series - Time series of the target variable (column vector)
%   m - Length of history (number of past time points to include)
%   L - Time lag (number of time points to skip before starting AR model)
%       Default: L = 1 (standard case, use immediately preceding values)
%
% Outputs:
%   x - Present values of driver variable
%   y - Present values of target variable
%   z - Past values of target variable (m previous time points, starting L steps back)
%
% For Transfer Entropy computation using I(x,y|z) where:
%   x = driver present
%   y = target present  
%   z = target past (history of length m, starting at lag L)
%
% Example: If L=3 and m=2, to predict time t, we use:
%   - Present: driver(t) and target(t)
%   - Past: target(t-4) and target(t-5) (skipping t-1, t-2, t-3)

    % Input validation
    if nargin < 3
        error('At least three inputs required: driver_series, target_series, m');
    end
    if nargin < 4 || isempty(L)
        L = 1;  % Default lag of 1 (standard case)
    end
    
    % Ensure inputs are column vectors
    if isrow(driver_series)
        driver_series = driver_series';
    end
    if isrow(target_series)
        target_series = target_series';
    end
    
    % Check that both series have the same length
    if length(driver_series) ~= length(target_series)
        error('Driver and target series must have the same length');
    end
    
    % Check that m is valid
    if ~isscalar(m) || ~isnumeric(m)
        error('m must be a numeric scalar');
    end
    if m ~= round(m)
        error('m must be an integer value');
    end
    if m < 1 || m >= length(target_series)
        error('m must be >= 1 and < length of time series');
    end
    
    n = length(target_series);
    
    % Extract present values (from time point (m+L+1) to end)
    % This ensures we have m past values available starting L steps back
    x = driver_series(m+L+1:n);  % Present driver values
    y = target_series(m+L+1:n);  % Present target values
    
    % Extract past values of target variable with lag L
    % z will be a matrix where each row contains m past values
    % starting L time points before the present
    num_samples = n - m - L;
    z = zeros(num_samples, m);
    
    for i = 1:num_samples
        % For sample i at time t=(m+L+i), collect m values starting at t-L-m
        % This means we skip L points (t-1, t-2, ..., t-L) and then take m points
        start_idx = i;  % This corresponds to t-L-m in the original series
        end_idx = i + m - 1;  % This corresponds to t-L-1 in the original series
        z(i, :) = target_series(start_idx:end_idx)';
    end
    
    % Alternative vectorized approach for z (more efficient):
    % z = hankel(target_series(1:m), target_series(m:n-1))';
    
end

% Example usage:
% % Generate sample data
% t = 1:100;
% driver = sin(0.1*t) + 0.1*randn(1,100);
% target = 0.3*driver + 0.7*sin(0.08*t) + 0.1*randn(1,100);
% 
% % Extract variables for TE computation
% m = 3;  % Use 3 time points of history
% [x, y, z] = extract_te_variables(driver, target, m);
% 
% % Now you can use x, y, z in your Transfer Entropy function:
% % te_value = your_mutual_info_function(x, y, z);