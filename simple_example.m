function simple_example
%
optim_options = optimoptions('quadprog', ...
                             'Display', 'none');
%
options = struct('opt_goal', 'strong', ...
                 'maxit', 1e5, ...
                 'tol_step', 1e-7, ...
                 'index_set_I', 'all', ...
                 'gamma', 0.95,...
                 'alpha', 'harmonic', ...
                 'opt_routine', 'quadprog', ...
                 'optim_options', optim_options,...
                 'llambda', 'equal'...
                 );
% 
x0 = 0.9;
f = @simple_2d_model;
C = struct('A', [1;-1], 'b', [1;1], 'Aeq', zeros(0,1),'beq', []);

% CALL THE ALGORITHM

[xend, stats] = proj_subgrad_minsearch(f, C, x0, options)




%% AUXILIARY FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = simple_2d_model(x, flag, options)

switch flag
    case 'objective'
        out = [x.^5/5; x.^3/3];
    case 'subgradient'
        out = [1, 1];
    otherwise
        error('Unknown flag for objective function')
end


%{
function ll = lambda_generator(m, k, Ik)
% create weighting factors for the objective based on 
% - the total number of objectives m
% - the current iteration count k
% - the set of actively used directions in the image space Ik
% WE NEED TO ENSURE A2!! That is important for the lambda-values
%
ll = ones(length(Ik),1);
ll = ll/sum(ll); % normalize to create a convex combination

function gamma = gamma_generator(k)
gamma = 0.95;
%}
















