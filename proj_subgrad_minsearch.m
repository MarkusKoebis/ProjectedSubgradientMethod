function [xend, stats] = proj_subgrad_minsearch(f, C, x0, options)
%
% INPUT: 
% - f ... structure describing the objective function with information
%   regarding a) the objectives themselves
%   b) the subgradient (depending on options a way to compute them)
%   c) the Lipschitz constant
% - C ... description of the constraint set 
%         To this point: Set of linear inequalities (like for fmincon/linprog)
%           @MAYBE: Widen the range
%
% - x0 ... initial iteration in decision space
%
% - options ... struct with options for the algorithm
%  fields: a) 'opt_goal' ... string 'weak'|'strong'
%          b) 'maxit' ... integer: maximum number of iterations
%          c) 'tol_step' ... double: error tolerance/ minimum step length 
%          d) gamma ... function int -> double in (0,1)/ string
%          e) alpha ... function int -> double (converging quickly to zero)
%                       / string
%          f) opt_routine ... name of optimization routine for internal
%                problems
%          g) optim_options ... structure to define internal optimization
%                options
%          h) llambda ... function (int, int, [int]) -> [double] or string
%          i) index_set_I ... function/string
%
%
%
% Step 0: We assume tat the given initial point is feasible
% @MAYBE: Test feasibility
k  = 0;
x  = x0;
f0 = get_objective(f, x0, options);
m  = size(f0, 1);
vk = Inf;

while k < options.maxit
    % Step 1: _____________________________________________________________
    %fprintf('k = %d\n', k);
    s = get_subgradient_directions(f, x, options);
    % @QUESTION: Can we somehow check for validity w.r.t. Lipschitz constant L?
    %
    %[~, stop_alg] = step_criterion(s, options);
    [~, stop_alg] = step_criterion(vk, options); % @TODO: Why is there s_k in the algorithm?
    % @MAYBE: Monitor error sizes?
    if stop_alg
        break
    end
    %
    Ik = wrapper_get_index_set_I(m, k, options);
    %
    eta = max([vecnorm(s,2,2); 1]);
    %
    llambda = wrapper_lambda_generator(m, k, Ik, options);
    %
    alpha_k = wrapper_alpha_generator(k, options);
    %
    vk = argmin_h(x, s, llambda, C, eta, alpha_k, Ik, options);
    %
    % STEP 2: _____________________________________________________________
    gamma_k = wrapper_gamma_generator(k, options);
    x       = x + gamma_k*vk;
    %
    k = k + 1;
end % main loop 

if k == options.maxit
    warning('Maximum number of iterations exceeded')
end
xend = x;
% @TODO: More information on the algorthim (e.g. cummulated internal statistics, tolerances etc.)
stats = struct('iter', k);




%% AUXILIARY FUNCTIONS
function s = get_subgradient_directions(f, x, options)
s = f(x, 'subgradient', options); % @TODO: We should actually choose one subgradient out of the set of subgradients (this could maybe be done via "options" in f already)

function f0 = get_objective(f, x, options)
%
f0 = feval(f, x, 'objective', options);


%
function vk = argmin_h(x, s, llambda, C, eta, alpha, Ik, options)
%
% main internal minimization routine: 
% min_{v in C - x} h_x (v)
% where h_x (v) := 0.5*||v||^2 + alpha/eta*(sum_{i in I_k} llambda_i s(:,i))'*v
% and C is described by linear equalities/inequalities (Aeq, beq, A, b)
%
n       = length(x);
linpart = alpha/eta*(s(:,Ik)*llambda);
beq     = C.beq - C.Aeq*x;
b       = C.b   - C.A*x;
%
switch options.opt_routine
    case {'quadprog'}
        % @MAYBE: upper/lower bounds and/or "better" initial guess
        vk = feval(options.opt_routine, ...
                 0.5*eye(n), linpart, C.A, b, C.Aeq, beq, ...
                                [], [], zeros(n,1), options.optim_options);
    otherwise
        error('Unknown optimization routine: %s', options.opt_routine)
end

% WRAPPER for creating index sets I_k %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Ik = wrapper_get_index_set_I(m, k, options)
if isa(options.index_set_I, 'function_handle')
    Ik = options.index_set_I(m, k);
    return
elseif isa(options.index_set_I, 'char')
    switch options.index_set_I
        case 'all'
            % use all directions in image space
            Ik = 1:m; return
        case 'one'
            % go through directions one by one
            Ik = mod(k, m) + 1; return
        case 'random'
            % choose directions in image space at random
            M = ceil(m*rand);
            Ik = ceil(m*rand(1, M)); return
    end
end
error('Unable to create index set I_k. Option not properly set.')


% WRAPPER for creating objective weightss lambda %%%%%%%%%%%%%%%%%%%%%%%%%%
function llambda = wrapper_lambda_generator(m, k, Ik, options)
if isa(options.llambda, 'function_handle')
    llambda = options.lambda_generator(m, k, Ik);
    return
elseif isa(options.llambda, 'char')
    switch options.llambda
        case 'equal'
            % all lambda values are equal all the time
            llambda = ones(length(Ik), 1);
            llambda = llambda/sum(llambda); % normalize to create a convex combination
            return
        case 'random'
            % @CAUTION: We need to ensure A2! Maybe it makes sense to
            % include a 'minimal value' for the lambdas?
            llambda = rand(length(Ik), 1);
            llambda = llambda/sum(llambda); % normalize to create a convex combination
            return
    end
elseif isa(options.llambda, 'double')
    if length(Ik) ~= length(options.llambda)
        error('Given values for weights lambda do not have correct dimension')
    else
        llambda = options.llambda;
        % @MAYBE: Check if really convex combination
    end
end
error('Unable to create vector of weights')


% WRAPPER for creating step lengths in each step %%%%%%%%%%%%%%%%%%%%%%%%%%
function gamma = wrapper_gamma_generator(k, options)
if isa(options.gamma, 'function_handle')
    gamma = options.gamma(k); return
elseif isa(options.gamma, 'char')
    switch options.gamma
        case 'random'
            gamma = rand; return
    end
elseif isa(options.gamma, 'double')
    gamma = options.gamma; return
    % @MAYBE: Check if this has the correct sign/dimension etc.
end
error('Unable to create step length gamma')

% WRAPPER for creating alpha sequence
function alpha = wrapper_alpha_generator(k, options)
if isa(options.alpha, 'function_handle')
    alpha = options.alpha(k); return
elseif isa(options.alpha, 'char')
    switch options.alpha
        case 'harmonic'
            alpha = 1/(k+1); return
            % @TODO: Check whether this is a feasible choice or maybe
            % rescale it
    end
end
error('Unable to create internal factor alpha')


function [err, stop_alg] = step_criterion(s, options)
stop_alg = 0;
s_norms = vecnorm(s);
if strcmp(options.opt_goal,'weak')
    err = max(s_norms);
elseif strcmp(options.opt_goal,'strong')
    err = min(s_norms);
else
    error('Unknown option for opt_goal: %s', options.opt_goal)
end
if err <= options.tol_step
    stop_alg = 1;
end
