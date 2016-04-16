clear all
close all

t = cputime;

% PARAMETERS
beta = .99; %discount factor 
alpha = 0.36;
delta = 0.025;

[PI, PI_star] = transmat;

PI_A = [7/8, 1/8; 1/8, 7/8];
% what are the unconditional transition probabilities for z?


% ASSET VECTOR
k_lo = 0;% lower bound of grid points
k_hi = 50;% upper bound of grid points
num_k = 150;

k = linspace(k_lo, k_hi, num_k); % asset (row) vector

% what is aggregate labor in each state?
agg_L = [0.96, 0.9, 0.96, 0.9]

% grid for aggregate capital
K_min = 30;
K_max = 45;
num_K = 16;
K = linspace(K_min, K_max, num_K);

% exogenous states:
z_grid = [1, 0.01];
A_grid = [1.01, 0.99];
Az_grid = [A_grid(1), z_grid(1)
           A_grid(2), z_grid(1)
           A_grid(1), z_grid(2)
           A_grid(2), z_grid(2)];
       
% draw random numbers for simulation
T_sim = 5500;
A_rand = unifrnd(0, 1, 1, 5501);;
A_high = zeros(1, 5501);
A_high(1, 1) = 1;
for t = 2:5500
    if A_high(1, t-1)==1;
        if A_rand(1, t) <= 7/8
            A_high(1, t) = 1;
        else
            A_high(1, t) = 0;
        end
    else
        if A_rand(1, t) <= 7/8
            A_high(1, t) = 0;
        else
            A_high(1, t) = 1;
        end
    end
end
        
    
% ...

% initial guess for coefficients
coeffs = [0 1
          0 1];

% what are factor prices for each of the states?
intrate =  bsxfun(@times, (alpha)*(agg_L'.^(1-alpha)).*Az_grid(:,1), K.^(alpha-1));

wage =  bsxfun(@times, (1-alpha)*(agg_L'.^(-alpha)).*Az_grid(:,1).*Az_grid(:,2), K.^(alpha));
wage = permute(wage, [3, 4, 1, 2]);
wage =  repmat(wage, [num_k, 1, 1, 1]);

% set up return function: 1st dim -> k, 2nd dim -> kprime, 3rd dim -> K
% 4th dim -> (A,z)
cons = bsxfun(@times, permute((1 + intrate - delta), [3, 4, 1, 2]), k');
cons = cons + wage;
cons = bsxfun(@minus, cons,  k);% consumption for every of the states

ret = log(cons); % current period utility
ret(cons<0) = -Inf;

% for each aggregate state figure out approximated K_prime according to the
% G~K function using current parameter guesses

% v_guess = repmat(permute(vfn([1 1 2 2], :), [3 2 1]), [num_K 1 1]);
coeffs_tol = 1;
vj = 1;
coeffj = 1;
r_2 = 0;

% iterate until r_2 > 0.95
while r_2 < 0.99
    v_guess = zeros(num_K, num_k, 4);
    v_tol = 1;
while v_tol > 0.001
    log_K_prime = repmat(coeffs*[ones(1, num_K); log(K)], [2, 1]);
    K_prime = exp(log_K_prime);
    K_prime(K_prime<30) = 30;
    K_prime(K_prime>45) = 45;
    pp = zeros(4, num_K, num_k);
    for i = 1:4
        for j = 1:num_k
    pp(i, :, j) =interp1(K, squeeze(permute(v_guess(:, j, i), [3, 1, 2])), K_prime(i,:));% set up continuation value
        end
    end
    V_exp = zeros(num_K, num_k, 4);
    for iAz = 1:4
        for iK = 1:num_K
            V_exp(iK, :, iAz) = PI(iAz,:)*squeeze(pp(:,iK,:)); % interpolate value function at Kprime to find the continuation
            % value, and take expectations according to PI
            
        end
    end
    
     val_mat = ret + beta*repmat(permute(V_exp, [4, 2, 3, 1]), [num_k, 1, 1, 1]);% write out r.h.s. of value fn, take max, find pol_indx, update val fn
     [vfn pol_indx] = max(val_mat, [], 2);
     vfn = squeeze(permute(vfn, [4, 1, 3, 2]));
     v_tol = abs(max(max(max(v_guess(:) - vfn(:)))));
     v_guess = vfn;% guess
     v_tol;
     vj = vj+1;
    % ...
    
end

pol_indx = permute(pol_indx, [3 1 4 2]);
pol_fn = k(pol_indx);


plot(k, pol_fn(:, :, 8))
plot(k, permute(vfn(8, :, :), [3, 2, 1]))
legend('z=1', 'z=2', 'z=3', 'z=4')



K_series = zeros(T_sim, 1);
K_series_p = zeros(T_sim, 1);
Mu = (1/(2*num_k))*ones(2, num_k);

for t = 1:5500
    t;
    K_agg = sum(Mu,1) * k';
    if K_agg<30
        K_agg = 30;
    elseif K_agg>45
        K_agg = 45;
    end
    K_series(t) = K_agg;
    
    if A_high(t)==1;
    log_K_series_p = coeffs(1,:)*[1, log(K_agg)]';
    K_series_p(t) = exp(log_K_series_p);
    else
    log_K_series_p = coeffs(2,:)*[1, log(K_agg)]';
    K_series_p(t) = exp(log_K_series_p);    
    end
    
    % interpolate the policy function according to the current K_agg
    
    
    % update distribution according to interpolated policy function
    % how to deal with the odd indices? For example, what if the
    % interpolated policy index in state (4,10,2) is 35.4? 
    % In that case we would distribute the mass in the distribution that is
    % in Mu(4,10,2) between cells 35 and 36 in MuPrime; in particular we
    % would put 40% into MuPrime(:,35,:) and 60% into MuPrime(:,36,:) (note
    % that the ':' states are K, which is determined by G~K, and the (A,z)
    % exogenous state which is determined by the simulated A shocks and the
    % PI matrix for transitions between z)
    
    
    pol_ind_interp = zeros(4, num_k);
    for i = 1:4
        for j = 1:num_k
            pol_ind_interp(i, j) = interp1(K, squeeze(pol_fn(i, j, :))', K_agg);
        end
    end
   
    
    pol_fn_1 = pol_ind_interp;
    
    if A_high(t)==1;
    pol_fn_1 = [pol_fn_1(1, :); pol_fn_1(3, :)];
    else
    pol_fn_1 = [pol_fn_1(2, :); pol_fn_1(4, :)];
    end
    
    % Interpolate the policy index and share
    pol_indx_1=zeros(2, num_k, 2);
    pol_shr_1=zeros(2, num_k, 2);
for ii = 1:2
     ii;
        for jj = 1:num_k
            jj;
            if pol_fn_1(ii,jj)<0;
                x1 = 2;
            else
                if find(k>pol_fn_1(ii,jj), 1)<150;
                x1=find(k>pol_fn_1(ii,jj), 1);
                else
                x1=150;
                end
            end
            x2=x1-1;
            x3=(pol_fn_1(ii,jj) - k(x2))/(k_hi/(num_k-1));
            pol_indx_1(ii,jj,1)=x2;
            pol_indx_1(ii,jj,2)=x1;
            pol_shr_1(ii,jj,1)=x3;
            pol_shr_1(ii,jj,2)=(1-x3);
        end
end
 

 % Compute the 2 by 2 transitional matrix of the two values of z
 PI_small = zeros(2, 2);
 
 if A_high(t)==1&A_high(t+1)==1;
     PI_small(1, 1) = PI(1,1)/(PI(1,1)+PI(1,3));
     PI_small(1, 2) = PI(1,3)/(PI(1,1)+PI(1,3));
     PI_small(2, 1) = PI(3,1)/(PI(3,1)+PI(3,3));
     PI_small(2, 2) = PI(3,3)/(PI(3,1)+PI(3,3));
 elseif A_high(t)==1&A_high(t+1)==0;
     PI_small(1, 1) = PI(1,2)/(PI(1,2)+PI(1,4));
     PI_small(1, 2) = PI(1,4)/(PI(1,2)+PI(1,4));
     PI_small(2, 1) = PI(3,2)/(PI(3,2)+PI(3,4));
     PI_small(2, 2) = PI(3,4)/(PI(3,2)+PI(3,4));
 elseif A_high(t)==0&A_high(t+1)==1;
     PI_small(1, 1) = PI(2,1)/(PI(2,1)+PI(2,3));
     PI_small(1, 2) = PI(2,3)/(PI(2,1)+PI(2,3));
     PI_small(2, 1) = PI(4,1)/(PI(4,1)+PI(4,3));
     PI_small(2, 2) = PI(4,3)/(PI(4,1)+PI(4,3));
 else A_high(t)==1&A_high(t+1)==0;
     PI_small(1, 1) = PI(2,2)/(PI(2,2)+PI(2,4));
     PI_small(1, 2) = PI(2,4)/(PI(2,2)+PI(2,4));
     PI_small(2, 1) = PI(4,2)/(PI(4,2)+PI(4,4));
     PI_small(2, 2) = PI(4,4)/(PI(4,2)+PI(4,4));
 end
     
 
 % Compute distribution Mu
 T_tilde_1 = zeros(num_k, num_k, 2, 2);
 T_tilde_2 = zeros(num_k, num_k, 2, 2);
for from_k = 1:num_k
     for from_s = 1:2
         T_tilde_1(from_k, pol_indx_1(from_s, from_k, 1), from_s, :) = PI_small(from_s,:);
         T_tilde_2(from_k, pol_indx_1(from_s, from_k, 2), from_s, :) = PI_small(from_s,:);
     end
end

     MuNew = zeros(size(Mu));
     for from_s = 1:2
         for to_s = 1:2
             MuNew(to_s,:) = MuNew(to_s,:) + squeeze(pol_shr_1(from_s, :, 1)).*Mu(from_s,:) * T_tilde_1(:,:,from_s,to_s) + squeeze(pol_shr_1(from_s, :, 2)).*Mu(from_s,:) * T_tilde_2(:,:,from_s,to_s);
         end
     end
     mu_tol = max(max((abs(Mu(:) - MuNew(:)))));
     Mu = MuNew;
    
end

% run a regression Kprime on K to estimate coefficients from the simulated 
% series


K_11 = [A_high(501:5499)'.*ones(1,4999)', A_high(501:5499)'.*log(K_series_p(501:5499, 1)), (1-A_high(501:5499))'.*ones(1,4999)', (1-A_high(501:5499))'.*log(K_series_p(501:5499, 1))];
%K_11 = [A_high(501:999)'.*ones(1,499)', A_high(501:999)'.*log(K_series_p(501:999, 1)), (1-A_high(501:999))'.*ones(1,499)', (1-A_high(501:999))'.*log(K_series_p(501:999, 1))];

coeffs_estimated = inv(K_11'*K_11)*K_11'*log(K_series(502:5500));
rss = sum((log(K_series(502:5500)) - K_11*coeffs_estimated).^2);
tss = sum((log(K_series(502:5500)) - mean(log(K_series(502:5500)))).^2);
r_2 = 1 - rss/tss
coeffs_estimated = reshape(coeffs_estimated, [2,2]);
coeffs_estimated = coeffs_estimated';


% now update coefficients SLOWLY moving towards the estimates
coeffs_new = 0.9 * coeffs + 0.1 * coeffs_estimated;
coeffs_tol = max(max((abs(coeffs_new(:) - coeffs(:)))));

coeffs = coeffs_new
coeffs_tol

coeffj = coeffj+1

end

e = cputime - t;
display (['runtime is ', num2str(e)])

plot(k, Mu)
