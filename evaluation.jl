## evaluation metrics  for Gibbs sampler

#include("ntl_gibbs.jl")

function mean_arrival_time_Lp(T_inferred::Vector{Int},T_truth::Vector{Int},p::Real)
  """
  Returns mean (normalized by number of elements in `T_inferred`) L^`p` norm
    of difference between `T_inferred` and `T_truth`.
  """
  if size(T_inferred,1) != size(T_truth,1)
    error("T_inferred and T_truth must be the same length.")
  end
  K = size(T_inferred,1)
  if isfinite(p)
    d = (1/K)*sum( abs.((T_inferred .- T_truth).^p) ).^(1/p)
  else # L^∞-norm
    d = (1/K)*maximum( abs.(T_inferred .- T_truth) )
  end
  return d
end

function mean_arrival_time_Lp(T_inferred::Array{Int,2},T_truth::Vector{Int},p::Real)
  """
  Returns mean (normalized by number of columns of `T_inferred`) of
    mean (normalized by number of rows of `T_inferred`) L^`p` norm of
    differences between columns of `T_inferred` and `T_truth`.
  """
  s = zero(Float64)
  for j in 1:size(T_inferred,2)
    s += mean_arrival_time_Lp(T_inferred[:,j],T_truth,p)
  end
  return s./size(T_inferred,2)
end

function total_variation_distance(p::Vector{Float64},q::Vector{Float64})
  """
  Returns total variation distance between two discrete probabiliy distributions
    `p` and `q`. Elements of `p` and `q` are assumed to corresponding support points.
    If `size(p,1) != size(q,1)` then the shorter of the two is padded with zeros.
  """
  if abs(1.0 - sum(p)) > eps() || abs(1.0 - sum(q)) > eps()
    error("Elements of `p` or `q` do not sum to 1.")
  end
  np = size(p,1)
  nq = size(q,1)
  if np==nq
    d = 0.5*sum( abs.(p .- q) )
  elseif np > nq
    d = total_variation_distance(p,[q; zeros(Float64,np-nq)])
  else
    d = total_variation_distance([p; zeros(Float64,nq-np)],q)
  end
  return d
end

function total_variation_distance(p::Vector{Int64},q::Vector{Int64})
  """
  Returns total variation distance between two discrete probabiliy distributions
    `p` and `q`. Elements of `p` and `q` are assumed to corresponding support points.
    If `size(p,1) != size(q,1)` then the shorter of the two is padded with zeros.
  """
  # if abs(1.0 - sum(p)) > eps() || abs(1.0 - sum(q)) > eps()
  #   error("Elements of `p` or `q` do not sum to 1.")
  # end
  np = size(p,1)
  nq = size(q,1)
  if np==nq
    d = 0.5*sum( abs.(p .- q) )
  elseif np > nq
    d = total_variation_distance(p,[q; zeros(Int64,np-nq)])
  else
    d = total_variation_distance([p; zeros(Int64,nq-np)],q)
  end
  return d
end

function deviance_information_criterion(PP::Vector{Int},T_gibbs::Array{Float64,2},alpha_gibbs::Vector{Float64})
  """
  Returns DIC based on arrival times and α (marginalizes Ψ_j's)
  """
  T_mean = mean(T_gibbs,2)
  alpha_mean = mean(alpha_gibbs)
  D_param_bar = log_CPPF(PP,T_mean,alpha_mean)

  M = size(T_gibbs,2)
  D_bar = zero(Float64)
  for m in 1:M
    D_bar += log_CPPF(PP,T[:,m],alpha[m])
  end

  return 2*D_bar/M - D_param_bar

end

function predictive_logprob(PP_train::Vector{Int},T_train::Vector{Int},
  PP_train_test::Vector{Int},T_train_test::Vector{Int},
  predictive_ia_dist::DiscreteDistribution,alpha::Float64)

  f = (x,y) -> predictive_ia_dist
  predictive_logprob(PP_train,T_train,PP_train_test,T_train_test,f,alpha)
end

function predictive_logprob(PP_train::Vector{Int},T_train::Vector{Int},
  PP_train_test::Vector{Int},T_train_test::Vector{Int},
  predictive_ia_dist::Function,alpha::Float64)
  """
  Returns predictive log-probability of `PP_train_test` and `T_train_test` given `PP_train` and
    `predictive_ia_dist`, which is a function that evaluates interarrivals in `T_train_test`
    at a specific parameter setting (e.g., a sample from the posterior given the
    training data)

  `PP_train_test` and `T_train_test` should be the stochastic extension (growth) of `PP_train` and `T_train`,
    i.e., when `PP_train` has `K_train` elements, then the first `K_train` elements
    of `PP_train_test` correspond to those same elements and `PP_train_test[k] >= PP_train[k]` should
    be true for all `k <= K_train`. Similarly, `T_train_test` shoud be an extension of `T_train`.
  """
  logp_train = logp_partition(PP_train,T_train,alpha,predictive_ia_dist,false)
  logp_test_train = logp_partition(PP_train_test,T_train_test,alpha,predictive_ia_dist,false)
  return logp_test_train - logp_train
end

function predictive_logprob(logp_train::Float64,
  PP_train_test::Vector{Int},T_train_test::Vector{Int},
  predictive_ia_dist::DiscreteDistribution,alpha::Float64)
  """
  Returns predictive log-probability when log-probability of training data
    (`logp_train`) has already been calculated.
  """

  f = (x,y) -> predictive_ia_dist
  predictive_logprob(logp_train,PP_train_test,T_train_test,f,alpha)
end

function predictive_logprob(logp_train::Float64,
  PP_train_test::Vector{Int},T_train_test::Vector{Int},
  predictive_ia_dist::Function,alpha::Float64)
  """
  Returns predictive log-probability when log-probability of training data
    (`logp_train`) has already been calculated.
  """

  logp_test_train = logp_partition(PP_train_test,T_train_test,alpha,predictive_ia_dist,true)
  return logp_test_train - logp_train
end


function sample_predicted_arrival_times(ia_dist::Function,T_end::Int,
  K_end::Int,n_end::Int,n_preds::Int)
  """
  Samples predicted interarrival times from `ia_dist` and adds them to `T_end`,
    which corresponds to the `K_end`-th arrival time, until `n_end + n_preds` is exceeded.
    The first sampled arrival time is forced to be greater than `n_end`.
  """
  zero_shift = 1 - minimum(ia_dist(1,1))
  K_start = K_end
  # sample first arrival
  new_arr = T_end + rand(ia_dist(T_end,K_start)) + zero_shift
  while new_arr < n_end
    new_arr = T_end + rand(ia_dist(T_end,K_start)) + zero_shift
  end

  if new_arr > (n_end + n_preds)
    return []
  else
    T = [new_arr]
    j = 1
    K = K_start + j
    while T[end] < (n_end + n_preds)
      j += 1
      append!(T,rand(ia_dist(T[j-1],K_start+j-1)) + T[j-1] + zero_shift)
    end
    if T[end] > (n_end + n_preds)
      pop!(T)
    end
    return T
  end

end


function sample_predicted_sequence(PP_train::Vector{Int},T_end::Int,
  ia_dist::DiscreteDistribution,alpha::Float64,n_preds::Int)

  f = (x,y) -> ia_dist
  sample_predicted_sequence(PP_train,T_end,f,alpha,n_preds)

end

function sample_predicted_sequence(PP_train::Vector{Int},T_end::Int,
  ia_dist::Function,alpha::Float64,n_preds::Int)

  K_train = size(PP_train,1)
  n_train = sum(PP_train)
  T_pred = sample_predicted_arrival_times(ia_dist,T_end,K_train,n_train,n_preds)
  K_test = size(T_pred,1)
  K_end = K_train + K_test

  PP = vcat(PP_train,zeros(Int64,size(T_pred,1)))
  Z = zeros(Int64,n_preds)

  if K_test==0
    for n in 1:n_preds
      Z[n] = wsample(1:K_train, PP .- alpha)
      PP[Z[n]] += 1
    end
    return Z,PP,T_pred
  else

    k = K_train
    for n in 1:n_preds
      if n+n_train <= T_pred[end] && n+n_train == T_pred[k+1-K_train]
        k += 1
        PP[k] = 1
        Z[n] = k
        k > K_end ? k = K_end : nothing
      else
        Z[n] = wsample(1:k, PP[1:k] .- alpha) # discounted size-biased sample
        PP[Z[n]] += 1
      end
    end
    return Z,PP,T_pred
  end
end

function sample_predicted_partition(PP_train::Vector{Int},T_end::Int,
  ia_dist::DiscreteDistribution,alpha::Float64,n_preds::Int)

  f = (x,y) -> ia_dist
  return sample_predicted_partition(PP_train,T_end,f,alpha,n_preds)

end

function sample_predicted_partition(PP_train::Vector{Int},T_end::Int,
  ia_dist::Function,alpha::Float64,n_preds::Int)

  Z_pred,PP,T_pred = sample_predicted_sequence(PP_train,T_end,ia_dist,alpha,n_preds)
  return PP,T_pred

end
