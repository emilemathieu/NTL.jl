###########################################################################
# Utilities for Gibbs updates
###########################################################################

# using StatsBase
# using Distributions

function logp_partition(PP::Vector{Int},T::Vector{Int},Psi::Vector{Float64},
        alpha::Float64,ia_dist::DiscreteDistribution,is_partition::Bool)
    """
    - `PP`: vector of partition block sizes ordered by arrival time
    - `T`: vector of arrival times
    - `Psi`: vector of beta random variables (can be log(Psi))
    - `alpha`: 'discount parameter' in size-biased reinforcement
    - `ia_dist`: distribution object corresponding to i.i.d. interarrivals
    - `is_partition`: Boolean flag for computing binomial coefficients
    """

    if all(Psi .<= 0)
      # warn("Psi in log space.")
      log_Psi = Psi
      log_Psi_c = log.(1 - exp.(log_Psi))
    elseif all(0 .<= Psi .<= 1)
      log_Psi = log.(Psi)
      log_Psi_c = log.(1 - Psi)
    else
      error("Invalid Ψ.")
    end
    # shift distributions with non-zero mass on zero
    zero_shift = Int(minimum(ia_dist) == 0)

    PP_bar = cumsum(PP)
    ia = T[2:end] .- T[1:(end-1)]

    K = size(Psi,1)
    idx = 1:(K-1)
    N = sum(PP)

    log_p = dot((PP[2:end] .- alpha .- 1),log_Psi[2:end]) + dot((PP_bar[1:(end-1)] .- idx.*alpha .- 1), log_Psi_c[2:end])
    # include arrival times
    log_p += sum(logpdf.(ia_dist, ia .- zero_shift))
    N - T[end] > 0 ? log_p += log(1 - cdf(ia_dist, N-T[end]-zero_shift)) : nothing
    log_p += -sum([lbeta(1 - alpha,T[j] - 1 - (j-1)*alpha) for j in 2:K])
    # include binomial coefficients if for a partition
    if is_partition
      log_p += sum([lbinom(PP_bar[j] - T[j],PP[j] - 1) for j in 2:K])
    end

    return log_p

end

function logp_partition(PP::Vector{Int},T::Vector{Int},
        alpha::Float64,ia_dist::DiscreteDistribution,is_partition::Bool)
    f = (x,y) -> ia_dist
    logp_partition(PP,T,alpha,f,is_partition)
end

function logp_partition(PP::Vector{Int},T::Vector{Int},
        alpha::Float64,ia_dist::Function,is_partition::Bool)
    """
    - `PP`: vector of partition block sizes ordered by arrival time
    - `T`: vector of arrival times
    - `alpha`: 'discount parameter' in size-biased reinforcement
    - `ia_dist`: function that creates an interarrival distribution at fixed parameters
    - `is_partition`: flag for computing binomial coefficients
    """

    # shift distributions with non-zero mass on zero
    zero_shift = Int(minimum(ia_dist(1,1)) == 0)

    PP_bar = cumsum(PP)
    # pop!(PP_bar)
    ia = T[2:end] .- T[1:(end-1)]

    K = length(PP)
    idx = 1:(K-1)
    N = sum(PP)

    log_p = log_CPPF(PP,T,alpha)

    if N - T[end] > 0
      p_gt = 1 - cdf(ia_dist(T[end],K), N-T[end]-zero_shift)
      abs(p_gt)<=eps(one(typeof(p_gt))) || p_gt < 0. ? log_p = -Inf : log_p += log(p_gt)
      # log_p += log(1 - cdf(ia_dist(T[end],K), N-T[end]-zero_shift))
    end
    log_p += sum( [logpdf(ia_dist(T[j-1],j-1),T[j]-T[j-1]-zero_shift) for j in 2:K ])
    # include binomial coefficients if for a partition
    if is_partition
      log_p += sum([lbinom(PP_bar[j] - T[j],PP[j] - 1) for j in 2:K])
    end

    return log_p

end

function logp_pred_partition(PP_train::Vector{Int},PP_test::Vector{Int},
        T::Vector{Int},
        alpha::Float64,ia_dist::DiscreteDistribution,
        is_partition_train::Bool,is_partition_test::Bool)
    f = (x,y) -> ia_dist
    logp_pred_partition(PP_train,PP_test,T,alpha,f,is_partition_train,is_partition_test)
end

function logp_pred_partition(PP_train::Vector{Int},PP_test::Vector{Int},
        T::Vector{Int},
        alpha::Float64,ia_dist::Function,
        is_partition_train::Bool,is_partition_test::Bool)

    zero_shift = Int(minimum(ia_dist(1,1)) == 0)
    K_train = length(PP_train)
    K_test = length(PP_test)

    T_train = T[1:K_train]

    PP_train_bar = cumsum(PP_train)
    N_train = PP_train_bar[end]
    logp_train = log_CPPF(PP_train,T_train,alpha)
    # is_partition_train ? logp_train += sum([lbinom(PP_train_bar[j] - T_train[j],PP_train[j] - 1) for j in 2:length(PP_train)]) : nothing

    PP_test_bar = cumsum(PP_test)
    logp_test = log_CPPF(PP_test,T,alpha)
    # is_partition_test ? logp_test += sum([lbinom(PP_test_bar[j] - T_test[j],PP_test[j] - 1) for j in 2:length(PP_test)]) : nothing
    if is_partition_test
      # predicted part of partition is as usual
      logp_test += sum([lbinom(PP_test_bar[j] - T[j],PP_test[j] - 1) for j in (K_train+1):length(PP_test)])
      # conditioned part of partition is constrained
      for j in 1:K_train
        ( PP_test[j] - N_train > 0 && PP_test_bar[j] - PP_train_bar[j] > 0 )?
          logp_test += lbinom(PP_test_bar[j] - N_train,PP_test[j] - PP_train[j]) : nothing
      end
    end

    N = PP_test_bar[end]

    if N - T[end] > 0
      p_gt = 1 - cdf(ia_dist(T[end],K_test), N-T[end]-zero_shift)
      abs(p_gt)<=eps(one(typeof(p_gt))) || p_gt < 0. ? logp_test = -Inf : logp_test += log(p_gt)
    end
    logp_test += sum( [logpdf(ia_dist(T[j-1],j-1),T[j]-T[j-1]-zero_shift) for j in (K_train+1):K_test])

    return logp_test - logp_train
end

# memoize this?
function lbinom(n::Int,k::Int)
    """
    computes log of binomial coefficient {`n` choose `k`} using `lgamma`
    """
    ret = lgamma(n+1) - lgamma(k+1) - lgamma(n - k + 1)
    return ret
end

function tally_ints(Z::Vector{Int},K::Int)
    """
    counts occurrences in `Z` of integers 1 to `K`
    - `Z`: vector of integers
    - `K`: maximum value to count occurences in `Z`
    """
    ret = zeros(Int,K)
    n = size(Z,1)
    idx_all = 1:n
    idx_j = trues(n)
    for j in 1:K
      for i in idx_all[idx_j]
        if Z[i]==j
          ret[j] += 1
          idx_j[i] = false
        end
      end
    end
    return ret
end

function initialize_alpha(prior_dist::ContinuousUnivariateDistribution)
  return [1 - rand(prior_dist)]
end

function update_ntl_alpha!(alpha::Vector{Float64},PP::Vector{Int},T::Vector{Int},log_prior::Function,w::Float64)
  """
  Slice-sampling update (with sampler parameter `w`) of NTL discount parameter
    `alpha`, conditioned on arrival times `T` and arrival-ordered block/vertex
    counts `PP`.
  """
  alpha_trans = log(1 - alpha[1])
  ss = x -> ntl_alpha_trans_logpdf(x,PP,T,log_prior)
  alpha_trans_new = slice_sampling(ss,w,alpha_trans)
  alpha[1] = 1 - exp(alpha_trans_new)
end

function ntl_alpha_logpdf(alpha::Float64,PP::Vector{Int},T::Vector{Int},log_prior::Function)
    """
    calculate unnormalized log-pdf proportional to `alpha` (discount in NTL)
      (Psi_j's are marginalized)

    log_prior is a function that returns the (possibly unnormalized) prior log-probability
      of `alpha`
    """
    # PP_bar = cumsum(PP)
    logp = log_prior(1 - alpha) + lgamma(PP[1] - alpha) - lgamma(sum(PP) - size(PP,1)*alpha) # prior is specified as a distribution on (0,Inf); alpha ∈ (-Inf,1)
    for j in 2:size(PP,1)
      logp +=  lgamma(PP[j] - alpha) - lbeta(1 - alpha,T[j] - 1 - (j-1)*alpha)
      # logp += lbeta(PP[j] - alpha, PP_bar[j-1] - (j-1)*alpha) - lbeta(1-alpha,T[j] - 1 - (j-1)*alpha)
    end
    return logp
end

function ntl_alpha_trans_logpdf(alpha_trans::Float64,PP::Vector{Int},T::Vector{Int},log_prior::Function)
    """
    calculate unnormalized log-pdf proportional to `alpha_trans`,
      the transformed discount (alpha) in NTL such that alpha = 1 - exp(alpha_trans)

      (Psi_j's are marginalized)

    log_prior is a function that returns the (possibly unnormalized) prior log-probability
      of `alpha`
    """
    return ntl_alpha_logpdf(1 - exp(alpha_trans),PP,T,log_prior) + alpha_trans
end

function seq2part(Z::Vector{Int})
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    """
    # transform Z into an ordered partition and get arrival times
    K = maximum(Z)
    PP = tally_ints(Z,K)

    # T = zeros(Int,K)
    # for j in 1:K
    #   T[j] = findfirst(Z.==j)
    # end
    return PP
end

function get_arrivals(Z::Vector{Int})
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    """

    K = maximum(Z)
    T = zeros(Int,K)
    for j in 1:K
      T[j] = findfirst(Z.==j)
    end
    return T
end

function logp_label_sequence(Z::Vector{Int},Psi::Vector{Float64},
        alpha::Float64,ia_dist::DiscreteDistribution)
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    - `Psi`: vector of beta random variables (can be log(Psi))
    - `alpha`: 'discount parameter' in size-biased reinforcement
    - `ia_dist`: distribution object corresponding to i.i.d. interarrivals
    """

    PP = seq2part(Z)
    T = get_arrivals(Z)
    log_p = logp_partition(PP,T,Psi,alpha,ia_dist,false)
    return log_p

end


function cluster_rm!(x::Vector{Vector{Float64}},k::Int)
  """
  removes column k from vector of vectors x (e.g. corresponding cluster params)
  """
  deleteat!(x,k)
end

function cluster_rm!(x::Vector{Float64},k::Int)
  """
  removes k-th entry from x
  """
  deleteat!(x,k)
end

function cluster_add!(x::Vector{Vector{Float64}},x_new::Vector{Float64},k::Int)
  """
  inserts `x_new` into `x` at entry `k`
  """
  insert!(x,k,x_new)
end

function cluster_add!(x::Vector{Float64},x_new::Float64,k::Int)
  """
  inserts `x_new` into `x` at entry `k`
  """
  insert!(x,k,x_new)
end

function cycle_elements_left!(V::Vector,start_idx::Int,end_idx::Int)
    """
    - `V`: Vector whose elements will be cycled
    - `start_idx`: start of cycle (will be moved to end)
    - `end_idx`: end of cycle
    """

    st = V[start_idx]
    for i in start_idx:(end_idx-1)
      V[i] = V[i+1]
    end
    V[end_idx] = st
    return V
end

function cycle_elements_left!(X::Array,start_idx::Int,end_idx::Int)
    """
    - `X`: Array whose columns will be cycled
    - `start_idx`: start of cycle (will be moved to end)
    - `end_idx`: end of cycle
    """

    for i in 1:size(X,1)
      X[i,:] = cycle_elements_left!(X[i,:],start_idx,end_idx)
    end
    return X
end

function cycle_elements_right!(V::Vector,start_idx::Int,end_idx::Int)
    """
    - `V`: Vector whose elements will be cycled
    - `start_idx`: start of cycle (will be moved to end)
    - `end_idx`: end of cycle
    """

    ed = V[end_idx]
    for i in end_idx:-1:(start_idx+1)
      V[i] = V[i-1]
    end
    V[start_idx] = ed
    return V

end

function cycle_elements_right!(X::Array,start_idx::Int,end_idx::Int)
    """
    - `X`: Array whose columns will be cycled
    - `start_idx`: start of cycle (will be moved to end)
    - `end_idx`: end of cycle
    """

    for i in 1:size(X,1)
      X[i,:] = cycle_elements_right!(X[i,:],start_idx,end_idx)
    end
    return X
end

function log_cppf_counts(PP::Vector{Int},alpha::Float64)
    """
    helper function for `log_CPPF` and `update_label_sequence`
    """
    gt1 = PP .> 1
    ret = sum( lgamma.(PP[gt1] .- alpha) ) - sum(gt1)*lgamma(1 - alpha)
    return ret
end

function log_cppf_arrivals(T::Vector{Int},alpha::Float64)
    """
    helper function for `log_CPPF` and `update_label_sequence`
    """
    K = size(T,1)
    ret = sum( lgamma.(T .- (1:K).*alpha) ) - sum( lgamma.(T[2:end] .- 1 .- (1:(K-1)).*alpha) )
    return ret
end

function log_cppf_arrivals(T::Vector{Int},arrival_offset::Int,alpha::Float64)
  """
  helper function for computing predictive log-probabilities

  `arrival_offset=K` indicates that the first element of `T` corresponds to the
    K-th arrival
  """
  K_end = arrival_offset - 1 + size(T,1)
  return sum( lgamma.(T .- (arrival_offset:K_end).*alpha) ) - sum( lgamma.(T .- 1 .- (arrival_offset-1):(K_end-1).*alpha) )
end

function log_CPPF(PP::Vector{Int},T::Vector{Int},alpha::Float64)
    """
    - `PP`: arrival-ordered vector of partition block sizes
    - `T`: arrival times
    - `alpha`: 'discount' parameter
    """

    n = sum(PP)
    K = size(T,1)

    logp = -lgamma(n - K*alpha) + log_cppf_arrivals(T,alpha) + log_cppf_counts(PP,alpha)
    return logp
end

function log_CPPF(Z::Vector{Int},alpha::Float64)
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    - `alpha`: 'discount' parameter
    """

    T = get_arrivals(Z)
    PP = seq2part(Z)
    logp = log_CPPF(PP,T,alpha)
    return logp
end

function get_num_blocks(Z::Vector{Int})
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    """

    n = size(Z,1)
    K = zeros(Int64,n)
    K[1] = 1
    max_z = 1
    for i in 2:n
      if Z[i] > max_z
        K[i] = K[i-1] + 1
        max_z += 1
      end
    end
    return K
end

function update_psi_parameters_sequence!(Psi::Vector{Float64},Z::Vector{Int},alpha::Float64)
    """
    - `Psi`: vector of current values of Psi
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    - `alpha`: 'discount' parameter
    """

    PP = seq2part(Z)
    update_psi_parameters_partition!(Psi,PP,alpha)
    # Psi[:] = [pu[i] for i in 1:size(Psi,1)]
    return Psi

end

function update_psi_parameters_partition!(Psi::Vector{Float64},PP::Vector{Int},alpha::Float64)
    """
    - `Psi`: vector of current values of Psi
    - `PP`: arrival-ordered vector of partition block sizes
    - `alpha`: 'discount' parameter
    """
    K = size(PP,1)
    PP_bar = cumsum(PP)

    Psi[1] = 1
    for j in 2:K
      Psi[j] = rand(Beta(PP[j]-alpha,PP_bar[j-1]-(j-1)*alpha))
    end
    return Psi
end


function log_sum_exp_weights(logw::Vector{Float64})
  """
  -`logw`: log of weights to be combined for a discrete probability distribution
  """

  maxlogw = maximum(logw)
  shift_logw = logw - maxlogw
  p = exp.(shift_logw)./sum(exp.(shift_logw))
  return p
end


function initialize_arrival_times(PP::Vector{Int},alpha::Float64,ia_dist::DiscreteDistribution)
  f = (x,y)->ia_dist
  initialize_arrival_times(PP,alpha,f)
end

function initialize_arrival_times(PP::Vector{Int},alpha::Float64,ia_dist::Function)
    """
    - `PP`: arrival-ordered vector of partition block sizes
    - `alpha`: 'discount' parameter
    - `ia_dist`: function that returns pdf corresponding to interarrivals
    """
    zero_shift = Int(minimum(ia_dist(1,1)) == 0)
    K = size(PP,1)
    PP_bar = cumsum(PP)
    n = PP_bar[end]

    typeof(ia_dist(1,1))==CRPinterarrival ? crp = true : crp = false
    crp ? crp_dist = ia_dist(1,1) : nothing

    T = zeros(Int,K)
    T[1] = 1

    for j in 2:K
      # determine support of interarrival
      supp = 1:(PP_bar[j-1] - T[j-1] + 1)
      # calculate pmf of conditional distribution
      log_p = zeros(Float64,size(supp,1))
      if crp
        log_p += crp_logpdf(crp_dist.theta,crp_dist.alpha,T[j-1],j-1,supp.-zero_shift)
      else
        log_p += logpdf.(ia_dist(T[j-1],j-1),supp.-zero_shift)
      end
      log_p += lbinom.(PP_bar[j] .- T[j-1] .- supp, PP[j] - 1)
      log_p += lgamma.(T[j-1] .+ supp .- j*alpha) .- lgamma.(T[j-1] .+ supp .- 1 .- (j-1)*alpha)
      # sample an update
      p = log_sum_exp_weights(log_p)
      T[j] = T[j-1] + wsample(supp,p)
      # println("initialized T_",j,"=",T[j])
    end

    return T
end

function sample_interarrival(j::Int,T_jm1::Int,T_jp1::Int,ia_dist::Function,
  zero_shift::Int,PP_bar_jm1::Int,PP_j::Int,alpha::Float64)
  """
  Utility function for arrival time updates
  """
  delta2 = T_jp1 - T_jm1
  # check to see if interarrivals are Geometric or CRP to short-cut some computtions
  typeof(ia_dist(1,1))==Distributions.Geometric{Float64} ? geom = true : geom = false
  typeof(ia_dist(1,1))==CRPinterarrival ? crp = true : crp = false
  crp ? crp_dist = ia_dist(1,1) : nothing
  ia_dist(1,1)==ia_dist(10,10) ? iid = true : iid = false

  # determine support
  supp = 1:min(delta2 - 1, PP_bar_jm1 - T_jm1 + 1)
  if size(supp,1) > 1
    # calculate pmf of conditional distribution
    log_p = zeros(Float64,size(supp,1))
    if crp
      log_p += crp_logpdf(crp_dist.theta,crp_dist.alpha,T_jm1,j-1,supp.-zero_shift)
    elseif !geom && iid # geometric interarrival prior is memoryless => doesn't contribute
      tmp = logpdf.(ia_dist(T_jm1,j-1),supp.-zero_shift)
      log_p += tmp .+ tmp[end:-1:1] # iid interarrivals are symmetric
    end
    log_p += lbinom.(PP_bar_jm1 .+ PP_j .- T_jm1 .- supp, PP_j - 1)
    log_p += lgamma.(T_jm1 .+ supp .- j*alpha) .- lgamma.(T_jm1 .+ supp .- 1 .- (j-1)*alpha)
    if !iid
      for s in supp
        if crp
          log_p[s] += crp_logpdf(crp_dist.theta,crp_dist.alpha,T_jm1+s,j,delta2 - s - zero_shift)
        else
          log_p[s] += logpdf(ia_dist(T_jm1+s,j),delta2 - s - zero_shift)
        end
      end
    end
    # sample an update
    p = log_sum_exp_weights(log_p)
    return wsample(supp,p)
  else
    return 1
  end
end

function sample_final_arrival(T_Km1::Int,K::Int,n::Int,ia_dist::Function,
  zero_shift::Int,PP_bar_Km1::Int,PP_K::Int,alpha::Float64)

  if T_Km1==(n-1)
    TK = n
  else
    typeof(ia_dist(1,1))==Distributions.Geometric{Float64} ? geom = true : geom = false
    typeof(ia_dist(1,1))==CRPinterarrival ? crp = true : crp = false
    crp ? crp_dist = ia_dist(1,1) : nothing

    supp = 1:min(n - T_Km1, PP_bar_Km1 - T_Km1 + 1)
    log_p = zeros(Float64,size(supp,1))

    if crp
      log_p += crp_logpdf(crp_dist.theta,crp_dist.alpha, T_Km1, K-1, supp.-zero_shift)
    elseif !geom
      log_p += logpdf.(ia_dist(T_Km1,K-1), supp.-zero_shift)
    end

    log_p += lbinom.(n .- T_Km1 .- supp, PP_K - 1)
    log_p += lgamma.(T_Km1 .+ supp .- K*alpha) .- lgamma.(T_Km1 .+ supp .- 1 .- (K-1)*alpha)

    if !geom
      for s in supp
        if crp
          p_gt = 1. - crp_cdf(crp_dist.theta,crp_dist.alpha,T_Km1+s,K,n-(T_Km1+s-zero_shift))
        else
          p_gt = 1. - cdf(ia_dist(T_Km1+s,K),n-(T_Km1+s-zero_shift)) # this can be arbitrarily close to zero, need to handle numerical instability
        end
        abs(p_gt)<=eps(one(typeof(p_gt))) || p_gt < 0. ? log_p[s] = -Inf : log_p[s] += log(p_gt)
      end
    end
    p = log_sum_exp_weights(log_p)
    TK = T_Km1 + wsample(supp,p)
  end
  return TK
end

function update_arrival_times!(T::Vector{Int},PP::Vector{Int},alpha::Float64,ia_dist::Function)
    """
    Takes advantage of multiple threads if possible.
    - `PP`: arrival-ordered vector of partition block sizes
    - `T`: current arrival times (to be updated)
    - `alpha`: 'discount' parameter
    - `ia_dist`: function that creates a distribution object corresponding to interarrival distribution
    """
    update_arrival_times_st!(T,PP,alpha,ia_dist)

end

function update_arrival_times!(T::Vector{Int},PP::Vector{Int},alpha::Float64,ia_dist::DiscreteDistribution)
  f = (x,y)->ia_dist
  update_arrival_times!(T,PP,alpha,f)
end

function update_arrival_times_mt!(T::Vector{Int},PP::Vector{Int},alpha::Float64,ia_dist::Function)
    """
    multi-threaded version
    - `PP`: arrival-ordered vector of partition block sizes
    - `T`: current arrival times (to be updated)
    - `alpha`: 'discount' parameter
    - `ia_dist`: function that creates a distribution object corresponding to interarrival distribution
    """
    zero_shift = Int(minimum(ia_dist(1,1)) == 0)
    K = size(T,1)
    PP_bar = cumsum(PP)
    n = PP_bar[end]

    evens = 2:2:(K-1)
    odds = 3:2:(K-1)

    for j in evens
      T[j] = T[j-1] + sample_interarrival(j,T[j-1],T[j+1],ia_dist,zero_shift,PP_bar[j-1],PP[j],alpha)
    end

    for j in odds
      T[j] = T[j-1] + sample_interarrival(j,T[j-1],T[j+1],ia_dist,zero_shift,PP_bar[j-1],PP[j],alpha)
    end

    T[K] = sample_final_arrival(T[K-1],K,n,ia_dist,zero_shift,PP_bar[K-1],PP[K],alpha)
end

function update_arrival_times_st!(T::Vector{Int},PP::Vector{Int},alpha::Float64,ia_dist::Function)
    """
    single-threaded version
    - `PP`: arrival-ordered vector of partition block sizes
    - `T`: current arrival times (to be updated)
    - `alpha`: 'discount' parameter
    - `ia_dist`: function that creates a distribution object corresponding to interarrival distribution
    """
    zero_shift = Int(minimum(ia_dist(1,1)) == 0)
    K = size(T,1)
    PP_bar = cumsum(PP)
    n = PP_bar[end]

    for j in 2:(K-1)
      T[j] = T[j-1] + sample_interarrival(j,T[j-1],T[j+1],ia_dist,zero_shift,PP_bar[j-1],PP[j],alpha)
    end

    T[K] = sample_final_arrival(T[K-1],K,n,ia_dist,zero_shift,PP_bar[K-1],PP[K],alpha)

    return T
end

function swap_elements!(x::Vector,i::Int,j::Int)
  """
  swap elements `i` and `j` of `x` in place
  """
  x[i],x[j] = x[j],x[i]
  return x
end

function update_block_order!(perm::Vector{Int},PP::Vector{Int},T::Vector{Int},alpha::Float64)
    """
    - `perm`: permutation of order of entries in `PP` (to be updated)
    - `PP`: partition of arrival-ordered block sizes
    - `T`: arrival times
    - `alpha`: 'discount' parameter
    Update is through a sequence of proposed adjacent transpositions.

    There are likely better (more efficient) ways to do this.
    """

    K = size(PP,1)
    PP_bar = cumsum(PP)

    for j in 1:(K-1)

      j==1 ? ppbar_jm1 = 0 : ppbar_jm1 = PP_bar[j-1]
      if PP[j]==PP[j+1] # swap is a 50-50 flip
        logp_swap = log(0.5)
        logp_noswap = log(0.5)
      elseif ppbar_jm1 + PP[j+1] >= T[j+1] - 1
        logp_swap = lgamma(ppbar_jm1 + PP[j+1] - T[j] + 1) - lgamma(PP_bar[j+1] - PP[j] - T[j+1] + 2)
        logp_noswap = lgamma(ppbar_jm1 + PP[j] - T[j] + 1) - lgamma(PP_bar[j] - T[j+1] + 2)
      else # impossible to swap blocks given arrival times
        logp_swap = -Inf
        logp_noswap = 0.
      end

      swap = wsample([true;false],log_sum_exp_weights([logp_swap;logp_noswap]))
      if swap
        swap_elements!(PP,j,j+1)
        swap_elements!(perm,j,j+1)
        (j == 1) ? PP_bar[j] =  PP[j] : PP_bar[j] = PP_bar[j-1] + PP[j]
        PP_bar[j+1] = PP_bar[j] + PP[j+1]
      end

    end
    return perm
end
