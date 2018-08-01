##### Chinese Restaurant Process arrival time utilities

type CRPinterarrival <: DiscreteUnivariateDistribution
  theta::Float64
  alpha::Float64
  n::Int
  k::Int
  # crp::Bool
end

Base.minimum(s::CRPinterarrival) = 1
Base.maximum(s::CRPinterarrival) = Inf

function partial(f,a...)
  (b...) -> f(a...,b...)
end

function CRP(a...)
  """
  Utility function for passing arguments `n` and `k` during sampling updates
  """
  partial(CRPinterarrival,a...)
end

Distributions.logpdf(s::CRPinterarrival,x::Int) = _logpdf(s,x)
Distributions.logpdf(s::CRPinterarrival,v::Vector{Int}) = _logpdf_batch(s,v)
function _logpdf(s::CRPinterarrival,x::Int)
  return crp_logpdf(s.theta,s.alpha,s.n,s.k,x)
end

function crp_logpdf(theta::Float64,alpha::Float64,n::Int,k::Int,x::Int)
  ka = k*alpha
  nt = n + theta
  logp = log(theta + ka) - log(nt)
  if x > 1
    nka = n - ka
    for i in 2:x
      logp += log(nka + i - 2) - log(nt + i - 1)
    end
  end
  return logp
end

function crp_logpdf(theta::Float64,alpha::Float64,n::Int,k::Int,v::Vector{Int})
  # for a vector of evaluation points
  vmax = maximum(v)
  vmin = minimum(v)
  nv = size(v,1)
  idx_all = 1:nv
  idx_j = trues(nv)
  ret = zeros(Float64,nv)

  ka = k*alpha
  nt = n + theta
  nka = n - ka
  logp = log(theta + ka) - log(nt)
  for j in 1:vmax
    j > 1 ? logp += log(nka + j - 2) - log(nt + j - 1) : nothing
    for i in idx_all[idx_j]
      if v[i]==j
        ret[i] = logp
        idx_j[i] = false
      end
    end
  end
  return ret
end

function _logpdf_batch(s::CRPinterarrival,v::Vector{Int})
  return crp_logpdf(s.theta,s.alpha,s.n,s.k,v)
end

Distributions.pdf(s::CRPinterarrival,x::Int) = _pdf(s,x)
function _pdf(s::CRPinterarrival,x::Int)
  return exp.(logpdf(s,x))
end

Distributions.cdf(s::CRPinterarrival,x::Int64) = _cdf(s,x)
function _cdf(s::CRPinterarrival,x::Int64)
  return crp_cdf(s.theta,s.alpha,s.n,s.k,x)
end

function crp_cdf(theta::Float64,alpha::Float64,n::Int,k::Int,x::Int)
  if x==0
    return 0.
  else
    ka = k*alpha
    nt = n + theta
    P1 = (theta + ka)/(nt)
    if x > 1
      nka = n - ka
      prev = P1
      P = 0
      for j in 2:x
        run = exp( log(prev) + log(nka + j - 2) - log(nt + j - 1) )
        P += run
        prev = run
      end
    end
    return x > 1 ? P1 + P : P1
  end
end

Base.Random.rand(s::CRPinterarrival) = _rand(s)
function _rand(s::CRPinterarrival)
  coin = 0
  ct = 0
  while coin != 1
    ct += 1
    p = (s.theta + s.alpha*s.k)/(s.theta + s.n + ct - 1)
    coin = rand(Bernoulli(p))
  end
  return ct
end

# slice sampling utilities
function crp_theta_logpdf(theta::Float64,alpha::Float64,k::Int,n::Int,log_prior::Function)
  """
  calculate unnormalized log-pdf proportional to `theta` in the CRP

  log_prior is a function that returns the (possibly unnormalized) prior log-probability
    of `theta`
  """
  if theta <= -alpha
    # println("theta: ",theta,"  //  alpha: ",alpha)
    return -Inf
  end
  logp = log_prior(theta,alpha)
  for j in 1:(k-1)
    logp += log(theta + j*alpha)
  end

  for m in 1:(n-1)
    logp += -log(theta + m)
  end
  return logp
end

function crp_theta_trans_logpdf(theta_trans::Float64,alpha::Float64,k::Int,n::Int,log_prior::Function)
  """
  computes log-pdf when theta has been transformed to the entire real line

  theta_trans = log(theta + alpha) (for fixed alpha)
  """
  theta = exp(theta_trans) - alpha
  return crp_theta_logpdf(theta,alpha,k,n,log_prior) + theta_trans
end

function crp_alpha_logpdf(alpha::Float64,theta::Float64,T::Vector{Int},n::Int,log_prior::Function)
  """
  calculate unnormalized log-pdf proportional to `alpha` in the CRP

  log_prior is a function that returns the (possibly unnormalized) prior log-probability
    of `alpha`
  """
  if theta <= -alpha || alpha > 1. || alpha < 0.
    return -Inf
  end
  k = size(T,1)
  logp = log_prior(alpha) + sum(log.(theta + alpha.*(1:(k-1))))
  logp += sum( lgamma.(T[2:end] .- 1 .- alpha.*(1:(k-1))) .- lgamma.(T[1:(k-1)] .- alpha.*(1:(k-1))) )
  return logp
end

function crp_alpha_trans_logpdf(alpha_trans::Float64,theta::Float64,T::Vector{Int},n::Int,log_prior::Function)
  """
  computes log-pdf when alpha has been transformed to the entire real line

  alpha_trans = log(alpha - max(0,-theta)) - log(1-alpha) for fixed theta
  """
  alpha = (exp(alpha_trans) + max(0,-theta))/(1 + exp(alpha_trans))
  return crp_alpha_logpdf(alpha,theta,T,n,log_prior) + alpha_trans - log(1 + exp(alpha_trans)) + log(1 - alpha)
end

function crp_alpha_trans_coupled_logpdf(alpha_trans::Float64,theta::Float64,PP::Vector{Int},T::Vector{Int},n::Int,log_prior::Function)
  """
  for the "coupled CRP" model, where the NTL alpha parameter is the same as the CRP alpha paramter
  computes log-pdf when alpha has been transformed to the entire real line

  alpha_trans = log(alpha - max(0,-theta)) - log(1-alpha) for fixed theta
  """
  alpha = (exp(alpha_trans) + max(0,-theta))/(1 + exp(alpha_trans))
  crp_arrivals_lpdf = crp_alpha_logpdf(alpha,theta,T,n,log_prior) # arrivals contribution
  trans_lpdf =  alpha_trans - log(1 + exp(alpha_trans)) + log(1 - alpha) # volume conrrection of transformation
  dummy_prior = x -> 0.
  crp_sbr_lpdf = ntl_alpha_logpdf(alpha,PP,T,dummy_prior) # size-biased reinforcement contribution

  return crp_arrivals_lpdf + crp_sbr_lpdf + trans_lpdf
end


# parameter updates
# uncoupled CRP
function update_crp_interarrival_params!(ia_params::Vector{Float64},PP::Vector{Int},T::Vector{Int},
  n::Int,log_prior_theta::Function,log_prior_alpha::Function,
  w_t::Float64,w_a::Float64,coupled::Bool)

  # update theta via slice sampling
  K = size(T,1)
  ss_gt = x -> crp_theta_trans_logpdf(x,ia_params[2],K,n,log_prior_theta)
  theta_trans = log(ia_params[1] + ia_params[2])
  theta_trans_ss = slice_sampling(ss_gt,w_t,theta_trans)
  ia_params[1] = exp(theta_trans_ss) - ia_params[2]

  if !coupled
    ss_ga = x -> crp_alpha_trans_logpdf(x,ia_params[1],T,n,log_prior_alpha)
  else
    ss_ga = x -> crp_alpha_trans_coupled_logpdf(x,ia_params[1],PP,T,n,log_prior_alpha)
  end
  alpha_trans = log(ia_params[2] - max(0.,-ia_params[1])) - log(1 - ia_params[2])
  alpha_trans_ss = slice_sampling(ss_ga,w_a,alpha_trans)
  ia_params[2] = (exp(alpha_trans_ss) + max(0,-ia_params[1]))/(1 + exp(alpha_trans_ss))

end

function crp_theta_loglik(theta::Float64,alpha::Float64,K::Int,n::Int)
  return sum( log.(theta .+ (1:(K-1)).*alpha) ) - sum( log.(theta .+ (1:(n-1))) )
end

function grad_crp_theta_loglik(theta::Float64,alpha::Float64,K::Int,n::Int)
  return sum( 1./(theta .+ (1:(K-1)).*alpha) ) - sum( 1./(theta .+ (1:(n-1))) )
end

function hess_crp_theta_loglik(theta::Float64,alpha::Float64,K::Int,n::Int)
  return -sum( 1./(theta .+ (1:(K-1)).*alpha).^2 ) + sum( 1./(theta .+ (1:(n-1))).^2 )
end

function initialize_crp_params(theta_prior::UnivariateDistribution,alpha_prior::UnivariateDistribution)
  alpha = mean(alpha_prior)
  theta = mean(theta_prior)
  return [theta; alpha]

end
