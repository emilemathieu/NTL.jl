# utility functions for Gibbs sampling

mutable struct GibbsSamplerControls
  """
    Structure to store the parameters of the Gibbs sampler
  """
  n_iter::Int64 # total number of Gibbs iterations to run
  n_burn::Int64  # burn-in
  n_thin::Int64 #  collect every `n_thin` samples
  n_print::Int64 # print status updates with timings every `n_print` iterations

  # set which components to update
  gibbs_psi::Bool # NTL Ψ paramters
  gibbs_alpha::Bool  # NTL alpha parameter
  gibbs_T::Bool # arrival times
  gibbs_ia_params::Bool # arrival time distribution parameters
  gibbs_sigma::Bool # order of blocks in partition/vertices in graph

  # set arrival model
  arrivals::String
  ia_distribution::Function
  # prior parameters
  ia_prior_params::Vector{Float64}
  n_ia_params::Int64

  # model-specific functions used within the sampler
  initialize_arrival_params::Function # for initializing arrival distn parameters
  update_arrival_params::Function # for updating arrival distn parameters
  ntl_alpha_log_prior::Function # for calculating the prior density of alpha
  ntl_alpha_prior_dist::UnivariateDistribution # for initializing alpha
  lpdf_ia_param_prior::Function # log pdf for interarrival parameter priors

  # slice-sampling parameters
  ss_w_alpha::Float64 # slice-sampling window parameter for alpha

  alpha_fixed::Float64 # when alpha is fixed (i.e., not to be updated)
  arrival_params_fixed::Vector{Float64} # when arrival distn parameters are fixed

end

struct GibbsSamplerOutput
 """
  Structure to store posterior samples
 """
  psi::Array{Float64,2}
  T::Array{Int64,2}
  alpha::Array{Float64,1}
  ia_params::Array{Float64}
  sigma::Array{Int64,2}
  log_joint::Vector{Float64}
  t_elapsed::Float64

end


function GibbsSampler(G::GibbsSamplerControls,PP::Vector{Int},T_data::Vector{Int})
  """
    Main function for running the Gibbs sampler.

    Inputs:
      - `G`: a GibbsSamplerControls structure
      - `PP`: a partition or degree vector
      - `T_data`: a vector of arrival times (pass an empty vector if unknown)
  """

  n_iter = G.n_iter
  n_burn = G.n_burn
  n_thin = G.n_thin

  gibbs_psi = G.gibbs_psi
  gibbs_T = G.gibbs_T
  gibbs_alpha = G.gibbs_alpha
  gibbs_ia_params = G.gibbs_ia_params
  gibbs_sigma = G.gibbs_sigma

  arrivals = G.arrivals
  ia_dist = G.ia_distribution
  ia_prior_params = G.ia_prior_params
  n_ia_params = G.n_ia_params

  w_alpha = G.ss_w_alpha

  initialize_arrival_params = G.initialize_arrival_params
  update_arrival_params! = G.update_arrival_params
  ntl_alpha_log_prior = G.ntl_alpha_log_prior
  ntl_alpha_prior_dist = G.ntl_alpha_prior_dist

  lpdf_ia_param_prior = G.lpdf_ia_param_prior

  alpha_fixed = G.alpha_fixed
  arrival_params_fixed = G.arrival_params_fixed

  K = size(PP,1)
  N = sum(PP)

  # println("Initializing sampler.")
  gibbs_psi ? psi_gibbs = zeros(Float64,K,Int(ceil((n_iter-n_burn)/n_thin))) : psi_gibbs = []
  gibbs_T ? T_gibbs = zeros(Int,K,Int(ceil((n_iter-n_burn)/n_thin))) : T_gibbs = []
  gibbs_alpha ? alpha_gibbs = zeros(Float64,Int(ceil((n_iter-n_burn)/n_thin))) : alpha_gibbs = []
  gibbs_ia_params ? ia_params_gibbs = zeros(Float64,n_ia_params,Int(ceil((n_iter-n_burn)/n_thin))) : ia_params_gibbs = []
  gibbs_sigma ? perm_gibbs = zeros(Int,K,Int(ceil((n_iter-n_burn)/n_thin))) : perm_gibbs = []
  log_joint_gibbs = zeros(Float64,Int(ceil((n_iter-n_burn)/n_thin)))

  # initialize
  gibbs_psi ? psi_current = 0.5*ones(Float64,K) : nothing
  gibbs_alpha ? alpha_current = initialize_alpha(ntl_alpha_prior_dist) : alpha_current = [alpha_fixed]
  if gibbs_ia_params
    arrival_params_current = initialize_arrival_params(ia_prior_params)
  else
    arrival_params_current = arrival_params_fixed
  end
  if gibbs_T
    T_current = initialize_arrival_times(PP,alpha_current[1],Geometric((K-1)/(N-1)))
  elseif isempty(T_data)
    error("Empty `T_data` is not compatible with not inferring arrival times.")
  else
    T_current = T_data
  end
  perm_current = collect(1:K) # initial permutation order


  # Gibbs sampler
  n_print = G.n_print
  ct_gibbs = 0 # counts when to store state of Markov Chain

  t_elapsed = 0.
  # println("Finished initializing sampler.")

  # println("Running Gibbs sampler.")
  tic();
  for s in 1:n_iter
    gibbs_psi ? update_psi_parameters_partition!(psi_current,PP[perm_current],alpha_current[1]) : nothing ;
    gibbs_T ? update_arrival_times!(T_current,PP[perm_current],alpha_current[1],ia_dist(arrival_params_current)) : nothing ;
    gibbs_sigma ? update_block_order!(perm_current,PP[perm_current],T_current,alpha_current[1]) : nothing ;
    gibbs_ia_params ? update_arrival_params!(arrival_params_current,PP[perm_current],T_current,N,ia_prior_params) : nothing ;
    if arrivals=="crp-coupled"
      alpha_current[1] = arrival_params_current[2]
    else
      gibbs_alpha ? update_ntl_alpha!(alpha_current,PP[perm_current],T_current,ntl_alpha_log_prior,w_alpha) : nothing ;
    end

    if (s > n_burn) && mod(s - n_burn,n_thin)==0
      ct_gibbs += 1 ;
      gibbs_psi ? psi_gibbs[:,ct_gibbs] = psi_current : nothing ;
      gibbs_T ? T_gibbs[:,ct_gibbs] = T_current : nothing ;
      gibbs_alpha ? alpha_gibbs[ct_gibbs] = alpha_current[1] : nothing ;
      gibbs_ia_params ? ia_params_gibbs[:,ct_gibbs] = arrival_params_current : nothing ;
      gibbs_sigma ? perm_gibbs[:,ct_gibbs] = perm_current : nothing ;
      log_joint_gibbs[ct_gibbs] = logp_partition(PP[perm_current],T_current,alpha_current[1],ia_dist(arrival_params_current),true) +
                      ntl_alpha_log_prior(1 - alpha_current[1]) + lpdf_ia_param_prior(arrival_params_current)
    end
    if mod(s,n_print)==0
      t_elapsed += toq();
      println("  >> Finished with ",s," / ",n_iter," samples. Elapsed time is ",t_elapsed," seconds.")
      tic();
    end
  end

  println("Finished running Gibbs sampler.")

  return GibbsSamplerOutput(psi_gibbs,T_gibbs,alpha_gibbs,ia_params_gibbs,perm_gibbs,log_joint_gibbs,t_elapsed)

end
