# Utilities for sampling experiments

function genSynDegrees(dataset_name::String,N::Int,ntl_alpha::Float64,ia_params::Vector{Float64})
  """
  generates synthetic data for ESS experiments
  """
  if startswith(dataset_name,"synthetic crp") # Synthetic data w/ CRP interarrivals

    crp_theta = ia_params[1]
    if endswith(dataset_name,"uncoupled")
      crp_alpha = ia_params[2] # [.25, .75]
    else
      ntl_alpha < 0 ? error("Coupled CRP cannot have negative α.") : crp_alpha = ntl_alpha
    end

    # create intearrival distribution object and synthetic data
    interarrival_dist = CRP(crp_theta,crp_alpha)
    Z_data, PP_data, T_data = generateLabelSequence(N,ntl_alpha,interarrival_dist)

    # gibbs_ia_params ? nothing : arrival_params_fixed = [crp_theta; crp_alpha]

    # function check
    assert(all(PP_data .== seq2part(Z_data)))
    assert(all(T_data .== get_arrivals(Z_data)))

  elseif dataset_name=="synthetic geometric" # Synthetic data w/ geometric interarrivals

    geom_p = ia_params[1]
    # create intearrival distribution object and synthetic data
    interarrival_dist = Geometric
    Z_data, PP_data, T_data = generateLabelSequence(N,ntl_alpha,interarrival_dist(geom_p))

    # gibbs_ia_params ? nothing : arrival_params_fixed = [geom_p]

    # function check
    assert(all(PP_data .== seq2part(Z_data)))
    assert(all(T_data .== get_arrivals(Z_data)))

  elseif dataset_name=="synthetic poisson" # Synthetic data w/ geometric interarrivals

    poisson_lambda = ia_params[1]
    # create intearrival distribution object and synthetic data
    interarrival_dist = Poisson
    Z_data, PP_data, T_data = generateLabelSequence(N,ntl_alpha,interarrival_dist(poisson_lambda))

    # gibbs_ia_params ? nothing : arrival_params_fixed = [geom_p]

    # function check
    assert(all(PP_data .== seq2part(Z_data)))
    assert(all(T_data .== get_arrivals(Z_data)))

  end
  return PP_data,T_data,Z_data
end


function arrivalSamplerSetup(arrivals,K,N)

  if startswith(arrivals,"crp")
    # include("crp.jl")
    # CRP arrival distribution
    ia_dist = v -> CRP(v[1],v[2])

    # prior on CRP parameters
    n_ia_params = 2
    theta_gamma_a = 0.1
    theta_gamma_b = 10.
    alpha_beta_a = 1.
    alpha_beta_b = 1.
    ia_prior_params = [theta_gamma_a; # prior on theta
                       theta_gamma_b;
                       alpha_beta_a; # prior on crp_alpha
                       alpha_beta_b]
   ia_theta_prior = Gamma(theta_gamma_a,theta_gamma_b)
   ia_alpha_prior = Beta(alpha_beta_a,alpha_beta_b)
   lpdf_ia_param_prior = pp -> logpdf(ia_theta_prior,pp[1]+pp[2]) + logpdf(ia_alpha_prior,pp[2])

    # slice sampling functions/parameters
    f_lp_t = (x,a) -> logpdf(ia_theta_prior,x+a)
    f_lp_a = x -> logpdf(ia_alpha_prior,x)
    w_t = 1.0 # slice sampling w parameter for crp_theta
    w_a = 1.0 # slice sampling w parameter for crp_alpha

    # CRP-specific sampling functions
    initialize_arrival_params = v -> initialize_crp_params(Gamma(v[1],v[2]),Beta(v[3],v[4]))
    if arrivals=="crp-uncoupled"
      update_arrival_params! = (ap,PP,T,n,pripar) -> update_crp_interarrival_params!(ap,PP,T,n,f_lp_t,f_lp_a,w_t,w_a,false)
    elseif arrivals=="crp-coupled"
      update_arrival_params! = (ap,PP,T,n,pripar) -> update_crp_interarrival_params!(ap,PP,T,n,f_lp_t,f_lp_a,w_t,w_a,true)
    else
      error("Unsupported arrival distribution specification.")
    end

  elseif arrivals=="geometric"
    # include("geometric_ia.jl")
    # Geometric interarrival distribution
    ia_dist = p -> Geometric(p[1])
    a_beta = 1.
    b_beta = 1.
    ia_prior_params = [a_beta; b_beta]
    ia_param_prior = Beta(a_beta,b_beta)
    lpdf_ia_param_prior = pp -> logpdf(ia_param_prior,pp[1])
    n_ia_params = 1
    # set update functions
    gibbs_ia_params ? nothing : arrival_params_fixed = [mean(ia_param_prior)]
    initialize_arrival_params = v -> [(K-1)/(N-1)]
    update_arrival_params! = update_geometric_interarrival_param!

  elseif arrivals=="poisson"
    # include("poisson_ia.jl")
    # Poisson interarrival distribution
    ia_dist = lambda -> Poisson(lambda[1])
    a_gamma = 0.01
    b_gamma = 10.
    ia_prior_params = [a_gamma; b_gamma]
    ia_param_prior = Gamma(a_gamma,b_gamma)
    lpdf_ia_param_prior = pp -> logpdf(ia_param_prior,pp[1])
    n_ia_params = 1
    # set update functions
    gibbs_ia_params ? nothing : arrival_params_fixed = mean(ia_param_prior)
    initialize_arrival_params = v -> [(N-1)/(K-1)]
    update_arrival_params! = update_poisson_interarrival_param!

  else
    error("Unsupported arrival distribution specification.")
  end

  return ia_prior_params,ia_dist,initialize_arrival_params,update_arrival_params!,lpdf_ia_param_prior,n_ia_params

end
