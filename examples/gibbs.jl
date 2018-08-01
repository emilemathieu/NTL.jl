# Gibbs samplers for Beta NTL models of partitions, graphs
using Distributions
using StatsBase

include("../dataset.jl")
include("../crp.jl")
include("../ntl_gibbs.jl")
include("../slice.jl")
include("../evaluation.jl")
include("../gibbs_util.jl")
include("../geometric_ia.jl")
include("../poisson_ia.jl")


###########################################################################
# 1. Gibbs sampler settings
###########################################################################
dataset_name = "synthetic geometric" # set dataset; see section 2. Data for options that are automated

arrivals = "geometric" # set arrival time model
save_output = false # whether or not to save sampler output

if startswith(dataset_name, "synthetic")
  N = 1000 # set size of synthetic data (# of ends of edges)
end

n_iter = 10000 # 50000  # total number of Gibbs iterations to run
n_burn = 5000   # burn-in
n_thin = 100     # collect every `n_thin` samples

n_print = 1000 # prints updates every `n_print` iterations

# set which components to update
gibbs_psi = true            # NTL Ψ paramters
gibbs_alpha = true          # NTL alpha parameter
gibbs_arrival_times = true  # arrival times
gibbs_ia_params = true     # arrival time distribution parameters
gibbs_perm_order = true   # order of blocks in partition/vertices in graph

## SET SEED
srand(0)


############################################################################
# 2. Data
############################################################################

if startswith(dataset_name,"synthetic crp") # Synthetic data w/ CRP interarrivals
  println("Synethsizing data.")
  # set generating parameter values
  ntl_alpha = 0.8
  crp_theta = 1.0
  if endswith(dataset_name,"uncoupled")
    crp_alpha = 0.6
  else
    crp_alpha = ntl_alpha
  end

  # create intearrival distribution object and synthetic data
  interarrival_dist = CRP(crp_theta,crp_alpha)
  Z_data, PP_data, T_data = generateLabelSequence(N,ntl_alpha,interarrival_dist)

  # function check
  assert(all(PP_data .== seq2part(Z_data)))
  assert(all(T_data .== get_arrivals(Z_data)))

elseif dataset_name=="synthetic geometric" # Synthetic data w/ geometric interarrivals
  println("Synethsizing data.")
  ntl_alpha = 0.8
  geom_p = 0.5

  # create intearrival distribution object and synthetic data
  interarrival_dist = Geometric
  Z_data, PP_data, T_data = generateLabelSequence(N,ntl_alpha,interarrival_dist(geom_p))

  # function check
  assert(all(PP_data .== seq2part(Z_data)))
  assert(all(T_data .== get_arrivals(Z_data)))

elseif dataset_name=="synthetic poisson" # Synthetic data w/ Poisson interarrivals
  println("Synethsizing data.")
  ntl_alpha = 0.5
  poisson_lambda = 4.0

  # create intearrival distribution object and synthetic data
  interarrival_dist = Poisson
  Z_data, PP_data, T_data = generateLabelSequence(N,ntl_alpha,interarrival_dist(poisson_lambda))

  # function check
  assert(all(PP_data .== seq2part(Z_data)))
  assert(all(T_data .== get_arrivals(Z_data)))

elseif dataset_name=="college msg" # use College Message data
  elist = readdlm("../data/CollegeMsg.txt",Int64)
  Z_data = vec(elist[:,1:2]')
  PP_data = seq2part(Z_data)
  T_data = get_arrivals(Z_data)

elseif dataset_name=="mathoverflow"
  Z_data = vec(readdlm("../data/sx-mathoverflow-Z.txt",Int64))
  PP_data = seq2part(Z_data)
  T_data = get_arrivals(Z_data)

end

# sort partition if necessary
if gibbs_perm_order
  PP_sort = sortrows(hcat(PP_data,collect(1:size(PP_data,1))),rev=true)
  # sort ties in ascending order of original order for plotting purposes
  maxdeg = maximum(PP_sort)
  for j in 1:maxdeg
    PP_sort[PP_sort[:,1].==j,2] = sort(PP_sort[PP_sort[:,1].==j,2],rev=false)
  end
  perm_data = PP_sort[:,2]
  PP = PP_sort[:,1]

else
  PP = deepcopy(PP_data)
  perm_data = collect(1:size(PP,1))

end

K = size(PP,1)
N = sum(PP)
gibbs_alpha ? nothing : alpha_fixed = ntl_alpha

println("Finished pre-processing data.")

###########################################################################
# 3. Settings for sampling the BNTL α parameter
###########################################################################

# Gamma prior parameters
ntl_gamma_a = 1.
ntl_gamma_b = 10.

# prior is specified as a distribution on (0,Inf);
# transformation to α ∈ (-Inf,1) will be performed during sampling as appropriate
ntl_alpha_prior_dist = Gamma(ntl_gamma_a,ntl_gamma_b)
ntl_alpha_log_prior = x -> logpdf(ntl_alpha_prior_dist,x)

w_alpha = 5.0 # slice sampling "window" parameter

###########################################################################
# 4. Modeled arrival time distribution settings
###########################################################################

if startswith(arrivals,"crp")
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

  gibbs_ia_params ? nothing : arrival_params_fixed = [mean(Gamma(ia_prior_params[1],ia_prior_params[2])); mean(Beta(ia_prior_params[3],ia_prior_params[4]))]

elseif arrivals=="geometric"
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


############################################################################
# 5. Gibbs sampler
############################################################################

# PP is the vector of block sizes/vertex degrees to be used.
# If block/vertex order is being inferred, PP should be sorted in descending order;
#   otherwise, it should be in arrival-order

sampler_control = GibbsSamplerControls(
                    n_iter,n_burn,n_thin,n_print,
                    gibbs_psi,gibbs_alpha,gibbs_arrival_times,gibbs_ia_params,gibbs_perm_order,
                    arrivals,ia_dist,ia_prior_params,n_ia_params,initialize_arrival_params,update_arrival_params!,
                    ntl_alpha_log_prior,ntl_alpha_prior_dist,lpdf_ia_param_prior,
                    w_alpha,
                    0.5,[0.5] # these are place-holders that aren't used when all parameters are being inferred
                  )

spl_out = GibbsSampler(sampler_control,PP,T_data)


############################################################################
# 6. Save sample output
############################################################################

if save_output
  using JLD
  using JSON
  using DataStructures

  datetime = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
  try
    Base.Filesystem.mkdir("./sampler_output")
  catch
    nothing
  end
  dirname = "./sampler_output/gibbs_" * datetime * "_"
  fname = "samples.jld"
  pathname = dirname * fname
  save(pathname,
      "psi_gibbs",spl_out.psi,
      "T_gibbs",spl_out.T,
      "alpha_gibbs",spl_out.alpha,
      "ia_params_gibbs",spl_out.ia_params,
      "perm_gibbs",spl_out.sigma,
      "N",N,"K",K,"t_elapsed",spl_out.t_elapsed,
      "PP",PP,"perm_data",perm_data,"T_data",T_data,"Z_data",Z_data)

  params = OrderedDict(
  "dataset_name" => dataset_name, "arrivals" => arrivals,
  "n_iter" => n_iter, "n_burn" => n_burn, "n_thin" => n_thin,
  "gibbs_psi" => gibbs_psi, "gibbs_alpha" => gibbs_alpha, "gibbs_arrival_times" => gibbs_arrival_times, "gibbs_ia_params" => gibbs_ia_params, "gibbs_perm_order" => gibbs_perm_order,
  "ntl_gamma_a" => ntl_gamma_a, "ntl_gamma_b" => ntl_gamma_b, "w_alpha" => w_alpha
  )
  if dataset_name == "synthetic crp"
      params["N"] = N
      params["ntl_alpha"] = ntl_alpha
      params["crp_theta"] = crp_theta
      params["crp_alpha"] = crp_alpha
  elseif dataset_name == "synthetic geometric"
      params["N"] = N
      params["ntl_alpha"] = ntl_alpha
      params["geom_p"] = geom_p
  end
  if arrivals == "crp"
      params["theta_gamma_a"] = theta_gamma_a
      params["theta_gamma_b"] = theta_gamma_b
      params["alpha_beta_a"] = alpha_beta_a
      params["alpha_beta_b"] = alpha_beta_b
      params["w_t"] = w_t
      params["w_a"] = w_a
  elseif arrivals == "geometric"
      params["a_beta"] = a_beta
      params["b_beta"] = b_beta
  end

  open(dirname * "params.json", "w") do f
      write(f, JSON.json(params))
  end
end
