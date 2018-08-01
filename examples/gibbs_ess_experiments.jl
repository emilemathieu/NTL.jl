# Gibbs sampling experiments for paper (ESS, parameter recovery)
using Distributions
using StatsBase
using MCMCDiagnostics

include("../dataset.jl")
include("../crp.jl")
include("../geometric_ia.jl")
include("../poisson_ia.jl")
include("../ntl_gibbs.jl")
include("../slice.jl")
include("../evaluation.jl")
include("../gibbs_util.jl")
include("../experiments_util.jl")


###########################################################################
# Experiment settings
###########################################################################
save_output = true
N_train = 1000 # size of sequence for synthetic data
N_test = 2000 # size of total sequence for computing predictive likelihoods

n_iter = 125000 # 50000  # total number of Gibbs iterations to run
n_burn = 25000   # burn-in
n_thin = 1000     # collect every `n_thin` samples

n_rep = 10 # number of sampling experiment repetitions

n_print = 1000 # prints updates every `n_print` iterations

ntl_alpha = 0.75 # -2.0 # "true" value of alpha that will be used to generate data

# set which components to update
gibbs_psi = true            # NTL Ψ paramters
gibbs_alpha = true          # NTL alpha parameter
gibbs_arrival_times = true  # arrival times
gibbs_ia_params = true     # arrival time distribution parameters
gibbs_perm_order = true   # order of blocks in partition/vertices in graph

datasets = ["synthetic crp-coupled","synthetic geometric"]
ia_params = [[1.0], [0.25]] # these correspond to the datasets
arrival_dists = ["crp-coupled","crp-uncoupled","geometric","poisson"]
# arrival_dists = ["geometric","poisson"]

n_ds = size(datasets,1)
n_ad = size(arrival_dists,1)

###########################################################################
# NTL alpha sampling settings
###########################################################################

ntl_gamma_a = 1.
ntl_gamma_b = 10.

# prior is specified as a distribution on (0,Inf);
#   transformation to alpha ∈ (-Inf,1) will be performed during sampling as appropriate
ntl_alpha_prior_dist = Gamma(ntl_gamma_a,ntl_gamma_b)
ntl_alpha_log_prior = x -> logpdf(ntl_alpha_prior_dist,x)

w_alpha = 5.0 # slice sampling "window" parameter

###########################################################################
# Experiment
###########################################################################

# initialize storage
N_data_all = zeros(Int64,n_ds)
K_data_all = zeros(Int64,n_ds)

PP_data_all = []
T_data_all = []
Z_data_all = []
spl_out_all = []

runtimes = zeros(Float64,n_ds,n_ad,n_rep)
ESS_alpha = zeros(Float64,n_ds,n_ad,n_rep)
ESS_logp = zeros(Float64,n_ds,n_ad,n_rep)
ESS_sigma_logd = zeros(Float64,n_ds,n_ad,n_rep)
ESS_T_logd = zeros(Float64,n_ds,n_ad,n_rep)
ESS_slack_logd = zeros(Float64,n_ds,n_ad,n_rep)

sigma_mean_d = zeros(Float64,n_ds,n_ad,n_rep)
sigma_se_d = zeros(Float64,n_ds,n_ad,n_rep)
T_mean_d = zeros(Float64,n_ds,n_ad,n_rep)
alpha_mean_d = zeros(Float64,n_ds,n_ad,n_rep)
alpha_se_d = zeros(Float64,n_ds,n_ad,n_rep)
slack_mean_d = zeros(Float64,n_ds,n_ad,n_rep)
slack_se_d = zeros(Float64,n_ds,n_ad,n_rep)

pred_ll_mean = zeros(Float64,n_ds,n_ad,n_rep)
pred_ll_se = zeros(Float64,n_ds,n_ad,n_rep)


for ds in 1:size(datasets,1)
  ## SET SEED
  srand(0)
  # generate synthetic data
  PP_test,T_test,Z_test = genSynDegrees(datasets[ds],N_test,ntl_alpha,ia_params[ds])
  PP_data_all = [PP_data_all; [PP_test]]
  T_data_all = [T_data_all; [T_test]]
  Z_data_all = [Z_data_all; [Z_test]]

  PP_data = seq2part(Z_test[1:N_train])
  T_data = get_arrivals(Z_test[1:N_train])

  slack_data = cumsum(PP_data)[1:(end-1)] .- T_data[2:end]
  # sort partition if necessary
  if gibbs_perm_order
    PP_sort = sortrows(hcat(PP_data,collect(1:size(PP_data,1))),rev=true)
    # PP_sort_perm = sortperm(PP_data,rev=true)
    # sort ties in ascending order of original order for plotting purposes
    maxdeg = maximum(PP_sort)
    for j in 1:maxdeg
      PP_sort[PP_sort[:,1].==j,2] = sort(PP_sort[PP_sort[:,1].==j,2],rev=false)
    end
    perm_data = PP_sort[:,2]
    PP = PP_sort[:,1]

    K_train = length(PP)
    PP_all_train_sort = PP_test[[perm_data;(K_train+1):end]]

  else
    PP = deepcopy(PP_data)
    perm_data = collect(1:size(PP,1))
  end

  K = size(PP,1)
  N = sum(PP)

  K_data_all[ds] = K
  N_data_all[ds] = N

  for ad in 1:size(arrival_dists,1)
    # set up required sampling functions and control
    ia_prior_params,ia_dist,initialize_arrival_params,update_arrival_params!,lpdf_ia_param_prior,n_ia_params =
        arrivalSamplerSetup(arrival_dists[ad],K,N)

    sampler_control = GibbsSamplerControls(
                        n_iter,n_burn,n_thin,n_print,
                        gibbs_psi,gibbs_alpha,gibbs_arrival_times,gibbs_ia_params,gibbs_perm_order,
                        arrival_dists[ad],ia_dist,ia_prior_params,n_ia_params,initialize_arrival_params,update_arrival_params!,
                        ntl_alpha_log_prior,ntl_alpha_prior_dist,lpdf_ia_param_prior,
                        w_alpha,0.5,[0.5]
                      )

    # run sampler
    println("Running sampler for ",ds," / ",n_ds," datasets; ",ad," / ",n_ad," arrival distributions.")
    for nr in 1:n_rep
      spl_out = GibbsSampler(sampler_control,PP,T_data)
      spl_out_all = [spl_out_all; [spl_out]]
      # process output
      runtimes[ds,ad,nr] = spl_out.t_elapsed
      ESS_alpha[ds,ad,nr] = ess_factor_estimate(spl_out.alpha)[1]
      ESS_logp[ds,ad,nr] = ess_factor_estimate(spl_out.log_joint)[1]

      sigma_d = mean( abs.(PP[spl_out.sigma] .- PP_data), 1 )
      ESS_sigma_logd[ds,ad,nr] = ess_factor_estimate(log.(sigma_d))[1]
      sigma_mean_d[ds,ad,nr] = mean(sigma_d)
      sigma_se_d[ds,ad,nr] = sqrt(var(sigma_d)/length(sigma_d))

      T_d = mean( abs.(spl_out.T .- T_data), 1 )
      ESS_T_logd[ds,ad,nr] = ess_factor_estimate(log.(T_d))[1]
      T_mean_d[ds,ad,nr] = mean(T_d)

      alpha_d = abs.(spl_out.alpha .- ntl_alpha)
      alpha_mean_d[ds,ad,nr] = mean(alpha_d)
      alpha_se_d[ds,ad,nr] = sqrt(var(alpha_d)/length(alpha_d))

      slack_d = mean(abs.(cumsum(PP[spl_out.sigma],1)[1:(end-1),:] .- spl_out.T[2:end,:] .- slack_data), 1)
      ESS_slack_logd[ds,ad,nr] = ess_factor_estimate(log.(slack_d))[1]
      slack_mean_d[ds,ad,nr] = mean(slack_d)
      slack_se_d[ds,ad,nr] = sqrt(var(slack_d)/length(slack_d))

      pred_ll = zeros(Float64,size(spl_out.alpha,1))
      for s in 1:size(spl_out.alpha,1)
        pred_ll[s] = logp_pred_partition(
                      PP[spl_out.sigma[:,s]],PP_all_train_sort[[spl_out.sigma[:,s];(K_train+1):end]],
                      [spl_out.T[:,s];T_test[(K_train+1):end]],
                      spl_out.alpha[s],ia_dist(spl_out.ia_params[:,s]),false,false) # treat as sequence
      end
      pred_ll_mean[ds,ad,nr] = mean(pred_ll)
      pred_ll_se[ds,ad,nr] = sqrt(var(pred_ll)/length(pred_ll))

      println("\n > Finished with ",nr," out of ",n_rep," repetitions.")
    end

  end

  println("\n Finished with ",ds," out of ",n_ds," datasets. \n")

end

######### Save output

if save_output
  using JLD
  using JSON
  using DataStructures

  datetime = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
  try
      Base.Filesystem.mkdir("./ess_output")
  catch
      nothing
  end
  dirname = "./ess_output/gibbs_" * datetime * "_"
  fname = "ess_results.jld"
  pathname = dirname * fname

  save(pathname,
      "runtimes",runtimes,
      "ESS_alpha",ESS_alpha,
      "ESS_logp",ESS_logp,
      "ESS_sigma_logd",ESS_sigma_logd,
      "sigma_mean_d",sigma_mean_d,
      "sigma_se_d",sigma_se_d,
      "ESS_T_logd",ESS_T_logd,
      "T_mean_d",T_mean_d,
      "alpha_mean_d",alpha_mean_d,
      "alpha_se_d",alpha_se_d,
      "ESS_slack_logd",ESS_slack_logd,
      "slack_mean_d",slack_mean_d,
      "slack_se_d",slack_se_d,
      "pred_ll_mean",pred_ll_mean,
      "pred_ll_se",pred_ll_se,
      "spl_out_all",spl_out_all)

  save(dirname * "syn_data.jld",
      "PP_data_all",PP_data_all,
      "T_data_all",T_data_all,
      "Z_data_all",Z_data_all,
      "K_data_all",K_data_all)

  params = OrderedDict(
  "datasets" => datasets, "arrival_dists" => arrival_dists, "ntl_alpha" => ntl_alpha, "N_train" => N_train, "N_test" => N_test, "ia_params" => ia_params,
  "n_iter" => n_iter, "n_burn" => n_burn, "n_thin" => n_thin, "n_rep" => n_rep,
  "gibbs_psi" => gibbs_psi, "gibbs_alpha" => gibbs_alpha, "gibbs_arrival_times" => gibbs_arrival_times, "gibbs_ia_params" => gibbs_ia_params, "gibbs_perm_order" => gibbs_perm_order,
  "ntl_gamma_a" => ntl_gamma_a, "ntl_gamma_b" => ntl_gamma_b, "w_alpha" => w_alpha
  )

  open(dirname * "params.json", "w") do f
      write(f, JSON.json(params))
  end

end
