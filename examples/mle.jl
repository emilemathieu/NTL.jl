using Optim
using StatsBase
include("../dataset.jl")
include("../likelihoods.jl")

percent = ARGS[1]
split_prop = parse(Float64, percent)/100
# fname must be a SNAP dataset, preprocessed according to README.md
fname = ARGS[2]

# Compute sufficient statistics used by the likelihood functions
degs, ts, _, _, _, _ = trainTestSplitSnapData("$fname", split_prop)
Tend = sum(degs)
dmap = countmap(degs)
ds = collect(keys(dmap))
dcounts = collect(values(dmap))
K = sum(dcounts)

# Fit a PYP to this dataset
result = optimize(params -> -pyp_llikelihood(params, ds, dcounts, ts, K, Tend),
                  (storage, params) -> neg_grad_pyp_llikelihood!(storage, params, ds, dcounts, ts, K, Tend),
                  [0., .5], LBFGS())
tau = exp(result.minimizer[1])/(1+exp(result.minimizer[1]))
theta = result.minimizer[2]
println("Tau ", tau, " Theta ", theta, )
println("Optimized ll ", -result.minimum)
println("Expected powerlaw ", 1+tau)

# Fit a Geom(beta)Beta Neutral-to-the-left model to this dataset
result = optimize(a -> -ntl_llikelihood(a, ds, dcounts, ts, K, Tend),
                  (storage, params) -> neg_grad_ntl_llikelihood!(storage, params, ds, dcounts, ts, K, Tend),
                  [0.], LBFGS())
alpha = 1 - exp(result.minimizer[1])
println("NTL alpha ", alpha)
ll_ntl = result.minimum
result = optimize(g -> -geom_llikelihood(g, K, Tend), [.5], LBFGS())
g = exp(result.minimizer[1])/(1+exp(result.minimizer[1]))
println("g ", g)
ll = ll_ntl + result.minimum
println("Optimized ll ", -ll)
println("Expected powerlaw ", 1+(1/g - alpha)/(1/g - 1))
