# Some examples of plots to look at output of Gibbs sampling
using MCMCDiagnostics
using Plots
gr()

# median permutation
med_perm = median(spl_out.sigma,2)

scatter(perm_data,med_perm,legend=false,color=log.(PP))

scatter(med_perm,log.(PP),legend=false,color=perm_data)

scatter(perm_data,mean(spl_out.sigma,2),legend=false,color=log.(PP))

# calculate ess per row
perm_ess_factor = [ess_factor_estimate(log.(spl_out.sigma[i,:])) for i in 1:K  ]
perm_ess = [perm_ess_factor[i][1] for i in 1:size(perm_ess_factor,1)]
perm_lag = [perm_ess_factor[i][2] for i in 1:size(perm_ess_factor,1)]
scatter(med_perm,perm_ess,legend=false,color=log.(PP))
scatter(perm_data,perm_ess,legend=false,color=log.(PP))
scatter(perm_lag,legend=false,color=log.(PP))

scatter(PP,perm_ess,legend=false,color=log.(PP))

# calculate ess per arrival time
med_T = median(spl_out.T,2)
plot(spl_out.T,legend=false)
plot!(T_data,lw=2,color="black",legend=false)


plot(cumsum(PP[spl_out.sigma],1)[1:(end-1),:] .- spl_out.T[2:end,:], legend=false)
plot!(cumsum(PP_data)[1:(end-1)] .- T_data[2:end],color="black",lw=2)


T_ess_factor = [ess_factor_estimate(log.(spl_out.T[i,:])) for i in 1:K  ]
T_ess = [T_ess_factor[i][1] for i in 1:size(T_ess_factor,1)]
T_lag = [T_ess_factor[i][2] for i in 1:size(T_ess_factor,1)]
scatter(T_ess,legend=false,color=log.(mean(PP[spl_out.sigma],2)))
scatter(T_lag,legend=false)


plot(PP_data,lw=1.5,legend=false); plot!(median(PP[spl_out.sigma],2),color="black",lw=1.5)

plot(PP_data,lw=1.5,legend=false); plot!(mean(PP[spl_out.sigma],2),color="black",lw=1.5)
