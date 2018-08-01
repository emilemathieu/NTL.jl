# Utilities for Gibbs sampling with geometric inter-arrivals

function update_geometric_interarrival_param!(p::Vector{Float64},PP::Vector{Int},T::Vector{Int},n::Int,params::Vector{Float64})
    """
    - `p`: current geometric parameter for the interarrival time distribution (will be updated in place)
    - `PP`: partition/degree vector
    - `T`: arrival times vector
    - `n`: number of observations
    - `params`: parameters of prior Beta distribution
    """
    K = size(T,1)
    update_geometric_interarrival_param!(p,K,n,params)

end

function update_geometric_interarrival_param!(p::Vector{Float64},K::Int,n::Int,params::Vector{Float64})
    """
    - `p`: current geometric parameter for the interarrival time distribution (will be updated in place)
    - `K`: number of blocks in partition (=# of arrivals)
    - `n`: number of observations
    - `params`: parameters of prior Beta distribution
    """
    # sample conjugate p
    p[1] = rand(Beta(K-1+params[1],n-K+params[2]))

end
