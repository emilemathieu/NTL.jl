using Distributions

function generateInterarrivalTimes(TK::Char, N::Int, interarrival_dist::DiscreteDistribution)
  ia_dist = (x,y) -> interarrival_dist
  generateInterarrivalTimes(TK,N,ia_dist)
end

function generateInterarrivalTimes(TK::Char, N::Int, interarrival_dist::Function)
    """
    - `TK`: 'T' if `N` is the total number of observations;
        'K' if `N` is the total number of arrivals
    - `N`: number of arrival times to generate (modulated by `TK`)
    - `interarrival`: distribution object to generate i.i.d. interarrivals
    """

    # check function arguments
    TK != 'K' && TK != 'T' ? error("`TK` must be 'T' or 'K'") : nothing

    zero_shift = Int(minimum(interarrival_dist(1,1)) == 0)

    if TK == 'K'
      T = zeros(Int64,N)
      T[1] = 1 # first arrival time is always 1
      for j in 2:N
        T[j] = rand(interarrival_dist(T[j-1],j-1)) + T[j-1] + zero_shift
      end
      # return T
    else
      T = [1]
      j = 1
      while T[j] < N
        j += 1
        append!(T,rand(interarrival_dist(T[j-1],j-1)) + T[j-1] + zero_shift)
      end
      if T[end] > N
        pop!(T)
      end
      # return T
    end
    return T
end

function generateLabelSequence(N::Int, alpha::Float64,
        interarrival_dist::DiscreteDistribution)
    ia_dist = (x,y) -> interarrival_dist
    generateLabelSequence(N,alpha,ia_dist)
end

function generateLabelSequence(N::Int, alpha::Float64,
        interarrival_dist::Function)
    """
    - `N`: number of observations in the sequence
    - `alpha`: 'discount parameter' in size-biased reinforcement
    - `interarrival_dist`: distribution object to generate interarrivals
    """
    Z = zeros(Int, N) # sequence of labels
    T = generateInterarrivalTimes('T', N, interarrival_dist)
    K = size(T,1) # number of clusters
    PP = zeros(Int, K) # arrival-ordered partition counts

    k = 0
    for n in 1:N
      if n <= T[end] && n == T[k+1]
        k += 1
        PP[k] = 1
        Z[n] = k
        k > K ? k = K : nothing
      else
        Z[n] = wsample(1:k, PP[1:k] .- alpha) # discounted size-biased sample
        PP[Z[n]] += 1
      end
    end
    return Z, PP, T
end

function generatePsis(T::Vector{Int},alpha::Float64)
    """
    - `T`: Arrival times
    - `alpha`: 'discount' parameter
    """
    K = size(T,1)
    Psi = zeros(Float64,K)
    Psi[1] = 1
    for j in 2:K
      Psi[j] = rand(Beta(1 - alpha, T[j] - 1 - (j-1)*alpha))
    end
    return Psi
end

function generateDataset(N::Int, D::Int, a::Float64, alpha::Float64,
        cluster_creator::Function, emission::Function, etype::Type)
    """
    - `N`: number of observations/documents
    - `D`: emission dimension
    - `a`: Geometric parameter for inter-arrival times
    - `alpha`: 'discount parameter' in size-biased reinforcement
    - `cluster_creator`: function generating random clusters
    - `emission`: function generating random emission given a cluster
    """
    z = zeros(Int, N)
    X = zeros(etype, N, D)

    # Sample arrival times
    T = 1
    Ts = [T]
    while T < N
        geo = rand(Geometric(a)) + 1
        T += geo
        push!(Ts, T)
    end

    # Sample observations/documents
    thetas = zeros(Float64, length(Ts)+1, D)
    K = 1
    nk = zeros(Int, K)
    for n in 1:N
        if Ts[K] == n
            # Sample and assign to a new cluster
            K += 1
            thetas[K, :] = cluster_creator()
            push!(nk, 0)
            z[n] = K
        else
            # Choose an existing cluster
            w = (nk - alpha) ./ (n - 1 - alpha*K)
            z[n] = wsample(1:K, w)
        end
        nk[z[n]] += 1
        X[n, :] = emission(thetas[z[n], :])
    end

    return z, X
end

function generateDirDataset(N::Int, D::Int, n_x::Int, a::Float64,
        alpha::Float64, dir_prior_param::Vector)
    # N: number of observations/documents
    # D: size of vocabulary
    # n_x: number of word per document
    # a: Geometric parameter for inter-arrival times
    # alpha: Neutral to the left parameter
    # dir_prior_param: Dirichlet's parameter
    @assert length(dir_prior_param) == D

    cluster_creator = () -> rand(Dirichlet(dir_prior_param))
    emission = (cluster) -> rand(Multinomial(n_x, cluster))

    z, X = generateDataset(N, D, a, alpha, cluster_creator, emission, Int)

    return z, sparse(X)
end

function generateGaussianDataset(N::Int, D::Int, a::Float64, alpha::Float64,
        sigma2::Float64, sigma2_observe::Float64)
    # N: number of observations/documents
    # D: size of vocabulary
    # a: Geometric parameter for inter-arrival times
    # alpha: Neutral to the left parameter
    # sigma: Variance of cluster creation
    # sigma_observe: Variance of emission

    cluster_creator = () -> rand(MvNormal(zeros(D), sqrt(sigma2)))
    emission = (cluster) -> rand(MvNormal(cluster, sqrt(sigma2_observe)))

    return generateDataset(N, D, a, alpha, cluster_creator, emission, Float64)
end

function generateDriftingGaussian(N::Int, D::Int, a::Float64, alpha::Float64,
        sigma2::Float64, sigma2_observe::Float64, drift::Vector{Float64})
    # N: number of observations/documents
    # D: size of vocabulary
    # a: Geometric parameter for inter-arrival times
    # alpha: Neutral to the left parameter
    # sigma: Variance of cluster creation
    # sigma_observe: Variance of emission
    # drift: The drift in the mean (starts at 0)

    cluster_mean = zeros(D)
    cluster_creator = function make_drift_cluster()
        cluster_mean += drift
        return rand(MvNormal(cluster_mean, sqrt(sigma2)))
    end
    emission = (cluster) -> rand(MvNormal(cluster, sqrt(sigma2_observe)))

    return generateDataset(N, D, a, alpha, cluster_creator, emission, Float64)
end
