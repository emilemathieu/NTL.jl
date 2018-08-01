using DataStructures

function parseSnapData(fname::String)
    PP, T, PP_test, T_test, n_test = trainTestSplitSnapData(fname, 1.0)
    return PP, T
end

function trainTestSplitSnapData(fname::String, split_prop::Float64=0.8)
    N = countlines(open(fname))
    split_n = Int(round(N*split_prop))
    n_test = N - split_n

    f = open(fname)
    arrival_times = OrderedDict{String, Int}()
    degrees = OrderedDict{String, Int}()
    #scope
    T = nothing
    PP = nothing
    for (i, ln) in enumerate(eachline(f))
        a = split(ln)
        start = a[1]
        terminal = a[2]
        if ~haskey(arrival_times, start)
            arrival_times[start] = 2*i - 1
            degrees[start] = 1
        else
            degrees[start] += 1
        end
        if ~haskey(arrival_times, terminal)
            arrival_times[terminal] = 2*i
            degrees[terminal] = 1
        else
            degrees[terminal] += 1
        end
        if i == split_n
            T = collect(values(arrival_times))
            PP = collect(values(degrees))
        end
    end
    # T_test is a pure extension of T
    T_test = collect(values(arrival_times))
    # PP is a pure extension of PP
    PP_test = collect(values(degrees))
    return PP, T, PP_test, T_test, 2*split_n, 2*n_test
end


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
    - `alpha`: BNTL α
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
