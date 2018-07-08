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

