### Slice Sampling routine
### code adapted from
### Michael Lindon A Functional Implementation of Slice Sampling in Julia
### https://michaellindon.github.io/julia/slice-sampling/


function slice_sampling(g,w::Float64,x::Float64)
    function lowerbound(L)
        g(L)<y ? L : lowerbound(L-w)
    end
    function upperbound(R)
        g(R)<y ? R : upperbound(R+w)
    end
    function shrinksample(L,R)
        z=rand(Uniform(L,R))
        if g(z)>y
            z
        elseif z>x
            shrinksample(L,z)
        else
            shrinksample(z,R)
        end
    end
    y=-1*rand(Exponential(1))+g(x)
    U=rand(Uniform(0,1))
    shrinksample(lowerbound(x-U*w),upperbound(x+(1-U)*w))
end
