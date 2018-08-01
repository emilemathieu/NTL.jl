# BetaNeutralToTheLeft.jl
Code to accompany UAI 2018 paper ['Sampling and Inference for Beta Neutral-to-the-Left Models of Sparse Networks'](http://auai.org/uai2018/proceedings/papers/185.pdf)  (['supplement'](http://auai.org/uai2018/proceedings/supplements/Supplementary-Paper185.pdf)).

## Data

### SNAP datasets

The [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/#temporal) contains a number of interesting temporal networks.
We recommend preprocessing the datasets as follows:

    wget https://snap.stanford.edu/data/$NAME.txt.gz
    gunzip $NAME.txt.gz
    sort -k3 -n $NAME.txt > sorted-$NAME.txt


## Maximum likelihood parameter estimation

See `examples/mle.jl` for an example of computing MLEs on massive datasets.


## Gibbs sampling arrival order, arrival times, parameters

See `examples/gibbs.jl` for an example of performing posterior inference over parameters
and latent variables on datasets of modest size (e.g., hundreds or thousands of nodes).
The code can be run interactively (i.e., section by section) or as a script
from the Julia REPL or command line.

See `examples/gibbs_plots.jl` for some example plots for assessing sampler output.

See `examples/gibbs_ess_experiments.jl` for code used to produce the tables in Section 5.1
of the paper.
