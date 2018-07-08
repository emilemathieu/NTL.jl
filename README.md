# BetaNeutralToTheLeft.jl
Code to accompany UAI paper ['Sampling and Inference for Beta Neutral-to-the-Left Models of Sparse Networks'](http://www.stats.ox.ac.uk/~bloemred/assets/papers/bntl-uai.pdf).

## Data

### SNAP datasets

The [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/#temporal) contains a number of interesting temporal networks.
We recommend preprocessing the datasets as follows

    wget https://snap.stanford.edu/data/$NAME.txt.gz
    gunzip $NAME.txt.gz
    sort -k3 -n $NAME.txt > sorted-$NAME.txt


## Maximum likelihood parameter estimation

See `examples/mle.jl` for an example of computing MLEs on massive datasets.

