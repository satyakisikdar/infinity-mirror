models="GCN_AE"  # space separated names of models, find the whole list in main.py
graph='input/karate.g'  # external edge list 
for model in $models
do
    # -t: number of independent trials
    # -c: number of cores
    # -s: selection strategy
    # -f: use an incomplete file as a starting point
    # -p: use pickles to prevent repeated computations
    # -z: return the features the model learns
    # -l: whether or not to take only the largest connected component
    echo python3 main.py -i $graph -m $model -t 1 -n 1 -c 1 -s fast -p -z
    python3 main.py -i $graph -m $model -t 1 -n 1 -c 1 -s fast -p -z
done
# output graph pickles are stored in infinity-mirror/output/pickles directory
# output feature pickles are stored in infinity-mirror/output/features directory
