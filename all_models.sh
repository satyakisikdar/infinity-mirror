models="NetGAN"  # "Kronecker CNRG ChungLu SBM"  # space separated names of models, find the whole list in main.py
graph='input/3-comm-new.g'  # external edge list 
# graph='clique_ring 500 4'  # for synthetic graphs - find list in Synthetic Graph class in src/graphio.py 
for model in $models
do
    # -t: number of independent trials
    # -c: number of cores
    # -s: selection strategy
    # -f: use an incomplete file as a starting point
    # -p: use pickles to prevent repeated computations
    # -z: return the features the model learns
    # -l: whether or not to take only the largest connected component
    echo python3 main.py -i $graph -m $model -t 1 -n 10 -c 1 -s fast -p -l
    python3 main.py -i $graph -m $model -t 1 -n 10 -c 1 -s fast -p -l
done
# output pickles are stored in infinity-mirror/output/pickles directory
