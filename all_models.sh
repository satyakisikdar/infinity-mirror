models="Linear_AE Linear_VAE Deep_GCN_AE Deep_GCN_VAE"  # space separated names of models, find the whole list in main.py
graph='input/eucore.g'  # external edge list 
# graph='clique_ring 500 4'  # for synthetic graphs - find list in Synthetic Graph class in src/graphio.py 
for model in $models
do
    # -t: number of independent trials
    # -c: number of cores
    # -s: selection strategy
    # -p: use pickles to prevent repeated computations
    echo python3 main.py -i $graph -m $model -t 50 -n 20 -c 1 -s fast -p      
    python3 main.py -i $graph -m $model -t 50 -n 20 -c 1 -s fast -p
done
# output pickles are stored in infinity-mirror/output/pickles directory
