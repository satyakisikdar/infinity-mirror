models="Kronecker CNRG ChungLu SBM BUGGE HRG ErdosRenyi"  # space separated names of models, find the whole list in main.py
graph=$1  # external edge list 
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
    echo nice -n 10 python3 main.py -i $graph -m $model -t 5 -n 10 -c 5 -s fast -p -l
    nice -n 10 python main.py -i $graph -m $model -t 5 -n 10 -c 5 -s fast -p -l
done
# output pickles are stored in infinity-mirror/output/pickles directory
