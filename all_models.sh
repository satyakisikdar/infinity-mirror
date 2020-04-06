models="CNRG HRG SBM BTER ChungLu ErdosRenyi GraphVAE"  # space separated names of models, find the whole list in main.py
graph='input/eucore.g'  # external edge list 
# graph='clique_ring 500 4'  # for synthetic graphs - find list in Synthetic Graph class in src/graphio.py 
for model in $models
do
    # t is #independent trials, c is #cores, s is selection strategy, -p makes it use pickles to prevent repeated computations
    echo python3 main.py -i $graph -m $model -t 50 -n 20 -c 10 -s fast -p      
    python3 main.py -i $graph -m $model -t 50 -n 20 -c 10 -s fast -p
done
