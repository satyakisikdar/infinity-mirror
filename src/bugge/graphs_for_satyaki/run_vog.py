from time import time 
import os
import subprocess
import networkx as nx
from sys import argv 

def read_graph(fname):
    g = nx.read_edgelist(fname, nodetype=int, create_using=nx.Graph())
    name = fname.split('/')[-1]
    name = name[: name.find('.g')]
    g.name = name
    print(f'Read {name}, n = {g.order()}, m = {g.size()}')
    return g
    
def vog(g):
    print('Starting VoG....')
    
    assert len(g) > 1, 'graph must have a name'
    name = g.name
    g = nx.convert_node_labels_to_integers(g, first_label=1)

    with open('../vog/DATA/{}.g'.format(name), 'w') as f:
        for u, v in g.edges():
            f.write('{},{},1\n'.format(u, v))

    matlab_code = []
    matlab_code.append("addpath('DATA');")
    matlab_code.append("addpath('STRUCTURE_DISCOVERY');")
    matlab_code.append("input_file = './DATA/{}.g';".format(name))

    matlab_code.append("unweighted_graph = input_file;")
    matlab_code.append("output_model_greedy = 'DATA';")
    matlab_code.append("output_model_top10 = 'DATA';")
        
    matlab_code.append("orig = spconvert(load(input_file));")
    matlab_code.append("orig(max(size(orig)),max(size(orig))) = 0;")
    matlab_code.append("orig_sym = orig + orig';")
    matlab_code.append("[i,j,k] = find(orig_sym);")

    matlab_code.append("orig_sym(i(find(k==2)),j(find(k==2))) = 1;")
    matlab_code.append("orig_sym_nodiag = orig_sym - diag(diag(orig_sym));")

    matlab_code.append("disp('==== Running VoG for structure discovery ====')")
    matlab_code.append("global model;")
    matlab_code.append("model = struct('code', {}, 'edges', {}, 'nodes1', {}, 'nodes2', {}, 'benefit', {}, 'benefit_notEnc', {});")
    matlab_code.append("global model_idx;")
    matlab_code.append("model_idx = 0;")
    matlab_code.append("SlashBurnEncode( orig_sym_nodiag, 2, output_model_greedy, false, false, 3, unweighted_graph);")

    matlab_code.append("quit;")

    print('\n'.join(matlab_code), file=open('../vog/{}_vog_code.m'.format(name), 'w'))

    start_time = time()

    completed_process = subprocess.run('cd ../vog; cat {}_vog_code.m | matlab'.format(name), shell=True)
    completed_process = subprocess.run(f'cd ../vog; ./run_vog.bash {g.name}', shell=True)
    

    print('VoG ran in {} secs'.format(round(time() - start_time, 3)))

    if completed_process.returncode != 0:
        print('error in matlab')
        return None


def main():
    if len(argv) < 2:
        print('Provide path to edge list')
        return 
    g = read_graph(argv[1])
    vog(g)


if __name__ == '__main__':
    main()
