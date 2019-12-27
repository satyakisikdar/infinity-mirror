"""
Container for different graph models
"""
import abc
import subprocess
import math
from time import time
from typing import List, Dict, Any, Union

import networkx as nx
import numpy as np

from src.utils import ColorPrint as CP
from src.utils import check_file_exists, load_pickle
from src.Graph import CustomGraph

__all__ = ['BaseGraphModel', 'ErdosRenyi', 'ChungLu', 'BTER', 'CNRG', 'HRG', 'Kronecker']

class BaseGraphModel:
    def __init__(self, model_name: str, input_graph: CustomGraph, **kwargs) -> None:
        self.input_graph: CustomGraph = CustomGraph(input_graph)  # networkX graph to be fitted
        assert self.input_graph.name != '', 'Input graph does not have a name'

        self.gname = self.input_graph.name  # name of the graph
        self.model_name: str = model_name  # name of the model
        self.params: Dict[Any] = {}  # dictionary of model parameters
        self.generated_graphs: List[CustomGraph] = []   # list of NetworkX graphs

        self._fit()  # fit the parameters initially

        return

    @abc.abstractmethod
    def _fit(self) -> None:
        """
        Fits the parameters of the model
        :return:
        """
        pass

    @abc.abstractmethod
    def _gen(self) -> CustomGraph:
        """
        Generates one graph
        """
        pass

    def update(self, new_input_graph: CustomGraph) -> None:
        """
        Update the model to (a) update the input graph, (b) fit the parameters
        :return:
        """
        CP.print_blue('Updating graph')

        self.input_graph = new_input_graph
        self._fit()  # re-fit the parameters

        return

    def generate(self, num_graphs: int, gen_id: int) -> None:
        """
        Generates num_graphs many graphs by repeatedly calling _gen
        maybe use a generator
        :param num_graphs:
        :param gen_id: generation id
        :return:
        """
        self.generated_graphs = []  # reset the list of graphs - TODO: maybe double check if this is necessary

        for i in range(num_graphs):
            g = self._gen()
            g.name = f'{self.input_graph.name}_{gen_id}_{i+1}'  # name of the generated graph - input graph name + gen id + iteration no
            g = CustomGraph(g, gen_id=gen_id)
            self.generated_graphs.append(g)

        return

    def __str__(self) -> str:
        st = f'name: "{self.model_name}", input_graph: "{self.input_graph.name}"'
        if len(self.params) > 0:
            st += f'params: {self.params}'
        return st

    def __repr__(self) -> str:
        return str(self)


class ErdosRenyi(BaseGraphModel):
    def __init__(self, input_graph: CustomGraph, **kwargs) -> None:
        super().__init__(model_name='Erdos-Renyi', input_graph=input_graph)
        if 'seed' in kwargs:
            seed = kwargs['seed']
        else:
            seed = None
        self.params['seed'] = seed
        return

    def _fit(self) -> None:
        """
        G(n, p)
        n: number of nodes
        p: probability of edges

        <m>: expected number of edges
        for fitting, p = <m> / (n * (n - 1) / 2)
        :return:
        """
        n = self.input_graph.order()
        m = self.input_graph.size()

        self.params['n'] = n
        self.params['p'] = m / (n * (n - 1) / 2)

        return

    def _gen(self) -> CustomGraph:
        assert 'n' in self.params and 'p' in self.params, 'Improper parameters for Erdos-Renyi'

        g = nx.erdos_renyi_graph(n=self.params['n'], p=self.params['p'], seed=self.params['seed'])

        return CustomGraph(g)

class ChungLu(BaseGraphModel):
    def __init__(self, input_graph: CustomGraph, **kwargs) -> None:
        super().__init__(model_name='Chung-Lu', input_graph=input_graph)
        return

    def _fit(self) -> None:
        self.params['degree_seq'] = sorted([d for n, d in self.input_graph.degree()], reverse=True)  # degree sequence

        return

    def _gen(self) -> CustomGraph:
        assert 'degree_seq' in self.params, 'imporper parameters for Chung-Lu'

        g = nx.configuration_model(self.params['degree_seq'])  # fit the model to the degree seq
        g = CustomGraph(g)  # make it into a simple graph
        g.remove_edges_from(nx.selfloop_edges(g))  # remove self-loops

        return g


class TransitiveChungLu(BaseGraphModel):
    """
    Chung-Lu with transitive closures - Pfeiffer, La Fond, Moreno, Neville - implementation not found
    https://ieeexplore.ieee.org/document/6406280/
    """
    def __init__(self, input_graph: CustomGraph, **kwargs) -> None:
        super().__init__(model_name='Transitive-Chung-Lu', input_graph=input_graph)
        return

    def _fit(self) -> None:
        raise NotImplementedError('Transitive Chung-Lu is not implemented yet')

    def _gen(self) -> CustomGraph:
        raise NotImplementedError('Transitive Chung-Lu is not implemented yet')


class BTER(BaseGraphModel):
    """
    BTER model by Tammy Kolda
    feastpack implementation at https://www.sandia.gov/~tgkolda/feastpack/feastpack_v1.2.zip
    """
    def __init__(self, input_graph: CustomGraph, **kwargs) -> None:
        super().__init__(model_name='BTER', input_graph=input_graph)
        return

    def _fit(self) -> None:
        pass  # the matlab code does the fitting

    def _gen(self) -> Union[CustomGraph, None]:
        g = self.input_graph

        # fix BTER to use the directory..
        CP.print_blue('Starting BTER... Checking for MATLAB.')

        completed_process = subprocess.run('matlab -h', shell=True, stdout=subprocess.DEVNULL)
        assert completed_process.returncode != 0, 'MATLAB not found'

        np.savetxt('./src/bter/{}.mat'.format(g.name), nx.to_numpy_matrix(g), fmt='%d')

        matlab_code = [
            "mex -largeArrayDims tricnt_mex.c",
            "mex -largeArrayDims ccperdegest_mex.c",
            f"G = dlmread('{g.name}.mat');",
            'G = sparse(G);',
            f"graphname = '{g.name}';",
            '',
            'nnodes = size(G, 1);',
            'nedges = nnz(G) / 2;',
            r"fprintf('nodes: %d edges: %d\n', nnodes, nedges);",
            '',
            'nd = accumarray(nonzeros(sum(G,2)),1);',
            "maxdegree = find(nd>0,1,'last');",
            r"fprintf('Maximum degree: %d\n', maxdegree);",
            '',
            '[ccd,gcc] = ccperdeg(G);',
            r"fprintf('Global clustering coefficient: %.2f\n', gcc);",
            '',
            r"fprintf('Running BTER...\n');",
            't1=tic;',
            '[E1,E2] = bter(nd,ccd);',
            'toc(t1);',
            r"fprintf('Number of edges created by BTER: %d\n', size(E1,1) + size(E2,1));",
            '',
            "fprintf('Turning edge list into adjacency matrix (including dedup)...');",
            't2=tic;',
            'G_bter = bter_edges2graph(E1,E2);',
            'toc(t2);',
            r"fprintf('Number of edges in dedup''d graph: %d\n', nnz(G)/2);",
            '',
            'G_bter = full(G_bter);',
            r"dlmwrite('{}_bter.mat', G_bter, ' ');".format(g.name)
            ]

        print('\n'.join(matlab_code), file=open('./src/bter/{}_code.m'.format(g.name), 'w'))

        if not check_file_exists(f'./bter/{g.name}_bter.mat'):
            start_time = time()
            completed_process =  subprocess.run('cd src/bter; cat {}_code.m | matlab -nosplash -nodesktop'.format(g.name), shell=True)
            print('BTER ran in {} secs'.format(round(time() - start_time, 3)))

            if completed_process.returncode != 0:
                print('error in matlab')
                return None

        assert check_file_exists(f'./src/bter/{g.name}_bter.mat'), 'MATLAB did not write a graph'
        bter_mat = np.loadtxt(f'./src/bter/{g.name}_bter.mat', dtype=int)

        g_bter = nx.from_numpy_matrix(bter_mat, create_using=CustomGraph())

        return g_bter


class CNRG(BaseGraphModel):
    """
    Satyaki's Clustering-Based Node Replacement Grammars https://github.com/satyakisikdar/cnrg
    """
    def __init__(self, input_graph: CustomGraph, **kwargs) -> None:
        super().__init__(model_name='CNRG', input_graph=input_graph)
        return

    def _fit(self) -> None:
        pass  # the Python code does the fitting

    def _gen(self) -> None:
        pass  # HRGs can generate multiple graphs at once

    def generate(self, num_graphs: int, gen_id:int) -> None:
        nx.write_edgelist(self.input_graph, f'./src/cnrg/src/tmp/{self.gname}.g', data=False)

        completed_process = subprocess.run(f'cd src/cnrg; python3 runner.py -g {self.gname} -n {num_graphs}',
                                           shell=True)
        assert completed_process.returncode == 0, 'Error in CNRG'
        output_pickle_path = f'./src/cnrg/output/{self.gname}_cnrg.pkl'
        assert check_file_exists(output_pickle_path)

        generated_graphs = load_pickle(output_pickle_path)
        self.generated_graphs = []  # reset generated graphs

        for i, gen_graph in enumerate(generated_graphs):
            gen_graph.name = self.input_graph.name
            gen_graph = CustomGraph(gen_graph)
            gen_graph.gen_id = gen_id
            gen_graph.name += f'_{i+1}'  # append the number of graph generated
            self.generated_graphs.append(gen_graph)

        assert isinstance(self.generated_graphs, list) and len(self.generated_graphs) == num_graphs, \
            'Failed to generate graphs'
        return


class HRG(BaseGraphModel):
    """
    Sal's Hyperedge Replacement Graph Grammars https://github.com/abitofalchemy/hrg-nm
    """
    def __init__(self, input_graph: CustomGraph, **kwargs) -> None:
        super().__init__(model_name='HRG', input_graph=input_graph)
        return

    def _fit(self) -> None:
        pass  # the Python code does the fitting

    def _gen(self) -> None:
        pass  # HRGs can generate multiple graphs at once

    def _make_graph(self, graph) -> CustomGraph:
        """
        This is needed since HRGs use NetworkX 1.x and that's incompatible with 2.x
        :param graph:
        :return:
        """
        custom_g = CustomGraph()
        custom_g.name = graph.name

        for u, nbrs in graph.edge.items():
            for v in nbrs.keys():
                custom_g.add_edge(u, v)
        return custom_g

    def generate(self, num_graphs: int, gen_id: int) -> None:
        nx.write_edgelist(self.input_graph, f'./src/hrg/{self.gname}.g', data=False)

        completed_process = subprocess.run(f'cd src/hrg; python2 -m pip install networkx==1.11; python2 exact_phrg.py --orig {self.gname}.g --trials {num_graphs}',
                                           shell=True)

        assert completed_process.returncode == 0, 'Error in HRG'

        output_pickle_path = f'./src/hrg/Results/{self.gname}_hstars.pickle'
        generated_graphs = load_pickle(output_pickle_path)

        self.generated_graphs = []
        for i, gen_graph in enumerate(generated_graphs):
            gen_graph.name = self.input_graph.name
            gen_graph = self._make_graph(gen_graph)
            gen_graph.gen_id = gen_id
            gen_graph.name += f'_{i+1}'  # adding the number of graph
            self.generated_graphs.append(gen_graph)

        assert isinstance(self.generated_graphs, list) and len(self.generated_graphs) == num_graphs, \
            'Failed to generate graphs'

        return


class Kronecker(BaseGraphModel):
    """
    Kronecker Graph Model from SNAP
    """
    def __init__(self, input_graph: CustomGraph, **kwargs) -> None:
        super().__init__(model_name='Kronecker', input_graph=input_graph)
        return

    def _fit(self) -> None:
        """
        call KronEM
        """
        # raise NotImplementedError('Check if KronEM can write in the right place')
        output_file = f'./src/snap/examples/graphs/{self.gname}-fit'
        if not check_file_exists(output_file):  # run kronem only if file does not exist
            CP.print_green(f'Running KronEM for {self.gname}')

            # write edgelist to the path, but graph needs to start from 1
            g = nx.convert_node_labels_to_integers(self.input_graph, first_label=1, label_attribute='old_label')
            directed_g = g.to_directed()  # kronecker expects a directed graph
            nx.write_edgelist(directed_g, f'src/snap/examples/graphs/{self.gname}.txt', data=False)

            bash_code = f'cd src/snap/examples/kronem; make; ./kronem -i:../graphs/{self.gname}.txt -o:../graphs/                                                                        {self.gname}-fit'
            completed_process = subprocess.run(bash_code, shell=True)#, stdout=subprocess.PIPE)
            assert completed_process.returncode == 0, 'Error in KronEM'

        assert check_file_exists(output_file), f'File does not exist {output_file}'

        with open(output_file) as f:
            last_line = f.readlines()[-1]
            last_line = last_line.replace(']', '')
            matrix = last_line[last_line.find('[') + 1: ]
            CP.print_blue('Initiator matrix:', matrix)
            self.params['initiator_matrix'] = matrix

        return

    def _gen(self) -> CustomGraph:
        """
        call KronGen
        """
        assert 'initiator_matrix' in self.params, 'Initiator matrix not found'

        orig_n = self.input_graph.order()

        kron_iters = int(math.log2(orig_n))  # floor of log2 gives a bound on kronecker iteration count
        if math.fabs(2**kron_iters - orig_n) > math.fabs(2**(kron_iters+1) - orig_n):
            kron_iters = kron_iters + 1
        matrix = self.params['initiator_matrix']
        CP.print_blue(f'Running kronGen with n={kron_iters}, matrix={matrix}')

        bash_code = f'cd src/snap/examples/krongen; make; ./krongen -o:../graphs/{self.gname}_kron.txt -m:"{matrix}" -i:{kron_iters}'
        completed_process = subprocess.run(bash_code, shell=True)  # , stdout=subprocess.PIPE)
        assert completed_process.returncode == 0, 'Error in KronGen'

        output_file = f'src/snap/examples/graphs/{self.gname}_kron.txt'
        assert check_file_exists(output_file), f'Output file does not exist {output_file}'

        graph = nx.read_edgelist(output_file, nodetype=int, create_using=CustomGraph())
        return graph


class ForestFire(BaseGraphModel):
    """
    Forest fire model from SNAP
    """


class AGM(BaseGraphModel):
    """
    Affiliation Graph Model from SNAP
    """


class StochasticBlockModel(BaseGraphModel):
    """
    Stochastic Block Model  - basic and degree corrected
    """


class RMAT(BaseGraphModel):
    """
    Recursive-matrix graph generators
    """


class ERGM(BaseGraphModel):
    """
    Exponential Random Graph Models
    """


class GraphNeuralNet(BaseGraphModel):
    """
    Graph Neural Network Based Models
    """

