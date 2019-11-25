'''
Container for different graph models
'''
import networkx as nx
from typing import List, Dict, Any, Union
import abc
import numpy as np
import subprocess
from time import time

from src.utils import check_file_exists
from src.utils import ColorPrint as CP


class BaseGraphModel:
    def __init__(self, model_name: str, input_graph: nx.Graph):
        self.input_graph: nx.Graph = input_graph  # networkX graph to be fitted
        self.model_name = model_name  # name of the model
        self.params: Dict[Any] = {}  # dictionary of model parameters
        self.generated_graphs: List[nx.Graph] = []   # list of NetworkX graphs

    @abc.abstractmethod
    def _fit(self):
        pass

    @abc.abstractmethod
    def _gen(self) -> nx.Graph:
        """
        Generates one graph
        """
        pass

    def generate(self, num_graphs: int):
        for _ in range(num_graphs):
            g = self._gen()
            self.generated_graphs.append(g)


    def __str__(self):
        return f'name: {self.model_name}, input_graph: {self.input_graph.name}, params: {self.params}'

    def __repr__(self):
        return str(self)


class ErdosRenyi(BaseGraphModel):
    def __init__(self, input_graph: nx.Graph):
        super().__init__(model_name='Erdos-Renyi', input_graph=input_graph)

    def _fit(self):
        n = self.input_graph.order()
        m = self.input_graph.size()

        self.params['n'] = n
        self.params['p'] = m / (n * (n - 1) / 2)

    def _gen(self) -> nx.Graph:
        self._fit()
        assert 'n' in self.params and 'p' in self.params, 'Improper parameters for Erdos-Renyi'
        return nx.erdos_renyi_graph(n=self.params['n'], p=self.params['p'])


class ChungLu(BaseGraphModel):
    def __init__(self, input_graph: nx.Graph):
        super().__init__(model_name='Chung-Lu', input_graph=input_graph)

    def _fit(self):
        self.params['degree_seq'] = sorted(deg for node, deg in self.input_graph.degree())

    def _gen(self) -> nx.Graph:
        self._fit()
        assert 'degree_seq' in self.params, 'imporper parameters for Chung-Lu'

        g = nx.configuration_model(self.params['degree_seq'])  # fit the model to the degree seq
        g = nx.Graph(g)  # make it into a simple graph
        g.remove_edges_from(nx.selfloop_edges(g))  # remove self-loops

        return g


class TransitiveChungLu(BaseGraphModel):
    """
    Chung-Lu with transitive closures - Pfeiffer, La Fond, Moreno, Neville - implementation not found
    https://ieeexplore.ieee.org/document/6406280/
    """
    def __init__(self, input_graph: nx.Graph):
        super().__init__(model_name='Transitive-Chung-Lu', input_graph=input_graph)

    def _fit(self):
        raise(NotImplementedError)

    def _gen(self) -> nx.Graph:
        raise(NotImplementedError)


class BTER(BaseGraphModel):
    """
    BTER model by Tammy Kolda
    feastpack implementation at https://www.sandia.gov/~tgkolda/feastpack/feastpack_v1.2.zip
    """
    def __init__(self, input_graph: nx.Graph):
        super().__init__(model_name='BTER', input_graph=input_graph)

    def _fit(self):
        pass  # the matlab code does the fitting

    def _gen(self) -> Union[nx.Graph, None]:
        g = self.input_graph

        # fix BTER to use the directory..
        CP.print_blue('Starting BTER... Checking for MATLAB.')

        completed_process = subprocess.run('matlab -h', shell=True)
        assert completed_process.returncode != 0, 'MATLAB not found'

        assert g.name != '', 'Graph name cannot be blank'
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

        g_bter = nx.from_numpy_matrix(bter_mat, create_using=nx.Graph())
        return g_bter
