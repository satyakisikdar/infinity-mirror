"""
Container for different graph models
"""
import abc
import math
import platform
import subprocess as sub
from time import time
from typing import List, Dict, Any, Union

import networkx as nx
import numpy as np

from src.cnrg.runner import get_grammar
from src.cnrg.src.generate import generate_graph
from src.gae.fit import fit_vae
from src.utils import ColorPrint as CP
from src.utils import check_file_exists, load_pickle, delete_files, get_blank_graph, get_graph_from_prob_matrix

__all__ = ['BaseGraphModel', 'ErdosRenyi', 'UniformRandom', 'ChungLu', 'BTER', 'CNRG', 'HRG', 'Kronecker']


class BaseGraphModel:
    __slots__ = ['input_graph', 'initial_gname', 'model_name', 'params', 'run_id']

    def __init__(self, model_name: str, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        self.input_graph: nx.Graph = input_graph  # networkX graph to be fitted
        assert self.input_graph.name != '', 'Input graph does not have a name'

        self.initial_gname: str = input_graph.name  # name of the initial graph
        self.model_name: str = model_name  # name of the model
        self.run_id = run_id  # run id prevents files from getting clobbered
        self.params: Dict[Any] = {}  # dictionary of model parameters

        return

    @abc.abstractmethod
    def _fit(self) -> None:
        """
        Fits the parameters of the model
        :return:
        """
        pass

    @abc.abstractmethod
    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        """
        Generates one graph with given gname and gen_id
        """
        pass

    def update(self, new_input_graph: nx.Graph) -> None:
        """
        Update the model to (a) update the input graph, (b) fit the parameters
        :return:
        """
        CP.print_none('Updating graph')

        self.input_graph = new_input_graph
        self._fit()  # re-fit the parameters

        return

    def generate(self, num_graphs: int, gen_id: int) -> List[nx.Graph]:
        """
        Generates num_graphs many graphs by repeatedly calling _gen
        maybe use a generator
        :param num_graphs:
        :param gen_id: generation id
        :param run_id: run_id keeps things separate when run in parallel
        :return:
        """
        # generated_graphs = Parallel(n_jobs=3)(
        #     delayed(self._gen)(gen_id=gen_id, gname=f'{self.gname}_{gen_id}_{self.run_id}_{i + 1}')
        #     for i in range(num_graphs)
        # )
        generated_graphs = [self._gen(gen_id=gen_id, gname=f'{self.input_graph.name}_{gen_id}_{self.run_id}_{i + 1}')
                            for i in range(num_graphs)]

        assert isinstance(generated_graphs, list), 'Parallel generation didnt work'

        return generated_graphs

    def __str__(self) -> str:
        st = f'name: "{self.model_name}", input_graph: "{self.input_graph.name}"'
        if len(self.params) > 0:
            st += f'params: {self.params}'
        return st

    def __repr__(self) -> str:
        return str(self)


class ErdosRenyi(BaseGraphModel):
    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='Erdos-Renyi', input_graph=input_graph, run_id=run_id)
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

    def _gen(self, gname: str, gen_id: int, ) -> nx.Graph:
        assert 'n' in self.params and 'p' in self.params, 'Improper parameters for Erdos-Renyi'

        g = nx.fast_gnp_random_graph(n=self.params['n'], p=self.params['p'], seed=self.params['seed'])
        g.name = gname
        g.gen_id = gen_id

        return g


class UniformRandom(BaseGraphModel):
    """
    model, a graph is chosen uniformly at random from the set of all graphs with n nodes and m edges.
    """

    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='Uniform-Random', input_graph=input_graph, run_id=run_id)
        if 'seed' in kwargs:
            seed = kwargs['seed']
        else:
            seed = None
        self.params['seed'] = seed
        return

    def _fit(self):
        n = self.input_graph.order()
        m = self.input_graph.size()

        self.params['n'] = n
        self.params['m'] = m

        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        assert 'n' in self.params and 'm' in self.params, 'Improper parameters for Uniform Random'

        g = nx.gnm_random_graph(n=self.params['n'], m=self.params['m'], seed=self.params['seed'])
        g.name = gname
        g.gen_id = gen_id

        return g


class ChungLu(BaseGraphModel):
    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='Chung-Lu', input_graph=input_graph, run_id=run_id)
        return

    def _fit(self) -> None:
        self.params['degree_seq'] = sorted([d for n, d in self.input_graph.degree()], reverse=True)  # degree sequence

        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        assert 'degree_seq' in self.params, 'imporper parameters for Chung-Lu'

        try:
            g = nx.configuration_model(self.params['degree_seq'])  # fit the model to the degree seq

        except nx.NetworkXError:  # config model failed
            g = get_blank_graph(gname)
            gname = 'blank_' + gname  # add blank to the name

        else:  # gets called only if the exception is not thrown
            g = nx.Graph(g)  # make it into a simple graph
            g.remove_edges_from(nx.selfloop_edges(g))  # remove self-loops

        g.name = gname
        g.gen_id = gen_id

        return g


class BTER(BaseGraphModel):
    """
    BTER model by Tammy Kolda
    feastpack implementation at https://www.sandia.gov/~tgkolda/feastpack/feastpack_v1.2.zip
    """

    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='BTER', input_graph=input_graph, run_id=run_id)
        # self.prep_environment()
        return

    def _fit(self) -> None:
        pass  # the matlab code does the fitting

    def prep_environment(self) -> None:
        """
        Prepare environment - check for MATLAB
        :return:
        """
        completed_process = sub.run('matlab -h', shell=True, stdout=sub.DEVNULL)
        assert completed_process.returncode != 0, 'MATLAB not found'
        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        g = self.input_graph

        # fix BTER to use the directory..
        CP.print_blue('Starting BTER...')

        graph_path = f'./src/bter/{g.name}_{self.run_id}.mat'
        np.savetxt(graph_path, nx.to_numpy_matrix(g), fmt='%d')

        matlab_code = [
            'mex -largeArrayDims tricnt_mex.c;',
            'mex -largeArrayDims ccperdegest_mex.c;',
            f"G = dlmread('{g.name}_{self.run_id}.mat');",
            'G = sparse(G);',
            f"graphname = '{g.name}_{self.run_id}';",
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
            r"dlmwrite('{}_{}_bter.mat', G_bter, ' ');".format(g.name, self.run_id),
            'quit;'
        ]

        matlab_code_filename = f'{g.name}_{self.run_id}_code.m'
        matlab_code_path = f'./src/bter/{matlab_code_filename}'

        print('\n'.join(matlab_code), file=open(matlab_code_path, 'w'))

        output_path = f'./src/bter/{g.name}_{self.run_id}_bter.mat'

        start_time = time()
        completed_process = sub.run(f'cd src/bter; cat {matlab_code_filename} | matlab -nosplash -nodesktop',
                                    shell=True,
                                    stdout=sub.DEVNULL, stderr=sub.DEVNULL)
        CP.print_blue(f'BTER ran in {round(time() - start_time, 3)} secs')

        if completed_process.returncode != 0:
            CP.print_blue('BTER failed!')
            g_bter = get_blank_graph(gname)

        elif not check_file_exists(output_path):
            CP.print_blue('BTER failed!')
            g_bter = get_blank_graph(gname)

        else:
            bter_mat = np.loadtxt(output_path, dtype=int)
            g_bter = nx.from_numpy_matrix(bter_mat, create_using=nx.Graph())
            g_bter.name = gname

        g_bter.gen_id = gen_id
        delete_files(graph_path, output_path, matlab_code_path)

        return g_bter


class CNRG(BaseGraphModel):
    """
    Satyaki's Clustering-Based Node Replacement Grammars https://github.com/satyakisikdar/cnrg
    """

    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='CNRG', input_graph=input_graph, run_id=run_id)
        return

    def _fit(self) -> None:
        grammar = get_grammar(self.input_graph, name=self.input_graph.name)
        self.params['grammar'] = grammar
        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        assert 'grammar' in self.params, 'Improper params. Grammar object is missing.'

        light_g = generate_graph(target_n=self.input_graph.order(), rule_dict=self.params['grammar'].rule_dict)
        g = nx.Graph()
        g.add_edges_from(light_g.edges())
        g.name = gname
        g.gen_id = gen_id

        return g


class HRG(BaseGraphModel):
    """
    Sal's Hyperedge Replacement Graph Grammars https://github.com/abitofalchemy/hrg-nm
    """

    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='HRG', input_graph=input_graph, run_id=run_id)
        self.prep_environment()
        return

    def _fit(self) -> None:
        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        pass  # HRGs can generate multiple graphs at once

    def _make_graph(self, graph) -> nx.Graph:
        """
        This is needed since HRGs use NetworkX 1.x and that's incompatible with 2.x
        :param graph:
        :return:
        """
        custom_g = nx.Graph()
        custom_g.name = graph.name

        for u, nbrs in graph.edge.items():
            for v in nbrs.keys():
                custom_g.add_edge(u, v)
        return custom_g

    def prep_environment(self) -> None:
        """
        Prepare the Python environment
        :return:
        """
        if check_file_exists('./envs/hrg'):
            return

        CP.print_blue('Making virtual environment for HRG')
        sub.run(
            'python2 -m pip install --user virtualenv; python2 -m virtualenv -p python2 ./envs/hrg;. ./envs/hrg/bin/activate; which python2;',
            shell=True,
            stdout=sub.DEVNULL)  # create and activate environment
        if 'Linux' not in platform.platform():
            completed_process = sub.run(
                'export CC=gcc-9; export CXX=g++-9;. ./envs/hrg/bin/activate; python2 -m pip install -r ./envs/requirements_hrg.txt',
                shell=True, stdout=sub.DEVNULL)  # install requirements for cnrg

        else:
            completed_process = sub.run(
                '. ./envs/hrg/bin/activate; python2 -m pip install -r ./envs/requirements_hrg.txt',
                shell=True, stdout=sub.DEVNULL)  # install requirements for cnrg

        assert completed_process.returncode == 0, 'Error while creating environment for HRG'
        return

    def generate(self, num_graphs: int, gen_id: int) -> Union[List[nx.Graph], None]:
        edgelist_path = f'./src/hrg/{self.initial_gname}_{self.run_id}.g'
        nx.write_edgelist(self.input_graph, edgelist_path, data=False)
        output_pickle_path = f'./src/hrg/Results/{self.initial_gname}_{self.run_id}_hstars.pickle'

        completed_process = sub.run(
            f'. ./envs/hrg/bin/activate; cd src/hrg; python2 exact_phrg.py --orig {self.initial_gname}_{self.run_id}.g --trials {num_graphs}; deactivate;',
            shell=True, stdout=sub.DEVNULL)

        if completed_process.returncode != 0:
            CP.print_blue(f'Error in HRG: "{self.input_graph.name}"')
            generated_graphs = None

        elif not check_file_exists(output_pickle_path):
            CP.print_blue(f'Error in HRG: "{self.input_graph.name}"')
            generated_graphs = None

        else:
            generated_graphs = []
            for i, gen_graph in enumerate(load_pickle(output_pickle_path)):
                gen_graph = self._make_graph(gen_graph)
                gen_graph.name = f'{self.input_graph.name}_{self.run_id}_{i + 1}'  # adding the number of graph
                gen_graph.gen_id = gen_id

                generated_graphs.append(gen_graph)

            assert isinstance(generated_graphs, list) and len(generated_graphs) == num_graphs, \
                'Failed to generate graphs'

        delete_files(edgelist_path, output_pickle_path)
        return generated_graphs


class Kronecker(BaseGraphModel):
    """
    Kronecker Graph Model from SNAP
    """

    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='Kronecker', input_graph=input_graph, run_id=run_id)
        if 'Linux' in platform.platform():
            self.kronfit_exec = './kronfit_linux'
            self.krongen_exec = './krongen_linux'

        else:
            self.kronfit_exec = './kronfit_mac'
            self.krongen_exec = './krongen_mac'
        return

    def _fit(self) -> None:
        """
        call KronFit
        """
        output_file = f'./src/kronecker/{self.initial_gname}_{self.run_id}-fit'

        # write edgelist to the path, but graph needs to start from 1
        g = nx.convert_node_labels_to_integers(self.input_graph, first_label=1, label_attribute='old_label')
        directed_g = g.to_directed()  # kronecker expects a directed graph

        edgelist_path = f'src/kronecker/{self.initial_gname}_{self.run_id}.txt'
        nx.write_edgelist(directed_g, edgelist_path, data=False)

        bash_code = f'cd src/kronecker; {self.kronfit_exec} -i:{self.initial_gname}_{self.run_id}.txt -o:{self.initial_gname}_{self.run_id}-fit'
        completed_process = sub.run(bash_code, shell=True)  # , stdout=sub.PIPE)

        if completed_process.returncode != 0:
            CP.print_blue(f'Error in KronFit: "{self.input_graph.name}"')
            matrix = []

        elif not check_file_exists(output_file):
            CP.print_blue(f'Error in KronFit: "{self.input_graph.name}"')
            matrix = []

        else:
            with open(output_file) as f:
                last_line = f.readlines()[-1]
                last_line = last_line.replace(']', '')
                matrix = last_line[last_line.find('[') + 1:]
            # CP.print_blue('Initiator matrix:', matrix)

        self.params['initiator_matrix'] = matrix
        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        """
        call KronGen
        """
        orig_n = self.input_graph.order()
        kron_iters = int(math.log2(orig_n))  # floor of log2 gives a bound on kronecker iteration count
        if math.fabs(2 ** kron_iters - orig_n) > math.fabs(2 ** (kron_iters + 1) - orig_n):
            kron_iters += 1

        assert 'initiator_matrix' in self.params, 'Initiator matrix not found'
        matrix = self.params['initiator_matrix']

        output_file = f'src/kronecker/{self.initial_gname}_{self.run_id}_kron.txt'

        if len(matrix) == 0:  # KronFit failed
            CP.print_blue(f'Error in KronGen: "{self.input_graph.name}"')
            graph = get_blank_graph(gname)

        else:
            bash_code = f'cd src/kronecker; ./{self.krongen_exec} -o:{self.initial_gname}_{self.run_id}_kron.txt -m:"{matrix}" -i:{kron_iters}'
            completed_process = sub.run(bash_code, shell=True, stdout=sub.PIPE)

            if completed_process.returncode != 0:
                CP.print_blue(f'Error in KronGen: "{self.input_graph.name}"')
                graph = get_blank_graph(gname)

            elif not check_file_exists(output_file):
                CP.print_blue(f'Error in KronGen: "{self.input_graph.name}"')
                graph = get_blank_graph(gname)

            else:
                graph = nx.read_edgelist(output_file, nodetype=int, create_using=nx.Graph())
                graph.name = gname

                delete_files(output_file)
        graph.gen_id = gen_id
        return graph


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


class GraphVAE(BaseGraphModel):
    """
    Graph Variational Autoencoder - from T. Kipf
    """
    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='GraphVAE', input_graph=input_graph, run_id=run_id)
        return

    def _fit(self) -> None:
        adj_mat = nx.adjacency_matrix(self.input_graph)  # converts the graph into a sparse adj mat
        prob_mat = fit_vae(adj_matrix=adj_mat)
        self.params['prob_mat'] = prob_mat

        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        assert 'prob_mat' in self.params, 'Improper params. Prob matrix object is missing.'
        g = get_graph_from_prob_matrix(self.params['prob_mat'])
        g.name = gname
        g.gen_id = gen_id

        return g
