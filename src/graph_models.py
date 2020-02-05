"""
Container for different graph models
"""
import abc
import math
import os
import platform
import random
import subprocess as sub
from itertools import combinations
from time import time
from typing import List, Dict, Any, Union, Set, Tuple

import networkx as nx
import numpy as np
from scipy import sparse

from src.graph_io import networkx_to_graphtool, graphtool_to_networkx
from src.graph_stats import GraphStats
from src.utils import ColorPrint as CP
from src.utils import check_file_exists, load_pickle, delete_files, get_blank_graph, get_graph_from_prob_matrix

__all__ = ['BaseGraphModel', 'ErdosRenyi', 'UniformRandom', 'ChungLu', 'BTER', 'CNRG', 'HRG', 'Kronecker',
           'GraphAE', 'GraphVAE', 'SBM', 'GraphForge', 'NetGAN']


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
        generated_graphs = []
        for i in range(num_graphs):
            g = self._gen(gen_id=gen_id, gname=f'{self.input_graph.name}_{gen_id}_{self.run_id}_{i + 1}')
            if not isinstance(g, nx.Graph):
                g = nx.Graph(g)  # make it into an undirected graph with no parallel edges
            self_loops = list(nx.selfloop_edges(g))
            g.remove_edges_from(self_loops)  # remove self loops
            generated_graphs.append(g)

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


class _BTER(BaseGraphModel):
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


class BTER(BaseGraphModel):
    """
        BTER model by Tammy Kolda
        feastpack implementation at https://www.sandia.gov/~tgkolda/feastpack/feastpack_v1.2.zip
    """

    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='BTER', input_graph=input_graph, run_id=run_id)
        return

    def _fit(self) -> None:
        # find degree distribution and avg clustering by degree
        g_stats = GraphStats(self.input_graph, run_id=-1)

        self.params['n'] = self.input_graph.order()
        self.params['degree_dist'] = g_stats.degree_dist(normalized=False)  # we need the counts
        self.params['degree_seq'] = g_stats['degree_seq']
        self.params['avg_cc_by_deg'] = g_stats.clustering_coefficients_by_degree()

        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        assert 'degree_dist' in self.params and 'avg_cc_by_deg' in self.params and 'n' in self.params, \
            'insufficient parameters for BTER'

        n, avg_cc_by_deg = self.params['n'], self.params['avg_cc_by_deg']
        degree_seq, degree_dist = self.params['degree_seq'], self.params['degree_dist']

        g = nx.empty_graph(n=n)  # adding n isolated nodes

        # preprocessing
        # step 1: assign n1 nodes to have degree 1, n2 nodes to have degree 2, ...
        assigned_deg: Dict[int, int] = {node: degree_seq[node] for node in g.nodes()}  # deg seq is sorted

        nx.set_node_attributes(g, values=assigned_deg, name='assigned_deg')

        # step 2: partition all nodes into affinity blocks, ideally blocks with degree d as d+1 nodes - no edges yet
        #         ignore degree 1 nodes
        node2block: Dict[int, int] = {}  # keyed by node, vals are block id
        block_members: Dict[int, Tuple[int, Set[int]]] = {}  # keyed by block_id, vals: expected degree, set of members

        idx = 0
        block_id = 0
        while idx < n - 1:  # idx is node id
            deg = assigned_deg[idx]
            if deg == 1:  # skip the degree 1 nodes
                idx += 1
                continue

            for j in range(deg + 1):  # assign deg+1 nodes to degree block of degree deg
                node = idx + j
                if node > n - 1:  # if node > n, break
                    break
                node2block[node] = block_id  # assign node to block

                if block_id not in block_members:  # update block_members data structure
                    block_members[
                        block_id] = deg, set()  # first item is the expected degree, second is the set of members
                block_members[block_id][1].add(node)

            block_id += 1  # update block id
            idx += deg + 1  # skip deg + 1 nodes

        # phase 1
        # step 3: add edges within each affinity block by fitting a dense ER graph depending on avg cc by degree
        phase1_edges = []

        for block_id, (exp_deg, members) in block_members.items():
            clustering_coeff = avg_cc_by_deg[exp_deg]
            prob = math.pow(clustering_coeff, 1 / 3)
            for u, v in combinations(members, 2):
                r = random.random()
                if r <= prob:
                    g.add_edge(u, v)
                    phase1_edges.append((u, v))

        # phase 2
        # step 4: Add edges between blocks by using excess degree. Expected degree: d_i, already incident: d_j. excess degree: d_i - d_j.
        #         Create a CL graph based on the excess degrees

        excess_degs = {node: max(0, assigned_deg[node] - g.degree(node))
                       for node in g.nodes()}  # dictionary of excess degs

        if sum(
                excess_degs.values()) % 2 != 0:  # excess degs do not sum to even degrees, decrease the node with max degree by 1
            max_deg_node, max_deg = max(excess_degs.items(), key=lambda x, y: y)
            excess_degs[max_deg_node] -= 1  # decrease it by 1 to make the sum even

        phase2_graph = nx.configuration_model(excess_degs.values(), create_using=nx.Graph())
        selfloops = list(nx.selfloop_edges(phase2_graph))
        phase2_graph.remove_edges_from(selfloops)
        g.add_edges_from(phase2_graph.edges())

        g.name = gname
        g.gen_id = gen_id
        return g


class CNRG(BaseGraphModel):
    """
    Satyaki's Clustering-Based Node Replacement Grammars https://github.com/satyakisikdar/cnrg
    """

    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='CNRG', input_graph=input_graph, run_id=run_id)
        return

    def _fit(self) -> None:
        from src.cnrg.runner import get_grammar
        grammar = get_grammar(self.input_graph, name=self.input_graph.name)
        self.params['grammar'] = grammar
        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        assert 'grammar' in self.params, 'Improper params. Grammar object is missing.'
        from src.cnrg.runner import generate_graph
        light_g = generate_graph(target_n=self.input_graph.order(), rule_dict=self.params['grammar'].rule_dict,
                                 tolerance_bounds=0.01)  # exact generation
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


class SBM(BaseGraphModel):
    """
    Stochastic Block Model  - degree corrected
    """

    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='SBM', input_graph=input_graph, run_id=run_id)
        return

    def _fit(self) -> None:
        import graph_tool.all as gt  # local import

        gt_g = networkx_to_graphtool(self.input_graph)  # convert to graphtool obj
        state = gt.minimize_blockmodel_dl(gt_g)  # run SBM fit
        self.params['state'] = state
        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        import graph_tool.all as gt  # local import

        assert 'state' in self.params, 'missing parameter: state for SBM'
        state = self.params['state']

        gen_gt_g = gt.generate_sbm(state.b.a,
                                   gt.adjacency(state.get_bg(), state.get_ers()).T)  # returns a graphtool graph
        g = graphtool_to_networkx(gen_gt_g)
        g.name = gname
        g.gen_id = gen_id

        return g


class GraphVAE(BaseGraphModel):
    """
    Graph Variational Autoencoder - from T. Kipf
    """

    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='GraphVAE', input_graph=input_graph, run_id=run_id)
        return

    def _fit(self) -> None:
        from src.gae.fit import fit_vae
        adj_mat = nx.adjacency_matrix(self.input_graph)  # converts the graph into a sparse adj mat
        prob_mat = fit_vae(adj_matrix=adj_mat)
        self.params['prob_mat'] = sparse.csr_matrix(prob_mat)  # turn this into a sparse CSR matrix

        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        assert 'prob_mat' in self.params, 'Improper params. Prob matrix object is missing.'
        g = get_graph_from_prob_matrix(self.params['prob_mat'])
        g.name = gname
        g.gen_id = gen_id

        return g


class GraphAE(BaseGraphModel):
    """
    Graph Autoencoder - from T. Kipf
    """

    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='GraphAE', input_graph=input_graph, run_id=run_id)
        return

    def _fit(self) -> None:
        from src.gae.fit import fit_ae

        adj_mat = nx.adjacency_matrix(self.input_graph)  # converts the graph into a sparse adj mat
        prob_mat = fit_ae(adj_matrix=adj_mat)
        self.params['prob_mat'] = prob_mat

        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        assert 'prob_mat' in self.params, 'Improper params. Prob matrix object is missing.'
        g = get_graph_from_prob_matrix(self.params['prob_mat'])
        g.name = gname
        g.gen_id = gen_id

        return g


class GraphForge(BaseGraphModel):
    """
    Spectral Graph Forge by Baldesi et al
    Copy 50% of the original
    """

    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='GraphForge', input_graph=input_graph, run_id=run_id)
        return

    def _fit(self) -> None:
        return  # does not need to fit

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        g = nx.spectral_graph_forge(self.input_graph, alpha=0.5)
        g.name = gname
        g.gen_id = gen_id
        return g


class NetGAN(BaseGraphModel):
    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='NetGAN', input_graph=input_graph, run_id=run_id)
        self.prep_environment()
        return

    def prep_environment(self) -> None:
        proc = sub.run('conda init bash; . ~/.bashrc; conda activate netgan', shell=True)
        os.makedirs('./src/netgan/dumps', exist_ok=True)  # make the directory to store the dumps
        if proc.returncode == 0:  # conda environment exists
            return

        CP.print_blue('Making conda environment for NetGAN')
        proc = sub.run('conda env create -f ./envs/netgan.yml', shell=True, stdout=None)  # create and activate environment

        assert proc.returncode == 0, 'Error while creating env for NetGAN'
        return

    def _fit(self) -> None:
        dump = f'./src/netgan/dumps'
        gname = f'{self.input_graph.name}_{self.run_id}'
        path = f'{dump}/{gname}.g'
        nx.write_edgelist(self.input_graph, path, data=False)

        proc = sub.run(f'conda init bash; . ~/.bashrc; conda activate netgan; python src/netgan/fit.py {gname} {path}; conda deactivate',
                       shell=True)
        assert proc.returncode == 0, 'NetGAN fit did not work'
        assert check_file_exists(f'{dump}/{gname}.pkl.gz'), 'pickle not found'
        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        pass  # NetGAN can generate multiple graphs at once

    def generate(self, num_graphs: int, gen_id: int) -> List[nx.Graph]:

        dump = f'./src/netgan/dumps'
        gname = f'{self.input_graph.name}_{self.run_id}'
        pickle_path = f'{dump}/{gname}.pkl.gz'

        proc = sub.run(f'conda init bash; . ~/.bashrc; conda activate netgan; python src/netgan/gen.py {gname} {pickle_path} {num_graphs}',
                       shell=True)

        assert proc.returncode == 0, 'error in NetGAN generate'
        output_pickle_path = f'{dump}/{gname}_graphs.pkl.gz'

        generated_graphs = []
        for i, gen_graph in enumerate(load_pickle(output_pickle_path)):
            gen_graph.name = f'{self.input_graph.name}_{self.run_id}_{i + 1}'  # adding the number of graph
            gen_graph.gen_id = gen_id
            generated_graphs.append(gen_graph)

        delete_files(output_pickle_path)
        return generated_graphs


class GraphRNN(BaseGraphModel):
    def __init__(self, input_graph: nx.Graph, run_id: int, **kwargs) -> None:
        super().__init__(model_name='NetGAN', input_graph=input_graph, run_id=run_id)
        return

    def _fit(self) -> None:
        from src.graphrnn.fit import fit
        graphs = []
        for _ in range(100):
            graphs.append(self.input_graph)
        args, dataset_loader, model, output = fit(graphs)
        self.params['args'] = args
        self.params['model'] = model
        self.params['output'] = output
        return

    def _gen(self, gname: str, gen_id: int) -> nx.Graph:
        from src.graphrnn.fit import gen
        assert 'args' in self.params
        assert 'model' in self.params
        assert 'output' in self.params
        gen_graphs = gen(args=self.params['args'], model=self.params['model'], output=self.params['output'])
        g.name = gname
        g.gen_id = gen_id
        return gen_graphs
