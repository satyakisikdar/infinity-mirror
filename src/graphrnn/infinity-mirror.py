"""
Fit method for GraphRNN
this is slightly different from the normal ones
1. Fit should take a list of 50 graphs and return a model object
2. Gen should take a model object and generate 50 graphs
"""
import os
import sys
from os.path import join
from pathlib import Path
from random import shuffle
from sys import argv

sys.path.extend(['./', './..', './../..'])

from src.graphrnn.args import Args
from src.graphrnn.data import Graph_sequence_sampler_pytorch
from src.graphrnn.train import *
from src.utils import ColorPrint as CP, load_pickle, ensure_dir


def fit(graphs: List[nx.Graph], model_type: str, gname: str, iteration: int, batch_size: int, batch_ratio: int):
    note_dict = {'mlp': 'GraphRNN_MLP', 'rnn': 'GraphRNN_RNN'}
    assert len(graphs) == 50, f'Expected graphs: 50, got: {len(graphs)}'
    assert model_type in note_dict

    cleaned_graphs = [g for g in graphs if g.order() != 0 and g.size() != 0]  # GraphRNN doesn't like empty graphs
    if len(cleaned_graphs) != len(graphs):
        CP.print_orange(f'Discared {len(graphs) - len(cleaned_graphs)} empty graphs')

    assert len(cleaned_graphs) > 0, 'All 50 graphs were discarded!!'
    graphs = cleaned_graphs

    args = Args(graph_type=gname, note=note_dict[model_type.lower()], batch_size=batch_size, batch_ratio=batch_ratio)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # str(args.cuda)

    # random.seed(123)
    shuffle(graphs)
    num_train = int(0.8 * len(graphs))
    num_valid = int(0.2 * len(graphs))
    graphs_test = graphs[num_train:]
    graphs_train = graphs[: num_train]
    graphs_validate = graphs[: num_valid]

    graph_validate_len = sum(g.order() for g in graphs_validate)
    graph_validate_len /= len(graphs_validate)
    print('graph_validate_len', graph_validate_len)

    graph_test_len = sum(g.order() for g in graphs_test)
    graph_test_len /= len(graphs_test)
    print('graph_test_len', graph_test_len)

    args.max_num_node = max(g.order() for g in graphs)
    max_num_edge = max(g.size() for g in graphs)
    min_num_edge = min(g.size() for g in graphs)

    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    dataset = Graph_sequence_sampler_pytorch(graphs_train, max_prev_node=args.max_prev_node,
                                             max_num_node=args.max_num_node)
    args.max_prev_node = dataset.max_prev_node

    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size * args.batch_ratio,
                                                                     replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 sampler=sample_strategy)

    assert 'GraphRNN_MLP' in args.note, f'Invalid mode: {args.note}'

    rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                    hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                    has_output=False)#.cuda()
    output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output,
                       y_size=args.max_prev_node)#.cuda()

    ### start training
    # todo try to catch the log data (might be the actual console logs)
    output_graphs = train(args, dataset_loader, rnn, output, iteration)
    print(f'len: {len(output_graphs)}')
    for og in output_graphs[: 10]:
        print(f'{og.name} n = {og.order()} m = {og.size()}')
    print()
    return output_graphs


def init(filename: str, gname: str = '', reindex_nodes: bool = False, first_label: int = 0,
         take_lcc: bool = True) -> nx.Graph:
    """
    :param filename: path to input file
    :param gname: name of the graph
    """
    possible_extensions = ['.g', '.gexf', '.gml', '.txt', '.mat']
    filename = filename
    path = Path(filename)
    # assert check_file_exists(path), f'Path: "{self.path}" does not exist'

    if gname == '':
        gname = path.stem

    graph: nx.Graph = nx.read_edgelist(path, create_using=nx.Graph, nodetype=int)
    graph.name = gname
    graph: nx.Graph = preprocess(graph, reindex_nodes=reindex_nodes, first_label=first_label, take_lcc=take_lcc)
    assert graph.name != '', 'Graph name is empty'
    return graph


def preprocess(graph: nx.Graph, reindex_nodes: bool, first_label: int = 0, take_lcc: bool = True) -> nx.Graph:
    """
    Preprocess the graph - taking the largest connected components, re-index nodes if needed
    :return:
    """
    # CP.print_none('Pre-processing graph....')
    # CP.print_none(f'Original graph "{self.gname}" n:{self.graph.order():,} '
    #              f'm:{self.graph.size():,} #components: {nx.number_connected_components(self.graph)}')

    if take_lcc and nx.number_connected_components(graph) > 1:
        ## Take the LCC
        component_sizes = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]

        # CP.print_none(f'Taking the largest component out of {len(component_sizes)} components: {component_sizes}')

        graph_lcc = nx.Graph(graph.subgraph(max(nx.connected_components(graph), key=len)))

        perc_nodes = graph_lcc.order() / graph.order() * 100
        perc_edges = graph_lcc.size() / graph.size() * 100
        # CP.print_orange(f'LCC has {print_float(perc_nodes)}% of nodes and {print_float(perc_edges)}% edges in the original graph')

        graph = graph_lcc

    selfloop_edges = list(nx.selfloop_edges(graph))
    if len(selfloop_edges) > 0:
        # CP.print_none(f'Removing {len(selfloop_edges)} self-loops')
        graph.remove_edges_from(selfloop_edges)  # remove self-loops

    if reindex_nodes:
        # re-index nodes, stores the old label in old_label
        graph = nx.convert_node_labels_to_integers(graph, first_label=first_label,
                                                   label_attribute='old_label')
        # CP.print_none(
        #    f'Re-indexing nodes to start from {first_label}, old labels are stored in node attr "old_label"')

    # CP.print_none(f'Pre-processed graph "{self.gname}" n:{self.graph.order():,} m:{self.graph.size():,}')
    return graph


if __name__ == '__main__':
    if len(argv) < 2:
        print('Enter dataset as an arg')
        exit(1)

    batch_size, batch_ratio = 10, 5  # 10, 5 is faster
    model_type = 'mlp'

    base_path = '/scratch365/ssikdar/infinity-mirror/output/graphrnn/'
    datasets = 'tree', 'clique-ring-500-4', 'flights', 'eucore', 'chess', 'enron', 'cond-mat', 'karate'
    dataset = argv[1]
    assert dataset in datasets, f'Invalid {dataset}, pick from {datasets}'
    path_to_pickles = join(base_path, f'{dataset}_size10_ratio5')
    ensure_dir(path_to_pickles)

    pattern = re.compile('pred_(\d+)_1000.dat')
    completed_files = [(f, int(re.fullmatch(pattern, f).groups()[0])) for f in os.listdir(path_to_pickles)
                       if re.fullmatch(pattern, f)]

    if len(completed_files) != 0:
        most_recent_completed_file, current_iteration = max(completed_files, key=lambda x: x[1])
        most_recent_completed_file = join(base_path, f'{dataset}_size10_ratio5', most_recent_completed_file)
        graphs = load_pickle(most_recent_completed_file)
        CP.print_orange(f'Resuming pickle from iteration {current_iteration}')
        assert len(graphs) == 50, f'Expected 50 graphs, found {len(graphs)}'
    else:
        most_recent_completed_file, current_iteration = f'/afs/crc.nd.edu/user/d/dgonza26/infinity-mirror/input/{dataset}.g', 0
        g = init(filename=most_recent_completed_file)
        graphs = [nx.Graph(g)] * 50

    gname = f'{dataset}_size10_ratio5'
    for iteration in range(current_iteration + 1, 21):
        print(f'\niteration {iteration}\n')
        print(f'input graph: {gname}')
        graphs = fit(graphs=graphs, model_type=model_type, gname=gname, iteration=iteration, batch_size=batch_size,
                     batch_ratio=batch_ratio)
