"""
Fit method for GraphRNN
this is slightly different from the normal ones
1. Fit should take a list of 1,024 graphs and return a model object
2. Gen should take a model object and generate 1,024 graphs
"""

from typing import List
from train import *

def fit(graphs: List[nx.Graph], model_type: str, gname: str, gen_id: int):
    note_dict = {'mlp': 'GraphRNN_MLP', 'rnn': 'GraphRNN_RNN'}
    assert len(graphs) == 1_024, f'Expected graphs: 1024, got: {len(graphs)}'
    assert model_type in note_dict

    args = Args(graph_type=gname, note=note_dict[model_type.lower()])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    # random.seed(123)
    shuffle(graphs)
    num_train = int(0.8 * len(graphs))
    num_valid = int(0.2 * len(graphs))
    graphs_test = graphs[num_train: ]
    graphs_train = graphs[: num_train]
    graphs_validate = graphs[: num_valid]

    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)
    print('graph_validate_len', graph_validate_len)

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    print('graph_test_len', graph_test_len)

    args.max_num_node = max(graphs[i].number_of_nodes() for i in range(len(graphs)))
    max_num_edge = max(graphs[i].number_of_edges() for i in range(len(graphs)))
    min_num_edge = min(graphs[i].number_of_edges() for i in range(len(graphs)))

    # args.max_num_node = 2000
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

    ### model initialization
    ## Graph RNN VAE model
    # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                   hidden_size=args.hidden_size, num_layers=args.num_layers).cuda()

    if 'GraphRNN_VAE_conditional' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).cuda()
        output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output,
                                           y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_MLP' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=False).cuda()
        output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output,
                           y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_RNN' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output).cuda()
        output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                           has_output=True, output_size=1).cuda()

    ### start training
    output_graphs = train(args, dataset_loader, rnn, output, gen_id)
    print(f'len: {len(output_graphs)}')
    for og in output_graphs[: 10]:
        print(f'{og.name} n = {og.order()} m = {og.size()}')
    print()
    return output_graphs


if __name__ == '__main__':
    # g = nx.karate_club_graph(); gname = 'karate'
    g = nx.ring_of_cliques(500, 4); gname = 'ring-cliq-500-4'
    g.name = gname
    graphs = [nx.Graph(g) for _ in range(1024)]
    model_type = 'mlp'

    for gen_id in range(1, 21):
        print(f'\nround {gen_id}\n')
        graphs = fit(graphs=graphs, model_type=model_type, gname=gname, gen_id=gen_id)
