import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from train import *

def fit(graphs):
    ''' performs GraphRNN model training on a specified list of graphs
        parameters:
            graphs (nx.Graph list): list of Networkx Graph() objects
        output:
            the thing
    '''
    # maybe not necessary?
    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    args = Args()
    args.max_prev_node=40
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run"+time, flush_secs=5)
    # /maybe not necessary?

    # compute the train/test/val split
    split = 0.8
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(split*graphs_len):]
    graphs_train = graphs[0:int(split*graphs_len)]
    graphs_validate = graphs[0:int((1 - split)*graphs_len)]

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)

    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)

    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    dataset = Graph_sequence_sampler_pytorch(graphs_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node, iteration=20)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))], num_samples=args.batch_size*args.batch_ratio, replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sample_strategy)

    # initialize the model
    if 'GraphRNN_VAE_conditional' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers,
                        has_input=True, has_output=False).cuda()
        output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_MLP' in args.note:
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers,
                        has_input=True, has_output=False).cuda()
        output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda()
    elif 'GraphRNN_RNN' in args.note:
        print('woah:', args.max_prev_node)
        rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers,
                        has_input=True, has_output=True, output_size=args.hidden_size_rnn_output).cuda()
        output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers,
                           has_input=True, has_output=True, output_size=1).cuda()

    train(args, dataset_loader, rnn, output)

    return args, dataset_loader, rnn, output

def gen(args, rnn, output, gen_num=10):
    G_pred = []
    while len(G_pred) < gen_num:
        G_pred_step = test_rnn_epoch(0, args, rnn, output, test_batch_size=16)
        G_pred.extend(G_pred_step)
    return G_pred

def main():
    args = Args()
    graphs = create_graphs.create(args)
    print('----------graphs made----------')

    args, dataset_loader, rnn, output = fit(graphs)
    print('==========model fit==========')

    generated = gen(args, rnn, output, gen_num=10)
    print('**********graphs generated**********')
    print(generated)
    print(len(generated))

    return 0

main()
