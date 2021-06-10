import time as tm
from typing import List

from tensorboard_logger import log_value

from src.graphrnn.data import decode_adj
from src.graphrnn.model import *
from src.graphrnn.utils import *
from src.graphrnn.utils import save_graph_list


def train_mlp_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x = x.cpu()
        y = y.cpu()

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = torch.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        if epoch % args.epochs_log == 0 and batch_idx == 0:  # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs, loss.data.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        # log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.data.item()
    return loss_sum / (batch_idx + 1)


def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time=1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = torch.zeros(test_batch_size, max_num_node, args.max_prev_node).cpu()  # normalized prediction score
    y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node).cpu()  # discrete prediction
    x_step = torch.ones(test_batch_size, 1, args.max_prev_node).cpu()
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = rnn.hidden.data.cpu()
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    # # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)
    return G_pred_list


def test_mlp_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = torch.zeros(test_batch_size, max_num_node, args.max_prev_node).cpu()  # normalized prediction score
        y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node).cpu()  # discrete prediction
        x_step = torch.ones(test_batch_size, 1, args.max_prev_node).cpu()
        for i in range(max_num_node):
            print('finish node', i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:, i:i + 1, :].cpu(), current=i, y_len=y_len,
                                               sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = rnn.hidden.data.cpu()
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def test_mlp_partial_simple_epoch(epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = torch.zeros(test_batch_size, max_num_node, args.max_prev_node).cpu()  # normalized prediction score
        y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node).cpu()  # discrete prediction
        x_step = torch.ones(test_batch_size, 1, args.max_prev_node).cpu()
        for i in range(max_num_node):
            print('finish node', i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised_simple(y_pred_step, y[:, i:i + 1, :].cpu(), current=i, y_len=y_len,
                                                      sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = rnn.hidden.data.cpu()
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def train_mlp_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x = x.cpu()
        y = y.cpu()

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = torch.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss

        loss = 0
        for j in range(y.size(1)):
            # print('y_pred',y_pred[0,j,:],'y',y[0,j,:])
            end_idx = min(j + 1, y.size(2))
            loss += binary_cross_entropy_weight(y_pred[:, j, 0:end_idx], y[:, j, 0:end_idx]) * end_idx

        if epoch % args.epochs_log == 0 and batch_idx == 0:  # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs, loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_' + args.fname, loss.data[0], epoch * args.batch_ratio + batch_idx)

        loss_sum += loss.data[0]
    return loss_sum / (batch_idx + 1)


########### train function for LSTM + VAE
def train(args, dataset_train, rnn, output, iteration) -> List[nx.Graph]:
    # check if load existing model
    epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    t_start = tm.time()
    G_pred = []  # list of predicted graphs
    while epoch <= args.epochs:
        time_start = tm.time()
        # train
        train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                        optimizer_rnn, optimizer_output,
                        scheduler_rnn, scheduler_output)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        if epoch % 20 == 0:
            print(f'epoch: {epoch}, time/epoch: {round((tm.time() - t_start) / 20, 2)}s')
            t_start = tm.time()

        # test
        sample_time = 0
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            G_pred = []
            while len(G_pred) < args.test_total_size:
                G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,
                                             sample_time=sample_time)
                G_pred.extend(G_pred_step)

            # save graphs
            # fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
            fname = f'{args.graph_save_path}{args.fname_pred}{iteration}_{epoch}.dat'
            save_graph_list(G_pred, fname)
            print(f'epoch: {epoch} test done, graphs saved at: {fname!r}')
        epoch += 1
    return G_pred
