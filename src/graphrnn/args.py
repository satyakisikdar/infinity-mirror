import os
from pathlib import Path 

### program configuration
class Args():
    def __init__(self, note='GraphRNN_MLP', graph_type='DD'):
        ### if clean tensorboard
        self.clean_tensorboard = False
        ### Which CUDA GPU device is used for training
        self.cuda = 0 

        ### Which GraphRNN model variant is used.
        # The simple version of Graph RNN
        self.note = note
        ## The dependent Bernoulli sequence version of GraphRNN
        # self.note = 'GraphRNN_RNN'

        ## for comparison, removing the BFS compoenent
        # self.note = 'GraphRNN_MLP_nobfs'
        # self.note = 'GraphRNN_RNN_nobfs'

        ### Which dataset is used to train the model
        self.graph_type = graph_type
        # self.graph_type = 'DD'
        # self.graph_type = 'caveman'
        # self.graph_type = 'caveman_small'
        # self.graph_type = 'caveman_small_single'
        # self.graph_type = 'community4'
        # self.graph_type = 'grid'
        # self.graph_type = 'grid_small'
        # self.graph_type = 'ladder_small'
        # self.graph_type = 'karate'
        # self.graph_type = 'grid_100'

        # self.graph_type = 'enzymes'
        # self.graph_type = 'enzymes_small'
        # self.graph_type = 'barabasi'
        # self.graph_type = 'barabasi_small'
        # self.graph_type = 'citeseer'
        # self.graph_type = 'citeseer_small'

        # self.graph_type = 'barabasi_noise'
        # self.noise = 10
        #
        # if self.graph_type == 'barabasi_noise':
        #     self.graph_type = self.graph_type+str(self.noise)

        # if none, then auto calculate
        self.max_num_node = None # max number of nodes in a graph
        self.max_prev_node = None # max previous node that looks back

        ### network config
        ## GraphRNN
        if 'small' in self.graph_type:
            self.parameter_shrink = 2
        else:
            self.parameter_shrink = 1
        self.hidden_size_rnn = int(128/self.parameter_shrink) # hidden size for main RNN
        self.hidden_size_rnn_output = 16 # hidden size for output RNN
        self.embedding_size_rnn = int(64/self.parameter_shrink) # the size for LSTM input
        self.embedding_size_rnn_output = 8 # the embedding size for output rnn
        self.embedding_size_output = int(64/self.parameter_shrink) # the embedding size for output (VAE/MLP)

        self.batch_size = 32 # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = 32
        self.num_layers = 4

        ### training config
        self.num_workers = 4 # num workers to load data, default 4
        self.batch_ratio = 32 # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.test_total_size = self.batch_ratio * self.batch_size
        self.epochs = 3_000 # now one epoch means self.batch_ratio x batch_size
        # self.epochs = 100  # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 100
        # self.epochs_test_start = 50
        self.epochs_test = 500
        # self.epochs_test = 50
        self.epochs_log = 100
        # self.epochs_log = 100
        self.epochs_save = 750
        # self.epochs_save = 100

        self.lr = 0.003
        self.milestones = [400, 1000]
        self.lr_rate = 0.3

        self.sample_time = 2 # sample time in each time step, when validating

        ### output config and make dirs

        # self.dir_input = "/dfs/scratch0/jiaxuany0/"
        self.dir_input = "./"
        self.model_save_path = self.dir_input+'model_save/' # only for nll evaluation
        self.graph_save_path = self.dir_input+'graphs/'
        self.figure_save_path = self.dir_input+'figures/'
        self.timing_save_path = self.dir_input+'timing/'
        self.figure_prediction_save_path = self.dir_input+'figures_prediction/'
        self.nll_save_path = self.dir_input+'nll/'


        self.load = False # if load model, default lr is very low
        self.load_epoch = 3000
        self.save = True


        ### baseline config
        # self.generator_baseline = 'Gnp'
        self.generator_baseline = 'BA'

        # self.metric_baseline = 'general'
        # self.metric_baseline = 'degree'
        self.metric_baseline = 'clustering'


        ### filenames to save intemediate and final outputs make dirs too 
        os.makedirs(f'{self.graph_save_path}/{self.note}', exist_ok=True)
        os.makedirs(f'{self.graph_save_path}/{self.note}/{self.graph_type}', exist_ok=True)

        # self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_'
        self.fname = f'{self.note}/{self.graph_type}'  # we dont care a whole lot about the number of layers and hidden sizes
        # self.fname_pred = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_pred_'
        self.fname_pred = f'{self.fname}/pred_'
        self.fname_train = f'{self.fname}/train_'
        self.fname_test = f'{self.fname}/test_'
        # self.fname_baseline = self.graph_save_path + self.graph_type + self.generator_baseline+'_'+self.metric_baseline
        self.fname_baseline = f'{self.graph_save_path}/{self.graph_type}/{self.generator_baseline}_{self.metric_baseline}'

