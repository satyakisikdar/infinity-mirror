import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.graphrnn.train import *

def gen(args, model, output, gen_num=1):
    generated = test_rnn_epoch(0, args, model, output, test_batch_size=gen_num)
    return generated


