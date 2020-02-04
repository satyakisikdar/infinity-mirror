import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.graphrnn.train import *

def gen(filename, gen_num=10):
    with open(filename, 'rb') as f:
        args, model, output = pickle.load(f)
    generated = test_rnn_epoch(0, args, model, output, test_batch_size=gen_num)
    with open('./src/graphrnn/dump/gen.p', 'wb') as f:
        pickle.dump(generated, f)
    return generated


