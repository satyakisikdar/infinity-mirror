from src.graph_models import Kronecker
from src.utils import load_pickle
from sys import argv

pickle_path = '/afs/crc.nd.edu/user/t/tford5/infinity-mirror/cleaned/clique-ring-500-4/Kronecker/list_20_1.pkl.gz'
graphs = load_pickle(pickle_path)

if len(argv) < 2:
    print('Enter generation 1/5/10/20 as an arg')
    exit()

gen = int(argv[1])
assert gen in (1, 5, 10, 20), 'invalid argument. pick between 1/5/10/20'

g = graphs[gen]
kron = Kronecker(g, 0)
print('Fitting Kronecker...')
kron.update(g)
print('gen:', gen, kron.params)
print(f'gen: {gen} {kron.params}', file=open(f'kron-param-{gen}.txt', 'w'))
