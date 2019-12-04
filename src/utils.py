import sys
from pathlib import Path
import pickle
from typing import Union, Any
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

sns.set_style("darkgrid")

def check_file_exists(path: Union[Path, str]) -> bool:
    """
    Checks if file exists at path
    :param path:
    :return:
    """
    if isinstance(path, str):
        path = Path(path)
    return path.exists()


def print_float(x: float) -> float:
    """
    Prints a floating point rounded to 3 decimal places
    :param x:
    :return:
    """
    return round(x, 3)


def load_pickle(path: Union[Path, str]) -> Any:
    """
    Loads a pickle from the path
    :param path:
    :return:
    """
    assert check_file_exists(path), f'{path} does not exist'
    return pickle.load(open(path, 'rb'))


def make_plot(y, kind='line', x=None, title='', xlabel='', ylabel='') -> None:
    if isinstance(y, dict):
        lists = sorted(y.items())
        x, y = zip(*lists)
    else: # if isinstance(x, list) or isinstance(x, np.array):
        x = list(range(len(y)))

    if kind == 'line':
        # plt.plot(x, y, marker='o', linestyle='--')
        sns.lineplot(x, y, marker='o', dashes='--') #, dashes=True)
    if kind =='scatter':
        # plt.scatter(x, y, marker='o')
        sns.scatterplot(x, y, alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return


class ColorPrint:
    @staticmethod
    def print_red(message, end='\n'):
        sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_green(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_orange(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_blue(message, end='\n'):
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_bold(message, end='\n'):
        sys.stdout.write('\x1b[1;37m' + message.strip() + '\x1b[0m' + end)
