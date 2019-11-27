import sys
from pathlib import Path

def check_file_exists(path):
    """
    Checks if file exists at path
    :param path:
    :return:
    """
    if isinstance(path, str):
        path = Path(path)
    return path.exists()


def print_float(x):
    """
    Prints a floating point rounded to 3 decimal places
    :param x:
    :return:
    """
    return round(x, 3)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
