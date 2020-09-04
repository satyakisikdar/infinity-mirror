import os

from src.utils import ColorPrint, latex_printer

def main():
    df_path = './dataframes/'
    for subdir, dirs, files in os.walk(df_path):
        for filename in files:
            if filename.split('.')[-1] == 'csv':
                path = os.path.join(df_path, filename)
                print(filename)
                latex_printer(path)
            else:
                ColorPrint.print_red(f'CAUTION: Skipped {filename}')
    return

if __name__ == '__main__':
    main()
