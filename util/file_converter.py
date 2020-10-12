#!/opt/localuser/anaconda3/bin python
import os, time, argparse
from datetime import datetime
import numpy as np
import pandas as pd
from tkinter import *
from tkinter import filedialog

def main(foldername):
    # select folder to convert
    if not foldername:
        print('No folder selected')
        return

    # search binary files
    bin_file = [os.path.join(fd, fn) for fd, _, files in os.walk(foldername) for fn in files if fn == 'mua.bin' and not 'kilosort' in fd]
    bin_file.sort(key=lambda x: os.path.getmtime(x))

    # mkdir
    save_path = os.path.join(foldername, 'kilosort')
    save_file = os.path.join(save_path, 'mua.bin')

    # select file
    while 1:
        n_file = len(bin_file)
        for i in range(n_file):
            print('{}: {}'.format(i, bin_file[i]))

        str = input('s(start), number (delete the file), q(quit): ')
        if str == 's':
            break 
        elif str.isnumeric():
            del bin_file[int(str)]
        elif str == 'q':
            return


    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # save metadata
    mtime = [datetime.fromtimestamp(os.path.getmtime(x)).strftime("%Y%m%d_%H%M%S") for x in bin_file]
    file_size = [os.path.getsize(x) for x in bin_file]
    n_sample = [int(os.path.getsize(x) / (4*160)) for x in bin_file]
    start = np.concatenate([[0], np.cumsum(n_sample)[:-1]])
    new_filesize = np.array(file_size) / 2 / 160 * 128


    df = pd.DataFrame({'filename': bin_file,
        'mtime': mtime,
        'filesize': file_size,
        'new_filesize': new_filesize,
        'n_sample': n_sample,
        'start': start})
    df.to_csv(os.path.join(save_path, 'meta.csv'))

    with open(save_file, 'wb') as f:
        for i_file in range(n_file):
            bin_data = np.memmap(bin_file[i_file], dtype='int16', mode='r', shape=(n_sample[i_file], 2*160))

            # calculate read size
            print("saving {}".format(bin_file[i_file]))
            bin_data[:, 1:256:2].tofile(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('foldername', type=str, help='foldername')
    arg = parser.parse_args()
    main(arg.foldername)
