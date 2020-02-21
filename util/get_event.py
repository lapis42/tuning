#!/opt/localuser/anaconda3/bin python
import os, sys
import numpy as np
import scipy.io as sio
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
#from utils import spikeglx 

class App(QWidget):
    def __init__(self):
        super().__init__()
        default_path = 'C:\\SGL_DATA'
        self.filename, _ = QFileDialog.getOpenFileName(self, directory=default_path, filter='Bin files (*.nidq.bin)')
        self.show()


def read_meta(bin_file):
    meta_file = bin_file.replace('.bin', '.meta')
    meta = {}
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta_data = f.read().splitlines()
            for m in meta_data:
                [key, item] = m.split('=')
                if item.isnumeric():
                    item = float(item)
                meta[key] = item
    else:
        print('No meta file')
    return meta


def read_bin(bin_file, meta):

    c_nidq = np.array(meta['snsMnMaXaDw'].split(','), dtype=int)
    n_nidq = int(os.path.getsize(bin_file) / (2 * c_nidq.sum()))

    dt_nidq = np.dtype([('mn', 'int16', c_nidq[0]), ('ma', 'int16', c_nidq[1])])
    d_nidq = np.memmap(bin_file, dtype=dt_nidq, mode='r', shape=(n_nidq,))

    return d_nidq


def main():
    # get file name
    app = QApplication(sys.argv)
    ex = App()
    bin_file = ex.filename
    ex.close()

    if bin_file == '':
        return

    # load meta data
    meta = read_meta(bin_file)
    sample_rate = meta['niSampRate']

    # load binary data
    data_nidq = read_bin(bin_file, meta)

    # find onset and offset time
    # photodiode on voltage: 0.98
    # sync pulse voltage: 0.59

    bit_per_voltage = 2**15 / 2
    voltage_on = data_nidq['ma'] >= (0.25 * bit_per_voltage)

    # since photodiode voltage is somewhat unstable, i tried to detect the event when voltage is maintained for 4 timestamps.
    d_voltage_on = ~voltage_on[:-4, :] & voltage_on[1:-3, :] & voltage_on[2:-2, :] & voltage_on[3:-1, :] & voltage_on[4:, :]
    d_voltage_off = voltage_on[:-4, :] & ~voltage_on[1:-3, :] & ~voltage_on[2:-2, :] & ~voltage_on[3:-1, :] & ~voltage_on[4:, :]

    data = {}
    data['sync_onset'] = np.argwhere(d_voltage_on[:, 1])[:, 0] / sample_rate
    start_time = data['sync_onset'][0] # realign by the first sync onset time

    data['sync_onset'] = data['sync_onset'] - start_time
    data['diode_onset'] = np.argwhere(d_voltage_on[:, 0])[:, 0] / sample_rate - start_time
    data['diode_offset'] = np.argwhere(d_voltage_off[:, 0])[:, 0] / sample_rate - start_time
    data['sync_offset'] = np.argwhere(d_voltage_off[:, 1])[:, 0] / sample_rate - start_time
    data['laser_onset'] = np.argwhere(d_voltage_on[:, 2])[:, 0] / sample_rate - start_time
    data['laser_offset'] = np.argwhere(d_voltage_off[:, 2])[:, 0] / sample_rate - start_time
    data['cue_onset'] = np.argwhere(d_voltage_on[:, 3])[:, 0] / sample_rate - start_time
    data['cue_offset'] = np.argwhere(d_voltage_off[:, 3])[:, 0] / sample_rate - start_time
    data['sample_rate'] = sample_rate
    data['fileTime'] = meta['fileCreateTime']

    print('Recording duration: {:.1f} min'.format(float(meta['fileTimeSecs']) / 60))
    print('n={} cue (ttl)'.format(len(data['cue_onset'])))
    print('n={} cue (diode)'.format(len(data['diode_onset'])))
    print('n={} laser'.format(len(data['laser_onset'])))
    print('n={} sync'.format(len(data['sync_onset'])))

    save_filename = bin_file.replace('.nidq.bin', '_data.mat')
    sio.savemat(save_filename, data)
    print('Saved {}'.format(save_filename))


if __name__ == "__main__":
    main()
