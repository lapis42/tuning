import os, sys, time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.stats import linregress
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import tuning.spike as spike

class Tuning():
    def __init__(self):
        self.event_file = None
        self.event_path = None
        self.event_filetime = None
        self.ptb_file = None
        self.spike_file = None
        
        self.sync_time_nidq = None
        self.sync_time_fpga = np.array([
            0, 1, 3, 64, 105, 181, 266, 284, 382, 469, 531,
            545, 551, 614, 712, 726, 810, 830, 846, 893, 983, 1024,
            1113, 1196, 1214, 1242, 1257, 1285, 1379, 1477, 1537, 1567, 1634,
            1697, 1718, 1744, 1749, 1811, 1862, 1917, 1995, 2047]) # in seconds 

        self.cue_type = None
        self.cue_time_nidq = None
        self.cue_time_fpga = None
        self.laser_time_nidq = None
        self.laser_time_fpga = None
        self.spike_time = None
        self.spike_fr = None
 

    def load_event(self, event_file): # read event time
        self.event_file = event_file

        # load event file
        event_data = sio.loadmat(self.event_file, squeeze_me=True)
        self.cue_time_nidq = event_data['cue_onset']
        self.sync_time_nidq = event_data['sync_onset']
        self.laser_time_nidq = event_data['laser_onset']
        self.event_path, _ = os.path.split(self.event_file)
        self.event_filetime = time.mktime(time.strptime(event_data['fileTime'], '%Y-%m-%dT%H:%M:%S'))

        self.sync_event()


    def sync_event(self): # sync event time
        n_sync = len(self.sync_time_nidq)
        self.sync_time_fpga = self.sync_time_fpga[:n_sync]
        
        # check sync
        _, _, r_value, _, _ = linregress(self.sync_time_nidq, self.sync_time_fpga)
        print("r-squared: {}".format(r_value**2))
        if (r_value**2) < 0.98:
            print("Abnormal sync!!!")

        f = interp1d(self.sync_time_nidq, self.sync_time_fpga, fill_value="extrapolate")
        self.cue_time_fpga = f(self.cue_time_nidq)
        self.laser_time_fpga = f(self.laser_time_nidq)


    def load_ptb(self, ptb_file):
        # loading ptb file
        self.ptb_file = ptb_file
        ptb_data = sio.loadmat(self.ptb_file, squeeze_me=True, struct_as_record=False)

        if 'result' in ptb_data.keys():
            self.cue_type = ptb_data['result'].directions
        else:
            self.cue_type = ptb_data['directions']


    def load_spike(self, spike_file):
        self.spike_file = spike_file
        df = pd.read_pickle(self.spike_file)

        n_unit = int(df['spike_id'].max())
        self.spike_time = np.zeros(n_unit, dtype=object)
        self.spike_fr = np.zeros(n_unit)
        
        duration = (df['frame_id'].iloc[-1] - df['frame_id'].iloc[0]) / 25000

        for i_unit in range(n_unit):
            in_unit = df['spike_id'] == (i_unit + 1)
            self.spike_time[i_unit] = df['frame_id'][in_unit].to_numpy() / 25000
            self.spike_fr[i_unit] = sum(in_unit) / duration


    def plot(self):
        if self.spike_time is None:
            print('You have run load_spike()')
            return
        elif self.cue_time_fpga is None:
            print('You have to run load_event()')
            return
        elif self.cue_type is None:
            print('You have to run load_ptb()')
            return

        window_cue = [-0.8, 1.3] # window for plot in seconds

        cue_time = self.cue_time_fpga
        cue_type = self.cue_type
        n_type = len(np.unique(cue_type)) # cue types
        n_unit = len(self.spike_time)

        mystyle(1, n_unit)
        linecolor = plt.cm.gist_ncar(np.linspace(0, 0.9, n_type))[:, 0:3]
        f = plt.figure()
        gs0 = gridspec.GridSpec(n_unit, 2, wspace = 0.3)
            
        for i_unit in range(n_unit):
            spike_time = self.spike_time[i_unit]
            raster, psth = spike.plot(spike_time, cue_time, cue_type,
                                      window=window_cue, resolution=2)
            
            # get mean and std firing rate for each stimulus type
            spike_angle_mean = np.zeros(n_type)
            spike_angle_se = np.zeros(n_type)
            for i_type in range(n_type):
                in_type = cue_type == (i_type * 30)
                spike_num_temp = spike.count_spike(spike_time, cue_time[in_type],
                                                   window=[0, 0.5]) / 0.5
                spike_angle_mean[i_type] = np.mean(spike_num_temp)
                spike_angle_se[i_type] = np.std(spike_num_temp) / np.sqrt(len(spike_num_temp))

            # plotting
            y_max = np.max(np.concatenate([psth['conv'].flatten() for i in range(n_type)]))
            gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[i_unit, 0], hspace=0.05)
            gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[i_unit, 1])
            ax000 = f.add_subplot(gs00[0])
            ax001 = f.add_subplot(gs00[1])
            ax010 = f.add_subplot(gs01[0])

            for i_type in range(n_type):
                ax000.plot(raster['x'][i_type], raster['y'][i_type], '.', color=linecolor[i_type, :])
                ax001.plot(psth['t'], psth['conv'][i_type, :], color=linecolor[i_type, :])

            ax010.errorbar(np.arange(12)*30, spike_angle_mean, yerr=spike_angle_se)
            fr_max = np.max(spike_angle_mean + spike_angle_se)
            ax010.text(240, fr_max*0.8, 'neuron {}: {:.1f}Hz'.format(i_unit+1, self.spike_fr[i_unit]))

            ax000.set_xlim(window_cue + np.array([0.5, -0.5]))
            ax000.set_ylim([0, len(cue_type)])
            ax000.set_xticklabels([])
            ax000.set_ylabel('Trial')

            ax001.set_xlim(window_cue + np.array([0.5, -0.5]))
            ax001.set_ylim([0, y_max])
            if i_unit == n_unit - 1:
                ax001.set_xlabel('Time from cue onset (s)')
            ax001.set_ylabel('Firing rate')

            ax010.set_xlim([0, 360])
            ax010.set_ylim([0, np.max(spike_angle_mean + spike_angle_se)])
            if i_unit < n_unit - 1:
                ax010.set_xticklabels([])
            else:
                ax010.set_xlabel('Cue direction')

        plt.show()
       

def mystyle(xratio=1, yratio=1):
    mpl.rcParams['figure.figsize'] = [4.25*xratio, 1.59375*yratio]
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['font.size'] = 6

    mpl.rcParams['figure.subplot.hspace'] = 0.075

    mpl.rcParams['axes.titlesize'] = 8
    mpl.rcParams['axes.labelsize'] = 6
    mpl.rcParams['axes.linewidth'] = 0.75
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    mpl.rcParams['xtick.labelsize'] = 5
    mpl.rcParams['ytick.labelsize'] = 5

    mpl.rcParams['lines.linewidth'] = 0.75
    mpl.rcParams['lines.markersize'] = 1
    mpl.rcParams['lines.markeredgewidth'] = 0


if __name__ == "__main__":
    event_file = '/mnt/data/bmi_20191210_0/event/bmi_20191210_g2_t0_data.mat'
    ptb_file = '/mnt/data/bmi_20191210_0/event/ptb_20191210_172855.mat'
    spike_file = '/mnt/data/bmi_20191210_0/spktag/test.pd'
    
    #event_file = '/mnt/data/bmi_20191210_0/event/bmi_20191210_g4_t0_data.mat'
    t = Tuning()
    t.load_event(event_file)

    breakpoint()

    t.load_ptb(ptb_file)
    t.load_spike(spike_file)
    t.plot()

