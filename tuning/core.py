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
            1697, 1718, 17441, 1749, 1811, 1862, 1917, 1995, 2047]) # in seconds 

        self.cue_type = None
        self.cue_time_nidq = None
        self.cue_time_off_nidq = None
        self.cue_time_fpga = None
        self.cue_time_off_fpga = None
        self.cue_duration = None
        self.iti_duration = None
        self.laser_time_nidq = None
        self.laser_time_off_nidq = None
        self.laser_time_fpga = None
        self.laser_time_off_fpga = None
        self.spike_time = None
        self.spike_fr = None
        self.spike_group = None
 

    def load_event(self, event_file): # read event time
        self.event_file = event_file

        # load event file
        event_data = sio.loadmat(self.event_file, squeeze_me=True)
        self.cue_time_nidq = event_data['cue_onset']
        self.cue_time_off_nidq = event_data['cue_offset']
        self.sync_time_nidq = event_data['sync_onset']
        self.laser_time_nidq = event_data['laser_onset']
        self.laser_time_off_nidq = event_data['laser_offset']
        self.event_path, _ = os.path.split(self.event_file)
        self.event_filetime = time.mktime(time.strptime(event_data['fileTime'], '%Y-%m-%dT%H:%M:%S'))

        self.sync_event()


    def sync_event(self): # sync event time
        n_sync = len(self.sync_time_nidq)
        self.sync_time_fpga = self.sync_time_fpga[:n_sync]
        
        # check sync (use 80% of sync pulse)

        n_sync_temp = int(n_sync * 0.8)
        slope, intercept, r_value, _, _ = linregress(self.sync_time_nidq[:n_sync_temp], self.sync_time_fpga[:n_sync_temp])
        print("slope: {}, intercept: {}, r-squared: {}".format(slope, intercept, r_value**2))
        if (r_value**2) < 0.98:
            print("Abnormal sync!!!")
        sync_diff = self.sync_time_nidq * slope + intercept - self.sync_time_fpga
        outlier = sync_diff >= 0.005 # 5 ms
        print("n={} outliers".format(sum(outlier)))
        
        plt.scatter(self.sync_time_nidq[~outlier], self.sync_time_fpga[~outlier], color='k')
        plt.scatter(self.sync_time_nidq[outlier], self.sync_time_fpga[outlier], color='r')
        plt.show()

        self.sync_time_fpga = self.sync_time_fpga[~outlier]
        self.sync_time_nidq = self.sync_time_nidq[~outlier]

        f = interp1d(self.sync_time_nidq, self.sync_time_fpga, fill_value="extrapolate")
        self.cue_time_fpga = f(self.cue_time_nidq)
        self.cue_time_off_fpga = f(self.cue_time_off_nidq)
        self.laser_time_fpga = f(self.laser_time_nidq)
        self.laser_time_off_fpga = f(self.laser_time_off_nidq)

        # remove abnormal signal
        laser_time = np.concatenate([self.laser_time_fpga, self.laser_time_off_fpga])
        laser_type = np.concatenate([np.ones_like(self.laser_time_fpga), np.zeros_like(self.laser_time_off_fpga)])
        idx = np.argsort(laser_time)
        laser_time = laser_time[idx]
        laser_type = laser_type[idx]

        out_on = np.where((np.diff(laser_type) == 0) & (laser_type[:-1] == 1))[0]
        if len(out_on) > 0:
            laser_time = np.delete(laser_time, out_on)
            laser_type = np.delete(laser_type, out_on)

        out_off = np.where((np.diff(laser_type) == 0) & (laser_type[:-1] == 0))[0]
        if len(out_off) > 0:
            laser_time = np.delete(laser_time, out_off+1)
            laser_type = np.delete(laser_type, out_off+1)

        self.laser_time_fpga = laser_time[laser_type==1]
        self.laser_time_off_fpga = laser_time[laser_type==0]


    def load_ptb(self, ptb_file):
        # loading ptb file
        self.ptb_file = ptb_file
        ptb_data = sio.loadmat(self.ptb_file, squeeze_me=True, struct_as_record=False)

        if 'result' in ptb_data.keys():
            self.cue_type = ptb_data['result'].directions
        else:
            self.cue_type = ptb_data['directions']

        self.cue_duration = ptb_data['params'].stimulusDuration
        self.iti_duration = ptb_data['params'].itiStart


    def load_spike(self, spike_file):
        self.spike_file = spike_file
        df = pd.read_pickle(self.spike_file)

        n_unit = int(df['spike_id'].max())
        self.spike_time = np.zeros(n_unit, dtype=object)
        self.spike_fr = np.zeros(n_unit)
        self.spike_group = np.zeros(n_unit, dtype=int)
        
        duration = (df['frame_id'].iloc[-1] - df['frame_id'].iloc[0]) / 25000

        for i_unit in range(n_unit):
            in_unit = df['spike_id'] == (i_unit + 1)
            if sum(in_unit) == 0:
                continue
            self.spike_group[i_unit] = df['group_id'][np.argwhere(in_unit.to_numpy())[0, 0]]
            self.spike_time[i_unit] = df['frame_id'][in_unit].to_numpy() / 25000
            self.spike_fr[i_unit] = sum(in_unit) / duration

    def load_kilosort(self, spike_time, spike_group):
        n_unit = spike_time.size
        self.spike_time = spike_time
        self.spike_fr = np.zeros(n_unit)
        self.spike_group = spike_group
        spkMin = np.Inf
        spkMax = 0
        for i in range(n_unit):
            if spike_time[i].size > 0:
                spkMin = np.min([spkMin, np.min(spike_time[i])])
                spkMax = np.max([spkMax, np.max(spike_time[i])])
        duration = spkMax - spkMin 

        for i_unit in range(n_unit):
            self.spike_fr[i_unit] = spike_time[i_unit].size / duration

    def read_spike(self, spike_time, max_channel):
        """
        spike_time: numpy 1-d array
        """

        n_unit = len(spike_time)
        self.spike_time = spike_time 
        self.spike_fr = np.zeros(n_unit)
        self.spike_group = max_channel 
        
        max_time = np.max([np.max(x) for x in spike_time])
        min_time = np.min([np.min(x) for x in spike_time])
        duration = max_time - min_time 

        for i_unit in range(n_unit):
            self.spike_fr[i_unit] = len(spike_time[i_unit]) / duration


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

        WINDOW = np.array([0, self.cue_duration])
        window_cue = WINDOW + np.array([-self.iti_duration-0.3, self.iti_duration+0.3]) # window for plot in seconds

        cue_time = self.cue_time_fpga
        cue_offset = self.cue_time_off_fpga
        cue_type = self.cue_type
        n_type = len(np.unique(cue_type)) # cue types
        n_unit = len(self.spike_time)

        mystyle(1, n_unit/2)
        linecolor = plt.cm.gist_ncar(np.linspace(0, 0.9, n_type))[:, 0:3]
        f = plt.figure()
        gs0 = gridspec.GridSpec(n_unit, 3, wspace = 0.3, hspace=0.3)
            
        for i_unit in range(n_unit):
            spike_time = self.spike_time[i_unit]
            if isinstance(spike_time, int):
                continue
            raster, psth = spike.plot(spike_time, cue_time, cue_type,
                                      window=window_cue, resolution=5)
            
            # get mean and std firing rate for each stimulus type
            spike_angle_mean = np.zeros(n_type)
            spike_angle_se = np.zeros(n_type)
            for i_type in range(n_type):
                in_type = cue_type == (i_type * 30)
                spike_num_temp = spike.count_spike(spike_time, cue_time[in_type], window=[0.1, WINDOW[1]]) / (WINDOW[1] - 0.1)
                spike_angle_mean[i_type] = np.mean(spike_num_temp)
                spike_angle_se[i_type] = np.std(spike_num_temp) / np.sqrt(len(spike_num_temp))

            # get cumulative plot and latency
            c = spike.get_latency(spike_time, cue_time, cue_offset, prob=0.01)

            # plotting
            y_max = np.max(np.concatenate([psth['conv'].flatten() for i in range(n_type)]))
            gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[i_unit, 0], hspace=0.05)
            gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[i_unit, 1])
            gs02 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[i_unit, 2])
            ax000 = f.add_subplot(gs00[0])
            ax001 = f.add_subplot(gs00[1])
            ax010 = f.add_subplot(gs01[0])
            ax020 = f.add_subplot(gs02[0])

            rect0 = mpl.patches.Rectangle((0, 0), WINDOW[1] - WINDOW[0], len(cue_type), edgecolor='none', facecolor='cyan', alpha=0.2)
            rect1 = mpl.patches.Rectangle((0, 0), WINDOW[1] - WINDOW[0], y_max, edgecolor='none', facecolor='cyan', alpha=0.2)
            ax000.add_patch(rect0)
            ax001.add_patch(rect1)
            for i_type in range(n_type):
                ax000.plot(raster['x'][i_type], raster['y'][i_type], '.', color=linecolor[i_type, :])
                ax001.plot(psth['t'], psth['conv'][i_type, :], color=linecolor[i_type, :])

            ax010.errorbar(np.arange(12)*30, spike_angle_mean, yerr=spike_angle_se)
            fr_max = np.max(spike_angle_mean + spike_angle_se)
            ax010.text(20, fr_max*0.05, 'neuron {} ({}): {:.1f} Hz'.format(i_unit+1, self.spike_group[i_unit], self.spike_fr[i_unit]), fontsize=4)

            ax020.plot(c['time'], c['event'], 'b')
            ax020.plot(c['time'], c['base_mean'], color=[0.5, 0.5, 0.5])
            ax020.plot(c['time'], c['base_up'], ':', color=[0.5, 0.5, 0.5])
            ax020.plot(c['time'], c['base_down'], ':', color=[0.5, 0.5, 0.5])
            cum_max = np.max(np.concatenate([c['event'], c['base_up']]))
            cum_min = np.min(np.concatenate([c['event'], c['base_down']]))
            y_range = cum_max - cum_min
            y_lim = [cum_min-0.1*y_range, cum_max+0.1*y_range]

            if c['latency_up'] is not None:
                ax020.plot(np.repeat(c['latency_up'], 2), y_lim, 'r:')
                ax020.text(0.02, 0.05*(y_lim[1] - y_lim[0]) + y_lim[0], 'latency: {:.1f} ms'.format(c['latency_up']*1000), fontsize=4)

            if c['latency_down'] is not None:
                ax020.plot(np.repeat(c['latency_down'], 2), y_lim, 'r:')
                ax020.text(0.02, 0.05*(y_lim[1] - y_lim[0]) + y_lim[0], 'latency: {:.1f} ms'.format(c['latency_down']*1000), fontsize=4)

            ax020.set_xlim([0, 0.08])
            ax020.set_ylim(y_lim)
            if i_unit < n_unit - 1:
                ax020.set_xticklabels([])

            ax000.set_xlim(window_cue + np.array([0.3, -0.3]))
            ax000.set_ylim([0, len(cue_type)])
            ax000.set_xticklabels([])
            #ax000.set_ylabel('Trial')

            ax001.set_xlim(window_cue + np.array([0.3, -0.3]))
            ax001.set_ylim([0, y_max])
            if i_unit == n_unit - 1:
                ax001.set_xlabel('Time from cue onset (s)')
            else:
                ax001.set_xticklabels([])
            #ax001.set_ylabel('Firing rate')

            ax010.set_xlim([0, 360])
            ax010.set_ylim([0, np.max(spike_angle_mean + spike_angle_se)])
            if i_unit < n_unit - 1:
                ax010.set_xticklabels([])
            else:
                ax010.set_xlabel('Cue direction')

            f.align_ylabels([ax000, ax001])

        return f

    def plot_laser(self):
        if self.spike_time is None:
            print('You have run load_spike()')
            return
        elif self.laser_time_fpga is None:
            print('You have to run load_event()')
            return


        WINDOW = np.array([0.00, 0.02])
        window_cue = WINDOW + np.array([-0.01, 0.01]) # window for plot in seconds

        cue_time = self.laser_time_fpga
        cue_offset = self.laser_time_off_fpga
        n_unit = len(self.spike_time)

        mystyle(0.5, n_unit/2)
        f = plt.figure()
        gs0 = gridspec.GridSpec(n_unit, 2, wspace = 0.3)
            
        for i_unit in range(n_unit):
            spike_time = self.spike_time[i_unit]
            if isinstance(spike_time, int):
                continue
            raster, psth = spike.plot(spike_time, cue_time, np.ones_like(cue_time),
                                      window=window_cue, resolution=5, binsize=0.001)
            
            # get cumulative plot and latency
            c = spike.get_latency(spike_time, cue_time, cue_offset, duration=0.01, offset=0.1, min_latency=0.002, prob=0.01)

            # plotting
            n_trial = len(cue_time)
            y_max = np.max(psth['bar'].flatten())
            gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[i_unit, 0], hspace=0.05)
            gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[i_unit, 1])
            ax000 = f.add_subplot(gs00[0])
            ax001 = f.add_subplot(gs00[1])
            ax010 = f.add_subplot(gs01[0])

            rect0 = mpl.patches.Rectangle((0, 0), WINDOW[1] - WINDOW[0], n_trial, edgecolor='none', facecolor='cyan', alpha=0.2)
            rect1 = mpl.patches.Rectangle((0, 0), WINDOW[1] - WINDOW[0], y_max, edgecolor='none', facecolor='cyan', alpha=0.2)
            ax000.add_patch(rect0)
            ax001.add_patch(rect1)
            ax000.plot(raster['x'][0], raster['y'][0], 'k.')
            ax001.plot(psth['t'], psth['bar'][0, :])

            ax010.plot(c['time'], c['event'], 'b')
            ax010.plot(c['time'], c['base_mean'], color=[0.5, 0.5, 0.5])
            ax010.plot(c['time'], c['base_up'], ':', color=[0.5, 0.5, 0.5])
            ax010.plot(c['time'], c['base_down'], ':', color=[0.5, 0.5, 0.5])
            cum_max = np.max(np.concatenate([c['event'], c['base_up']]))
            cum_min = np.min(np.concatenate([c['event'], c['base_down']]))
            y_range = cum_max - cum_min
            y_lim = [cum_min-0.1*y_range, cum_max+0.1*y_range]

            ax010.text(0.001, 0.95*(y_lim[1] - y_lim[0]) + y_lim[0], 'neuron {} ({}): {:.1f} Hz'.format(i_unit+1, self.spike_group[i_unit], self.spike_fr[i_unit]), fontsize=4)

            if c['latency_up'] is not None:
                ax010.plot(np.repeat(c['latency_up'], 2), y_lim, 'r:')
                ax010.text(0.001, 0.05*(y_lim[1] - y_lim[0]) + y_lim[0], 'latency: {:.1f} ms'.format(c['latency_up']*1000), fontsize=4)

            if c['latency_down'] is not None:
                ax010.plot(np.repeat(c['latency_down'], 2), y_lim, 'r:')
                ax010.text(0.001, 0.05*(y_lim[1] - y_lim[0]) + y_lim[0], 'latency: {:.1f} ms'.format(c['latency_down']*1000), fontsize=4)

            ax010.set_xlim([0, 0.01])
            ax010.set_ylim(y_lim)
            if i_unit < n_unit - 1:
                ax010.set_xticklabels([])

            ax000.set_xlim(window_cue)
            ax000.set_ylim([0, n_trial])
            ax000.set_xticklabels([])
            #ax000.set_ylabel('Trial')

            ax001.set_xlim(window_cue)
            ax001.set_ylim([0, y_max])
            if i_unit == n_unit - 1:
                ax001.set_xlabel('Time from light onset (s)')
            else:
                ax001.set_xticklabels([])
            #ax001.set_ylabel('Firing rate')

            f.align_ylabels([ax000, ax001])

        return f


       

def mystyle(xratio=1, yratio=1):
    mpl.rcParams['figure.figsize'] = [4.25*xratio, 1.59375*yratio]
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['font.size'] = 6

    mpl.rcParams['figure.subplot.hspace'] = 0.075

    mpl.rcParams['axes.titlesize'] = 8
    mpl.rcParams['axes.labelsize'] = 4
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    mpl.rcParams['xtick.labelsize'] = 4
    mpl.rcParams['ytick.labelsize'] = 4
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['lines.markersize'] = 1
    mpl.rcParams['lines.markeredgewidth'] = 0


if __name__ == "__main__":
    event_file = '/mnt/data/bmi_20191210_0/event/bmi_20191210_g2_t0_data.mat'
    ptb_file = '/mnt/data/bmi_20191210_0/event/ptb_20191210_172855.mat'
    spike_file = '/mnt/data/bmi_20191210_0/spktag/test.pd'
    
    #event_file = '/mnt/data/bmi_20191210_0/event/bmi_20191210_g4_t0_data.mat'
    t = Tuning()
    t.load_event(event_file)

    t.load_ptb(ptb_file)
    t.load_spike(spike_file)
    t.plot()

