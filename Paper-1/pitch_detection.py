# Importing the libraries

# Python in-built modules
import os
import glob
import time

# External modules
import librosa
import librosa.display
import scipy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker


class Pitch_Detect:

    def __init__(self, y, sr, Ws):
        '''
        Arguments:
        y - sampled audio signal
        sr - sampling frequency (Hz)
        lag - delay
        Ws - window size

        '''
        self.y = y
        self.sr = sr
        self.N = Ws


    def hpf(self, filter_stop_freq=50, filter_pass_freq=200, filter_order=1001):
        '''
        High-pass filter (to remove low frequency sounds not under consideration)
        
        Arguments:
        filter_stop_freq - Cut-off frequency band min (Hz)
        filter_pass_freq - Cut-oof frequency band max (Hz)
        filter_order - Order of the filter

        Returns:
        filtered_audio
        
        '''

        # High-pass filter params
        nyquist_rate = self.sr / 2.
        desired = (0, 0, 1, 1)
        bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
        filter_coefs = scipy.signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

        # Apply high-pass filter
        filtered_audio = scipy.signal.filtfilt(filter_coefs, [1], self.y)
        return filtered_audio


    def camdf(self, y_clip, tau):
        '''
        Pitch detection algorithm - Circular Average Magnitude Difference Function (CAMDF).
        Time domain technique.
        Avoids computationally intensive multiplications involved in Correlation by doing additions instead

        Arguments:
        y_clip - Portion of audio samples inside the window size, Ws
        tau - lag or delay

        Returns:
        D - CAMDF value for the particular lag value

        '''
        D = 0.0
        for n in range(self.N):
            D += abs(y_clip[(n + tau)%self.N] - y_clip[n])
        return D


    def pitch_curve(self):
        '''
        Plots the pitch curve at different lag values. The minimum point corresponds to the pitch
        of the audio segment

        Returns:
        pitch_list - pitch corresponding to each window
        ran - number of windows

        '''
        l = self.N
        N = len(self.y[:l+1])
        pitch_list = []
        ran = (len(self.y)//(l//2))-2
        
        for i in range(ran):
            camdf_list = []
            y_clip = self.y[(l//2)*i:(l//2)*i+l+1]
            for i in range(l):
                camdf_list.append(self.camdf(y_clip=y_clip, tau=i))
            interval = camdf_list[4:100]
            min_D = min(interval)
            pitch_detected = round(self.sr/(interval.index(min_D)+4),2)
            pitch_list.append(pitch_detected)
        
        return pitch_list, ran

    def plot_figure(self, pitch_list, ran):
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111)
        ax.plot(pitch_list)
        scale_x = ran/10.0
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
        ax.xaxis.set_major_formatter(ticks_x)
        ax.set_xlabel("time (s)")
        ax.set_ylabel('Pitch (hz)')
        plt.grid('on')
        plt.tight_layout()
        plt.show()


    def run(self):
        y = self.hpf(filter_stop_freq=50, filter_pass_freq=200, filter_order=1001)
        pitch_list, ran = self.pitch_curve()
        
        self.plot_figure(pitch_list, ran)