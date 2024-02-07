'''
Created by CG and JL on 20230505
Collection of functions to do complex averaging (C-avg) of ringdown data.0
'''

# Standard library imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["savefig.facecolor"] = 'white'
import matplotlib.colors as mcolors
import numpy.random as npr
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import pickle
import time
import csv
import warnings

# Import custom code stored in the NewEP3Exp/Code directory
import sys
sys.path.append(sys.path[0].split('NewEP3Exp')[0] + 'NewEP3Exp/Code')
sys.path.append(sys.path[0].split('NewEP3Exp')[0] + 'NewEP3Exp/Data/routine logs/calibration coefficients')
import file_handling 
import fitting.standard_fit_routines as fit
import fitting.Rotating_frame_fit_routines as rotfit

'''
class 1
this is to be used when we save all data from LIA
'''
class extract_data(object):
    
    def __init__(self, data, dataspecs, savefolder, measurement_type = 'simple ringdown', stack_spec = [0.01, 0.01, 0, 0]):
        '''
        This gets all data and data specs in-situ, makes a folder to save C-avg-ed data.
        '''

        #distribute dataspecs
        self.unpack_dataspecs(dataspecs, measurement_type)
        [self.start_trimT, self.k2, self.c1, self.c2] = stack_spec
        self.tLIA = self.settling_time(self.bandwidth, self.order, signal_level = 99)
        self.shift = self.c2*self.tLIA + 1*self.t_wait
        
        #load data and kill nans
        self.dat = data
        [self.minv, self.maxv] = self.killnans()
        
        #make arrays to operate on
        self.time = self.dat[:,0] - self.dat[0,0]#timestamps
        self.TTL1 = self.dat[:,1]#booled TTL1 - for ringdowns
        self.TTL2 = self.TTL1
        self.demod2X = self.dat[:,3]*1e3
        self.demod2Y = self.dat[:,4]*1e3
        self.demod5X = self.dat[:,5]*1e3
        self.demod5Y = self.dat[:,6]*1e3
        self.C2 = self.demod2X + 1j*self.demod2Y
        self.C5 = self.demod5X + 1j*self.demod5Y
        
        
        #distribute save folder
        self.save_folder = savefolder

        
        return
    def unpack_dataspecs(self, dataspecs, measurement_type):
        if measurement_type == "simple ringdown":
            if len(dataspecs) == 9:
                warnings.warn("Some DAq specifications passed to the simple ringdown averager have been depreciated \n use [demod2F, deod5F, drive time, rest time, bandwidth, order, wait time]")
                self.demod2f, self.demod5f, _, self.setdrivetime, self.setresttime, self.bandwidth, self.order, _, self.t_wait = dataspecs
            elif len(dataspecs) == 7:
                self.demod2f, self.demod5f, self.setdrivetime, self.setresttime, self.bandwidth, self.order, self.t_wait = dataspecs
            else:
                raise ValueError("data specification array isn't a valid length")
        else:
            raise ValueError("We're not setup to average that data yet")



    def killnans(self):
        '''
        Function for removing nan-s
        Pervious version fails is there's data revival (i.e. nans before valid data)
        This version grabs the largest single chunk of non nan-y data
        '''
        start, end = self.largest_non_nan_chunk() # get the first and last indicies of largest non-NaN block
        self.dat = self.dat[start:end, :]
        if start == 0 and end == len(self.dat[:,0]):
            self.check3 = 0
        else:
            self.check3 = 1
        return [start+1, end+1]

    def trim_driveOFF_markers(self):
        '''
        Inferring what's going on.
        Gets the indicies at which the drive TTL turns off
        If that happens before absolute time 10 ms, throw it away. 
        '''
        d = np.diff(self.time)
        if any(d) > 0.01:
            raise warnings.warn("There's a jump in your data, averaging is going to fail")
        dtdi = np.average(d) # average time that passes per index
        min_index = int(np.round(self.start_trimT/dtdi)) # if a marker is before start_trimT, we throw it out
        driveOFF_indicies = np.where((np.diff(self.TTL2) == -1) == 1)[0]
        driveOFF_indicies = driveOFF_indicies[driveOFF_indicies>min_index] # clip off any marker that might come before min_index

    def simple_ringdown(self):
        '''
        Function to assemble all simple ringdown data appropriately and do C-avg-ing.
        '''

        #sort out remnant bugs in data
        self.tempS = np.where((np.diff(self.TTL2) == -1)*1 == 1)[0] # indicies where we turn the drive off

        # apparently indicies self.tempS shifted forwards in time by 0.01 seconds
        # I think the point of this block of code is to add a 0.01 second buffer on the beginning of a ringdown?
        self.tempS2 = self.ift(self.time, self.tfi(self.time, self.tempS, self.minv, self.maxv)-self.start_trimT, self.minv, self.maxv) 
        self.starterS = self.tempS2[self.tempS2 > 0]
        self.casesS = np.where(self.tempS2 < 0)[0] 
        
        #markers for simple ringdowns
        self.tempS3 = np.where((np.diff(self.TTL2) == -1)*1 == 1)[0] + self.minv -1 -1 # what is this?
        self.iloopstartS = np.delete(self.tempS3, self.casesS)
        self.tloopstartS = self.tfi(self.time, self.iloopstartS, self.minv, self.maxv) + self.t_wait
        self.tLIAstartS = self.tloopstartS
        self.tLIAendS = self.tLIAstartS + self.tLIA
        self.t1stackS = self.tfi(self.time, self.starterS, self.minv, self.maxv) # literally not used anywhere
        self.i1stackS = self.starterS
        self.tDAqendS = self.tloopstartS + self.setresttime
        self.t2stackS = self.tDAqendS - self.k2
        self.i2stackS = self.ift(self.time, self.t2stackS, self.minv, self.maxv)
        #to avoid recent C-avg-ing BS
        self.i2stackS = self.i2stackS[self.i2stackS > self.i1stackS[0]]
        self.i2stackS = self.i2stackS[0:len(self.i1stackS)]
        
        self.devS = np.std(self.i2stackS - self.i1stackS)
        self.looplen = len(self.i2stackS) - 3 # do not touch this line

        
        #stacking simple ringdowns
        self.initstackS = np.stack((self.time, self.TTL2, self.C2, self.C5), axis = -1)
        self.stackS = np.zeros((self.looplen, (self.i2stackS - self.i1stackS)[0] + 1, 4), dtype = 'complex_')
        self.phasestackS = np.zeros((self.looplen, (self.i2stackS - self.i1stackS)[0] + 1, 3))
        
        self.stackS[:,:,0] = self.initstackS[self.i1stackS[0]:self.i2stackS[0] + 1, 0]
        self.stackS[:,:,1] = self.initstackS[self.i1stackS[0]:self.i2stackS[0] + 1, 1]
        self.phasestackS[:,:,0] = np.real(self.initstackS[self.i1stackS[0]:self.i2stackS[0] + 1, 0])
        
        self.elements = [i for i in range(self.looplen)]
        for e in self.elements:
            self.stackS[e,:,2] = self.initstackS[self.i1stackS[e]:self.i2stackS[e] + 1, 2]
            self.stackS[e,:,3] = self.initstackS[self.i1stackS[e]:self.i2stackS[e] + 1, 3]
        for e in self.elements:
            self.stackS[e,:,2] = self.stackS[e,:,2]/(np.exp(+1j*np.angle(self.stackS[e,self.ift(self.time, self.start_trimT - self.time[0], self.minv, self.maxv),2])))
            self.stackS[e,:,3] = self.stackS[e,:,3]/(np.exp(+1j*np.angle(self.stackS[e,self.ift(self.time, self.start_trimT - self.time[0] + self.shift, self.minv, self.maxv),3])))
            self.phasestackS[e,:,1] = np.unwrap(np.angle(self.stackS[e,:,2]))
            self.phasestackS[e,:,1] = (self.phasestackS[e,:,1] - self.phasestackS[e,self.ift(self.time, self.start_trimT - self.time[0], self.minv, self.maxv) - self.minv + 1 + 1,1])/(2*np.pi)
            self.phasestackS[e,:,2] = np.unwrap(np.angle(self.stackS[e,:,3]))
            self.phasestackS[e,:,2] = (self.phasestackS[e,:,2] - self.phasestackS[e,self.ift(self.time, self.start_trimT - self.time[0] + self.shift, self.minv, self.maxv) - self.minv + 1 + 1,2])/(2*np.pi)
        
        #do C-avg
        self.meanmode1S = (np.sum(self.stackS[:,:,2], axis = 0))/self.looplen
        self.absmode1S = np.abs(self.meanmode1S)
        self.argmode1S = np.unwrap(np.angle(self.meanmode1S))
        self.argmode1S = self.argmode1S - self.argmode1S[self.ift(self.time, self.start_trimT - self.time[0], self.minv, self.maxv) - self.minv + 1 + 1]
        self.meanmode2S = (np.sum(self.stackS[:,:,3], axis = 0))/self.looplen
        self.absmode2S = np.abs(self.meanmode2S)
        self.argmode2S = np.unwrap(np.angle(self.meanmode2S))
        self.argmode2S = self.argmode2S - self.argmode2S[self.ift(self.time, self.start_trimT - self.time[0] + self.shift, self.minv, self.maxv) - self.minv + 1 + 1]
        
        temp = np.stack((np.abs(self.stackS[0,:,0]), np.abs(self.stackS[0,:,1]), self.absmode1S, self.argmode1S/ (2*np.pi), self.absmode2S, self.argmode2S/(2*np.pi)),axis = -1)
        return self.stackS, self.phasestackS, temp, self.elements
    



    
     
    
    def return_Cavg_data(self, filename, savefolder, save = True):
        '''
        Calls simple_ringdown and loop_ringdown functions and collates both (simple and loop ringdowns) of the C-avg-ed data in an array. 
        This array is supplemented with relevant headers and saved.
        '''

        stackS, phasestackS, tempS, elementsS = self.simple_ringdown()
        
        temptot = np.stack((tempS[:,0], tempS[:,1], tempS[:,2], tempS[:,3], tempS[:,4], tempS[:,5]),axis = -1)
        
        l1 = "demod2 (Hz) = {}".format(self.demod2f)+"\n"
        l2 = "demod5 (Hz) = {}".format(self.demod5f)+"\n" 
        
        l4 = "drive time (s) = {}".format(self.setdrivetime)+"\n" 
        l5 = "rest time (s) = {}".format(self.setresttime)+"\n"
        l6 = "LIA filter order = {}".format(int(self.order))+"\n"
        l7 = "LIA BW (Hz) = {}".format(self.bandwidth)+"\n"
        l8 = "LIA response time (s) = {}".format(self.tLIA)+"\n"
        l9 = "check 1 = {}".format(self.devS)+"\n"
        l10 = "check 2 = {}".format(((len(self.casesS)>1)*1))+"\n"
        l10a = "check 3 = {}".format(self.check3)+"\n"
        l11 = "# valid loops = {}".format(self.looplen)+"\n"
        l11a = "t_wait (s) = {}".format(self.t_wait)+"\n"
        l12 = "time (simple, s), TTL (simple) , abs demod 2 (simple, mV), arg demod 2 (simple, 2pi*rad), abs demod 5 (simple, mV), arg demod 5 (simple. 2pi*rad)"
        
        preamble = l1+l2+l4+l5+l6+l7+l8+l9+l10+l10a+l11+l11a+l12
        
        if save == True:
            save_filename, _ = file_handling.make_filename(filename, savefolder, 'csv')
            file_handling.savetxtdate(save_filename, temptot, delimiter = ',', header = preamble)
        
        return temptot
    


    def plot_simple(self, filename, savefolder, plot = False, save = True):
        '''
        Plots simple ringdowns with all bells and whistles, with options to show plot and save. Slow af!
        '''

        #import what is to be plotted
        stackS, phasestackS, tempS, elementsS = self.simple_ringdown()
        
        #set initial plot specs
        fsz = 10
        wid = 1
        lwid = 0.75
        plwid = 2.75
        thwid = 1
        mpl.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = fsz
        plt.rcParams['axes.linewidth'] = wid
        mpl.rcParams['axes.formatter.useoffset'] = False
        
        #make canvas
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(13,6),facecolor='white',sharex='col',
                            gridspec_kw=dict({'height_ratios': [1, 1]}, hspace=0.05,wspace = 0.15))

        #fill up canvas

        #demod 2 abs plot
        axs[0,0].xaxis.set_tick_params(which='major', size=5, labelsize = fsz, width=wid, direction='in', top='on', pad = 5)
        axs[0,0].yaxis.set_tick_params(which='major', size=5, labelsize = fsz, width=wid, direction='in', right='on', pad = 5)
        for e in self.elements:
            axs[0,0].plot(tempS[:,0], np.abs(stackS[e,:,2]), color='darkturquoise', linewidth = thwid, alpha = 0.1)
        axs[0,0].plot(tempS[:,0], tempS[:,2], color='teal', linewidth = plwid, label = 'demod 2 data')
        axs[0,0].axvspan(self.tLIAstartS[0], self.tLIAendS[0], facecolor='#B0E2FF', alpha=0.45, zorder=2.5, label = 'LIA window')
        axs[0,0].autoscale(enable=True, axis='both', tight=False)
        axs[0,0].grid(True, color = 'lightgray', linewidth = lwid)
        axs[0,0].set_yscale("log")
        axs[0,0].set_ylabel("Abs[response] (mV)", labelpad=10)
        axs[0,0].legend(fontsize = fsz, loc = 'best', frameon = True)

        axs[0,0].minorticks_off()
        axs[0,0].margins(0.025, 0.05)
        
        #demod 2 arg plot
        axs[1,0].xaxis.set_tick_params(which='major', size=5, labelsize = fsz, width=wid, direction='in', top='on', pad = 5)
        axs[1,0].yaxis.set_tick_params(which='major', size=5, labelsize = fsz, width=wid, direction='in', right='on', pad = 5)
        for e in self.elements:
            axs[1,0].plot(tempS[:,0], phasestackS[e,:,1], color='darkturquoise', linewidth = thwid, alpha = 0.1)
        axs[1,0].plot(tempS[:,0], tempS[:,3], color='teal', linewidth = plwid)
        axs[1,0].axvspan(self.tLIAstartS[0], self.tLIAendS[0], facecolor='#B0E2FF', alpha=0.45, zorder=2.5)
        axs[1,0].autoscale(enable=True, axis='both', tight=False)
        axs[1,0].grid(True, color = 'lightgray', linewidth = lwid)
        axs[1,0].set_ylabel("Arg[response] (2$\pi$)", labelpad=10)
        axs[1,0].set_xlabel("time (s)", labelpad=10)
        axs[1,0].minorticks_off()
        axs[1,0].margins(0.025, 0.05)

        #demod 5 abs plot
        axs[0,1].xaxis.set_tick_params(which='major', size=5, labelsize = fsz, width=wid, direction='in', top='on', pad = 5)
        axs[0,1].yaxis.set_tick_params(which='major', size=5, labelsize = fsz, width=wid, direction='in', right='on', pad = 5)
        for e in self.elements:
            axs[0,1].plot(tempS[:,0], np.abs(stackS[e,:,3]), color='darkorange', linewidth = thwid, alpha = 0.1)
        axs[0,1].plot(tempS[:,0], tempS[:,4], color='sienna', linewidth = plwid, label = 'demod 5 data')
        axs[0,1].axvspan(self.tLIAstartS[0], self.tLIAendS[0], facecolor='#B0E2FF', alpha=0.45, zorder=2.5)
        axs[0,1].autoscale(enable=True, axis='both', tight=False)
        axs[0,1].grid(True, color = 'lightgray', linewidth = lwid)
        axs[0,1].legend(fontsize = fsz, loc = 'best', frameon = True)
        axs[0,1].set_yscale("log")
        axs[0,1].minorticks_off()
        axs[0,1].margins(0.025, 0.05)

        #demod 5 arg plot
        axs[1,1].xaxis.set_tick_params(which='major', size=5, labelsize = fsz, width=wid, direction='in', top='on', pad = 5)
        axs[1,1].yaxis.set_tick_params(which='major', size=5, labelsize = fsz, width=wid, direction='in', right='on', pad = 5)
        for e in self.elements:
            axs[1,1].plot(tempS[:,0], phasestackS[e,:,2], color='darkorange', linewidth = thwid, alpha = 0.1)
        axs[1,1].plot(tempS[:,0], tempS[:,5], color='sienna', linewidth = plwid)
        axs[1,1].axvspan(self.tLIAstartS[0], self.tLIAendS[0], facecolor='#B0E2FF', alpha=0.45, zorder=2.5)
        axs[1,1].autoscale(enable=True, axis='both', tight=False)
        axs[1,1].grid(True, color = 'lightgray', linewidth = lwid)
        axs[1,1].set_xlabel("time (s)", labelpad=10)
        axs[1,1].minorticks_off()
        axs[1,1].margins(0.025, 0.05)

        #save or not
        if save == True:
            plotname, _ = file_handling.make_plotfilename(filename,savefolder)
            plt.savefig(plotname, dpi = 400, bbox_inches='tight')
        
        #show or not
        if plot == False:
            plt.close()

        return
    


    






    #various helper functions
    def xstr(self, obj):
        #makes a string
        if obj is None:
            return ""
        else:
            return str(obj)


    def bw2tc(self, bandwidth, filterorder):
        '''
        calculate the lockin timeconstant from the bandwidth and the filter order
        scaling factor taken from LIA programming manual, directly corresponds to the 
        Low-Pass Filter order on LIA gui 
        Bandwidth is in Hz
        '''
        scale = np.array([1.0, 0.643594, 0.509825, 0.434979, 0.385614, 0.349946, 0.322629, 0.300845])[int(filterorder)-1]

        timeconstant = scale/(2*np.pi*bandwidth)
        return timeconstant 

    def settling_time(self, bandwidth, filterorder, signal_level = 99):
        '''
        Calculate the time after the loop ends to reject data.
        Settling times taken from lockin manual
        Assumes we want to wait until the signal has reached 
        90%, 95%, or 99% of it's steady state value
        Bandwidth is in Hz
        '''
        tau = self.bw2tc(bandwidth, int(filterorder))
        
        if signal_level == 90:
            wait90 = [2.3, 3.89, 5.32, 6.68, 7.99, 9.27, 10.53, 11.77]
            return tau*wait90[int(filterorder)-1]
        elif signal_level == 95:
            wait95 = [3, 4.7, 6.3, 7.8, 9.2, 11, 12, 13]
            return tau*wait95[int(filterorder)-1]
        elif signal_level == 99:
            wait99 = [4.61, 6.64, 8.41, 10.05, 11.60, 13.11, 14.57, 16]
            return tau*wait99[int(filterorder)-1]
        else:
            print("invalid signal level!")
            return 0
        
    def tfi(self, array, i, minv, maxv):
        #given an array index i, find the corresponding timestamp, with all safety measures in place.
        alen = len(array)
        slope = (array[alen-1]-array[1-1])/(maxv-minv)
        off = (array[1-1]*maxv-array[alen-1]*minv)/(maxv-minv)
        
        return slope*(i+1) + off

    def ift(self, array, t, minv, maxv):
        #given a timestamp, find the corresponding array index, with all safety measures in place.
        alen = len(array)
        slope = (maxv-minv)/(array[alen-1]-array[1-1])
        off = (array[alen-1]*minv - array[1-1]*maxv)/(array[alen-1]-array[1-1])

        return (np.rint(slope*t + off) - 1).astype(int)

    def marker_to_bool(self, markerdat):
        # Convert maker data from floating point to boolean so we're not saving a bunch of useless data
        return np.where(markerdat > 2, 1, 0)
    
    
    def largest_non_nan_chunk(self):
        # Collapse self.dat onto a 1D array
        arr = np.sum(self.dat, axis = 1)
        # Create an array that is True where arr is not NaN, and False where arr is NaN
        mask = np.isfinite(arr)
        # Use np.diff to find the indices where the mask changes
        diff = np.diff(mask.astype(int))
        # The start indices of the chunks are one more than the indices where diff is 1
        starts = np.where(diff == 1)[0] + 1
        # The stop indices of the chunks are the indices where diff is -1
        stops = np.where(diff == -1)[0]
        # If the first element is not NaN, then the first chunk starts at index 0
        if mask[0]:
            starts = np.r_[0, starts]
        # If the last element is not NaN, then the last chunk stops at the last index
        if mask[-1]:
            stops = np.r_[stops, len(arr) - 1]
        # Find the lengths of the chunks
        lengths = stops - starts
        # Find the index of the longest chunk
        longest_chunk_index = np.argmax(lengths)
        return starts[longest_chunk_index], stops[longest_chunk_index]


