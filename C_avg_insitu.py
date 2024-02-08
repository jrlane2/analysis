'''
Created by CG and JL on 20230505
Collection of functions to do complex averaging (C-avg) of ringdown data.0
'''

# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["savefig.facecolor"] = 'white'
import matplotlib.colors as mcolors
import numpy.random as npr
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import warnings
import numbers

# Import custom code stored in the NewEP3Exp/Code directory
import sys
sys.path.append(sys.path[0].split('NewEP3')[0] + 'NewEP3Exp/Code')
sys.path.append(sys.path[0].split('NewEP3')[0] + 'NewEP3Exp/Data/routine logs/calibration coefficients')
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
        # define measurement type
        self.measurement_type = measurement_type

        # distribute dataspecs
        self.unpack_dataspecs(dataspecs)
        [self.start_trimT, self.end_trimT, self.c1, self.c2] = stack_spec
        self.tLIA = settling_time(self.bandwidth, self.order, signal_level = 99) # get lockin settling time
        # since there's no phase coherent signal during the drive, wait for time self.shift before aligning demod 5 phase
        self.shift = self.c2*self.tLIA + 1*self.t_wait 
        
        #load data and kill nans
        self.dat = data
        [self.minv, self.maxv] = self.killnans()
        
        #make arrays to operate on
        self.time = self.dat[:,0] - self.dat[0,0] # timestamps
        self.TTL1 = self.dat[:,1] #TTL1, converted to booleran - for interleaved control loops OR simple rigndown
        self.TTL2 = self.dat[:,2] #TTL1, converted to booleran - for interleaved ringdowns
        self.demod2X = self.dat[:,3]*1e3
        self.demod2Y = self.dat[:,4]*1e3
        self.demod5X = self.dat[:,5]*1e3
        self.demod5Y = self.dat[:,6]*1e3
        self.C2 = self.demod2X + 1j*self.demod2Y # complex data from demod 2
        self.C5 = self.demod5X + 1j*self.demod5Y # complex data from demod 5
        
        
        #distribute save folder
        self.save_folder = savefolder
        return
    
    def simple_ringdown(self):
        '''
        Wrapper around average_ringdown for "simple" ringdowns"
        By simple ringdown, I mean any measurement that isn't interleaved (For example, bare membrane checks or sheets)
        In that case, TTL1 is the marker and TTL2 doesn't do anything
        '''
        self.setlooptime = 0
        self.setpreptime = 0
        self.average_ringdown(1)

        return


    def average_ringdown(self, TTL):
        '''
        Function to assemble all simple ringdown data appropriately and do C-avg-ing.
        Assemble and average all the data triggered by the TTL signal going to channel TTL
        '''

        # Get the indicies at which the drive turns off and the indicies at which we start individual time traces
        self.get_start_indicies(TTL)

        # this is how long an individual shot is in time
        indv_trace_time = self.start_trimT + self.setresttime - self.end_trimT + self.t_wait + self.setlooptime + self.setpreptime

        # and this is how long it is in indecies
        indv_trace_len = int(np.rint(indv_trace_time/self.dtdi)) + 1

        # This is how long we have to wait for the demod 5 to catch up
        self.demod5_index_wait = int(np.rint(self.shift/self.dtdi))
        demod5_align_index  = self.TTL_avg_off + self.demod5_index_wait


        # stack em'!
        # The phase of the demod 2 stack is aligned to the end of the drive, which is indexed by TTL off
        self.C2stack = stackem_aligned(self.C2, self.start_avg_index, indv_trace_len, phasealign_index = self.TTL_avg_off)
        # The phase of the demod 5 stack is aligned to slightly afterwords, since there's no phase coherent signal during the drive
        self.C5stack = stackem_aligned(self.C5, self.start_avg_index, indv_trace_len, phasealign_index = demod5_align_index)

        #do averaging
        self.C2avg = np.average(self.C2stack, axis = 1)
        self.C2abs = np.abs(self.C2avg) 
        self.C2arg = np.unwrap(np.angle(self.C2avg)) - np.unwrap(np.angle(self.C2avg))[self.TTL_avg_off] # this should only subtract off 2pi*n
        self.C5avg = np.average(self.C5stack, axis = 1)
        self.C5abs = np.abs(self.C5avg)
        self.C5arg = np.unwrap(np.angle(self.C5avg)) - np.unwrap(np.angle(self.C5avg))[demod5_align_index] # this should only subtract off 2pi*n

        # other convenient arrays
        self.time_avg = self.time[self.start_avg_index[0]:self.start_avg_index[0]+indv_trace_len]
        if TTL == 1:
            self.TTL_avg = self.TTL1[self.start_avg_index[0]:self.start_avg_index[0]+indv_trace_len]
        elif TTL == 2:
            self.TTL_avg = self.TTL2[self.start_avg_index[0]:self.start_avg_index[0]+indv_trace_len]
        
        return 




    
     
    
    def return_Cavg_simple_ringdown(self, filename, savefolder, save = True):
        '''
        Calls simple_ringdown and loop_ringdown functions and collates both (simple and loop ringdowns) of the C-avg-ed data in an array. 
        This array is supplemented with relevant headers and saved.
        '''

        # tempS = self.simple_ringdown()
        
        # temptot = np.stack((tempS[:,0], tempS[:,1], tempS[:,2], tempS[:,3], tempS[:,4], tempS[:,5]),axis = -1)
        if hasattr(self, 'time_avg') == False:
            self.simple_ringdown()
        
        save_data = np.real(np.stack((self.time_avg, self.TTL_avg, self.C2abs, self.C2arg/(2*np.pi), self.C5avg, self.C5arg/(2*np.pi)),axis = -1))
        
        l1 = "demod2 (Hz) = {}".format(self.demod2f)+"\n"
        l2 = "demod5 (Hz) = {}".format(self.demod5f)+"\n" 
        l3 = "drive time (s) = {}".format(self.setdrivetime)+"\n" 
        l4 = "rest time (s) = {}".format(self.setresttime)+"\n"
        l5 = "LIA filter order = {}".format(int(self.order))+"\n"
        l6 = "LIA BW (Hz) = {}".format(self.bandwidth)+"\n"
        l7 = "LIA response time (s) = {}".format(self.tLIA)+"\n"
        l8 = "NaNs in data = {}".format(bool(self.check3))+"\n"
        l9 = "# valid loops = {}".format(np.shape(self.C5stack)[1])+"\n"
        l10 = "t_wait (s) = {}".format(self.t_wait)+"\n"
        l11 = "time (simple, s), TTL (simple) , abs demod 2 (simple, mV), arg demod 2 (simple, 2pi*rad), abs demod 5 (simple, mV), arg demod 5 (simple. 2pi*rad)"
        
        preamble = l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11

        if save == True:
            save_filename, _ = file_handling.make_filename(filename, savefolder, 'csv')
            file_handling.savetxtdate(save_filename, save_data, delimiter = ',', header = preamble)
        
        return save_data
    


    def plot_simple_ringdown(self, filename, savefolder, plot = False, save = True):
        '''
        Plots simple ringdowns with all bells and whistles, with options to show plot and save. Slow af!
        '''

        #import what is to be plotted
        if hasattr(self, 'time_avg') == False:
            self.simple_ringdown()

        # Markers for plotting for plotting
        self.tloopstartS = self.time[self.driveOFF_index] + self.t_wait # where we're calling the start of the loop
        self.tLIAstartS = self.tloopstartS # Start maker for lockin wait window
        self.tLIAendS = self.tLIAstartS + self.tLIA # End Marker for lockin wait window

        # number of neuron plots, i.e. number of constituents in the average
        elements = np.shape(self.C5stack)[1]




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
        for e in range(elements):
            axs[0,0].plot(self.time_avg, np.abs(self.C2stack[:,e]), color='darkturquoise', linewidth = thwid, alpha = 0.1)
        axs[0,0].plot(self.time_avg, self.C2abs, color='teal', linewidth = plwid, label = 'demod 2 data')
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
        for e in range(elements):
            # subtract off annoying multiples of 2pi
            phze = np.unwrap(np.angle(self.C2stack[:,e])) - np.unwrap(np.angle(self.C2stack[:,e]))[self.TTL_avg_off]
            axs[1,0].plot(self.time_avg, phze/(2*np.pi), color='darkturquoise', linewidth = thwid, alpha = 0.1)
        axs[1,0].plot(self.time_avg, self.C2arg/(2*np.pi), color='teal', linewidth = plwid)
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
        for e in range(elements):
            axs[0,1].plot(self.time_avg, np.abs(self.C5stack[:,e]), color='darkorange', linewidth = thwid, alpha = 0.1)
        axs[0,1].plot(self.time_avg, self.C5abs, color='sienna', linewidth = plwid, label = 'demod 5 data')
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
        for e in range(elements):
            # subtract off annoying multiples of 2pi
            demod5_align_index  = self.TTL_avg_off + self.demod5_index_wait
            phze = np.unwrap(np.angle(self.C5stack[:,e])) - np.unwrap(np.angle(self.C5stack[:,e]))[demod5_align_index]
            axs[1,1].plot(self.time_avg , phze/(2*np.pi), color='darkorange', linewidth = thwid, alpha = 0.1)
        axs[1,1].plot(self.time_avg , self.C5arg/(2*np.pi), color='sienna', linewidth = plwid)
        axs[1,1].axvspan(self.tLIAstartS[0], self.tLIAendS[0], facecolor='#B0E2FF', alpha=0.45, zorder=2.5)
        axs[1,1].autoscale(enable=True, axis='both', tight=False)
        axs[1,1].grid(True, color = 'lightgray', linewidth = lwid)
        axs[1,1].set_xlabel("time (s)", labelpad=10)
        axs[1,1].minorticks_off()
        axs[1,1].margins(0.025, 0.05)

        #save or not
        if save == True:
            plotname, _ = file_handling.make_plotfilename(filename,savefolder)
            plt.savefig(plotname, dpi = 150, bbox_inches='tight')
        
        #show or not
        if plot == False:
            plt.close()

        return
    


    


    # Auxillary atribute functions
    
    def unpack_dataspecs(self, dataspecs):
        '''
        Unpacks the data specifications and sets them as class attributes
        '''
        if self.measurement_type == "simple ringdown":
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

    def get_start_indicies(self, TTL):
        '''
        Gets the indicies at which the drive TTL turns off
        If that happens before absolute time start_trimT (default 10 ms), throw it away. 
        Returns indicies where the drive turns off, and the beginning indicies of
        each ringdown to be averaged
        '''

        d = np.diff(self.time)
        if any(d > 0.01):
            raise warnings.warn("There's a jump in your data, averaging is going to fail")
        self.dtdi = np.average(d) # average time that passes per index

        self.TTL_avg_off = int(np.rint(self.start_trimT/self.dtdi)) # if a marker is before start_trimT, we throw it out

        if TTL == 1:
            driveOFF_indicies = np.where((np.diff(self.TTL1) == -1) == 1)[0] # indicies marking t = 0 (drive off)
        elif TTL == 2:
            driveOFF_indicies = np.where((np.diff(self.TTL2) == -1) == 1)[0] # indicies marking t = 0 (drive off)
        else:
            raise ValueError("We're not setup to average that data yet")

        self.driveOFF_index = driveOFF_indicies[driveOFF_indicies > self.TTL_avg_off + 1] # clip off any marker that might come before min_index
        self.start_avg_index = self.driveOFF_index - self.TTL_avg_off # indicies marking the data we start averaging

        return

    def ift(self, tstamp):
        #given a timestamp, find the corresponding array index
        if type(tstamp) == np.ndarray:
            return np.array([np.argmin(np.abs(self.time - i)) for i in tstamp])
        elif is_number(tstamp):
            return np.argmin(np.abs(self.time - tstamp))
        else:
            raise ValueError("tstamp is not a number or array of numbers")

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












#various helper functions
    












def xstr(obj):
    #makes a string
    if obj is None:
        return ""
    else:
        return str(obj)
    
def bw2tc(bandwidth, filterorder):
    '''
    calculate the lockin timeconstant from the bandwidth and the filter order
    scaling factor taken from LIA programming manual, directly corresponds to the 
    Low-Pass Filter order on LIA gui 
    Bandwidth is in Hz
    '''
    scale = np.array([1.0, 0.643594, 0.509825, 0.434979, 0.385614, 0.349946, 0.322629, 0.300845])[int(filterorder)-1]

    timeconstant = scale/(2*np.pi*bandwidth)
    return timeconstant 

def settling_time(bandwidth, filterorder, signal_level = 99):
    '''
    Calculate the time after the loop ends to reject data.
    Settling times taken from lockin manual
    Assumes we want to wait until the signal has reached 
    90%, 95%, or 99% of it's steady state value
    Bandwidth is in Hz
    '''
    tau = bw2tc(bandwidth, int(filterorder))
    
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
    
def marker_to_bool(markerdat):
    # Convert maker data from floating point to boolean so we're not saving a bunch of useless data
    return np.where(markerdat > 2, 1, 0)

def is_number(n):
    return isinstance(n, numbers.Number)

def stackem_aligned(data, start_indicies, length, phasealign_index = 0):
    '''
    break up a 1D array data into a 2D array
    Aray data is broken up into length long chunks, starting at the indicies in start_indicies
    The phase of each chunk is aligned to the phase of at index phasealign_index
    '''
    length = int(length)
    if start_indicies[-1] + length > len(data):
        k = len(start_indicies) - 1
    else:
        k = len(start_indicies)

    stack = np.zeros((length, k), dtype = 'complex_')
    for i in range(k):
        phasor_offset = np.exp(1j*np.angle(data[start_indicies[i] + phasealign_index]))
        stack[:,i] = data[start_indicies[i]:start_indicies[i]+length]/phasor_offset
    
    return stack