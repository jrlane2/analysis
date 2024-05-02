'''
Created by CG and JL on 20230509

Heavily modified by JL on 20240209
This is designed to fit a pair of simple ringdowns in tandem.
It only fits the data off ONE demod, the one near the frequency you drive at
So it fits one demod from 2 experiments.
'''

# Standard library imports
import os
import matplotlib as mpl
import matplotlib.colors as mcolors

mpl.rcParams["savefig.facecolor"] = 'white'
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import numbers
import numpy.random as npr
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from datetime import datetime

# Import custom code stored in the NewEP3Exp/Code directory
import sys
sys.path.append(sys.path[0].split('NewEP3Exp')[0] + 'NewEP3Exp/Code')
sys.path.append(sys.path[0].split('NewEP3Exp')[0] + 'NewEP3Exp/Data/routine logs/calibration coefficients')
import file_handling 
# import fitting.standard_fit_routines as fit
# import fitting.Rotating_frame_fit_routines as rotfit
import pickle
import time
import pandas as pd
import lmfit
import numpy.random as npr
import time
import math


'''
helper functions
'''

def xstr(obj):
    #output is string form of input
    if obj is None:
        return ""
    else:
        return str(obj)
    
def bw2tc(bandwidth, order):
    '''
    calculate the lockin timeconstant from the bandwidth and the filter order
    scaling factor taken from LIA programming manual, directly corresponds to the 
    Low-Pass Filter order on LIA gui 
    Bandwidth is in Hz
    '''
    scale = np.array([1.0, 0.643594, 0.509825, 0.434979, 0.385614, 0.349946, 0.322629, 0.300845])[order-1]

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
    tau = bw2tc(bandwidth, filterorder)
    
    if signal_level == 90:
        wait90 = [2.3, 3.89, 5.32, 6.68, 7.99, 9.27, 10.53, 11.77]
        return tau*wait90[filterorder-1]
    elif signal_level == 95:
        wait95 = [3, 4.7, 6.3, 7.8, 9.2, 11, 12, 13]
        return tau*wait95[filterorder-1]
    elif signal_level == 99:
        wait99 = [4.61, 6.64, 8.41, 10.05, 11.60, 13.11, 14.57, 16]
        return tau*wait99[filterorder-1]
    else:
        print("invalid signal level!")
        return 0

def is_number(n):
    # Check and see if a variable is a number
    return isinstance(n, numbers.Number)

def ift(array, t,):
    #given a timestamp, find the corresponding array index
    if type(t) == np.ndarray:
        return np.array([np.argmin(np.abs(array - i)) for i in t])
    elif is_number(t):
        return np.argmin(np.abs(array - t))
    else:
        raise ValueError("tstamp is not a number or array of numbers")

def unpack_FR(FR):
    L1 = np.abs(FR.params['L1'].value)
    L2 = np.abs(FR.params['L2'].value)
    K1 = np.abs(FR.params['K1'].value)
    K2 = np.abs(FR.params['K2'].value)
    phiL1 = FR.params['phiL1'].value % (2*np.pi)
    phiL2 = FR.params['phiL2'].value % (2*np.pi)
    phiK1 = FR.params['phiK1'].value % (2*np.pi)
    phiK2 = FR.params['phiK2'].value % (2*np.pi)
    phiKrel = FR.params['phiKrel'].value % (2*np.pi)

    bkgR1_1, bkgI1_1 = FR.params['bkgR1_1'].value, FR.params['bkgI1_1'].value
    bkgR1_2, bkgI1_2 = FR.params['bkgR1_2'].value, FR.params['bkgI1_2'].value
    bkgR2_1, bkgI2_1 = FR.params['bkgR2_1'].value, FR.params['bkgI2_1'].value
    bkgR2_2, bkgI2_2 = FR.params['bkgR2_2'].value, FR.params['bkgI2_2'].value

    fbar = FR.params['fbar'].value
    df = FR.params['df'].value
    gammabar = FR.params['gammabar'].value
    dgamma = FR.params['dgamma'].value

    return [L1, L2, K1, K2, phiL1, phiL2, phiK1, phiK2, phiKrel, bkgR1_1, bkgI1_1, 
                bkgR1_2, bkgI1_2, bkgR2_1, bkgI2_1, bkgR2_2, bkgI2_2, fbar, df, gammabar, dgamma]

def unpack_FR_return(FR):
    # unpacks the fit report for saving in an array when we loop over this
    header = ""
    values = np.array([])
    for i in FR.params:
        header = header + i + ', '
        if i == "L1" or i == "L2" or i == "K1" or i == "K2":
            values = np.append(values, np.abs(FR.params[i].value))
        elif i[:3] == "phi":
            values = np.append(values, FR.params[i].value%(2*np.pi))
        else:
            values = np.append(values, FR.params[i].value)
    return header, values

'''
file loaders, assemblers and savers

'''

class bare_loader(object):
    '''
    Class for loading the bare membrane parameters
    Loads up all the bare ringdown measurements and plots them out
    Aftwerwards, it you can figure out the interpolated bare mode frequencies
    By passing self.tDAq to the interpolation function (which you can get by reading off
    a timestamp on the file)
    '''
    def __init__(self):
        return
    
    def load_controlOFF_ringdowns(self, folderbare, filebare, folderanalysis, plot = True, save = True):
  
        #load bare ringdown fits
        bare_tab = np.loadtxt(folderbare + filebare, delimiter = ',')
        bm1l = bare_tab[:,[1,2,4]]
        bm2l = bare_tab[:,[1,3,5]]
        t0a = bm1l[0,0]
        tend = max(bm2l[:,0])
        
        '''
        We interpolate the bare mode frequencies to get bare mode frequency as a function of wall clock time.
        '''
        self.f1start = bm1l[0,1]
        self.f2start = bm2l[0,1]
        polyfitorder = 1*bool(len(bm1l) == 2) + 3*bool(len(bm1l) > 2)
        bf1fit = np.polyfit(bm1l[:,0] , bm1l[:,1] - self.f1start, deg = polyfitorder, cov=False)
        self.bf1poly = np.poly1d(bf1fit)
        bf2fit = np.polyfit(bm2l[:,0] , bm2l[:,1] - self.f2start, deg = polyfitorder, cov=False)
        self.bf2poly = np.poly1d(bf2fit)

        #take mean for bare gamma
        self.baregam1 = np.mean(np.abs(bm1l[:,2]))
        self.baregam2 = np.mean(np.abs(bm2l[:,2]))
        save_array = np.array([0, (tend-t0a), polyfitorder, self.f1start, bf1fit, self.baregam1, self.f2start, bf2fit, self.baregam2], dtype = object)

        #now make a plot
        tdense = np.linspace(0, (tend-t0a), 100)

        #Edit the font, font size, and axes width
        mpl.rcParams['font.family'] = 'Arial'
        mpl.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.linewidth'] = 2

        #Create figure object and store it in a variable called 'fig'
        fig = plt.figure(figsize=(7, 4),facecolor='white')

        #Add axes object to our figure that takes up entire figure
        ax = fig.add_axes([0, 0, 1, 1])

        #Edit the major and minor ticks of the x and y axes
        ax.xaxis.set_tick_params(which='major', size=5, width=2, direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', size=3, width=2, direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', size=5, width=2, direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', size=3, width=2, direction='in', right='on')

        #plot
        ax.plot(bm1l[:,0] , bm1l[:,1] - bm1l[0,1], 'o', color='forestgreen', label='mode 1', markersize = 10)
        ax.plot(tdense, self.bf1poly(tdense),  color='forestgreen')
        ax.plot(bm2l[:,0] , bm2l[:,1] - bm2l[0,1], 'o', color='darkorange', label='mode 2', markersize = 10)
        ax.plot(tdense, self.bf2poly(tdense),  color='darkorange')
        # Add the x and y-axis labels
        ax.set_xlabel('DAq time (hr)', labelpad=10)
        ax.set_ylabel('deviation in bare mode frequency (Hz)', labelpad=10)
        ax.legend(fontsize = 16, loc = 'best', frameon = True, fancybox = True)
        ax.grid(True)

        #save or not
        if save == True:
            #save
            plotname, pathname1 = file_handling.make_plotfilename('bare mode frequencies',folderanalysis)
            plt.savefig(plotname,dpi = 300, bbox_inches = 'tight');
            name, path = file_handling.make_filename('bare mode frequencies', folderanalysis,'txt')
            tppreamble = "tstart (hr), tend (hr), order, f1start (Hz), f1fitcoeff, baregam1 (Hz), f2start (Hz), f2fitcoeff, baregam2 (Hz)"
            np.savetxt(name, save_array, header = tppreamble, fmt='%s')

        #plot or not
        if plot == False:
            plt.close()

        self.t0 = datetime.strptime(str((pd.read_csv(folderbare + "C_avg//" + "mode1_bare_0.csv", nrows = 0).columns)[0]).replace('# ',''), '%Y-%m-%d %H:%M:%S.%f')

        #return start time, start bare f1, bare f1 interpolation, bare gam1, start bare f2, bare f2 interpolation, bare gam2
        return 


    def load_for_timestamps(self, filename):
        '''
        loads time stamp of each data set
        t0: start of the sheet
        returns the time since the data has started
        '''
        self.tDAq = ((datetime.strptime(str((pd.read_csv(filename, nrows = 0).columns)[0]).replace('# ',''),'%Y-%m-%d %H:%M:%S.%f') - self.t0).total_seconds())/3600

        return self.tDAq
    
    def load_tDAq(self, fname1, fname2):
        '''
        when you want the average of two file times, for some reason...
        '''
        t1 = ((datetime.strptime(str((pd.read_csv(fname1, nrows = 0).columns)[0]).replace('# ',''),'%Y-%m-%d %H:%M:%S.%f') - self.t0).total_seconds())/3600
        t2 = ((datetime.strptime(str((pd.read_csv(fname2, nrows = 0).columns)[0]).replace('# ',''),'%Y-%m-%d %H:%M:%S.%f') - self.t0).total_seconds())/3600

        self.tDAq = (t1 + t2)/2
        return self.tDAq
        
    def get_BareMembraneParams(self):
        if hasattr(self, 'tDAq') and hasattr(self, 'f1start'):

            self.barefm1 = self.f1start + self.bf1poly(self.tDAq)
            self.barefm2 = self.f2start + self.bf2poly(self.tDAq)

            self.BareMembraneParams = [2*np.pi*self.barefm1, 2*np.pi*self.barefm2, 2*np.pi*self.baregam1, 2*np.pi*self.baregam2]
            return self.BareMembraneParams 
        else:
            print("Need to load time stamp and bare mode frequencies first!")
            return

class guess_loader(object):
    '''
    We've somehow backed ourselves into a corner over loading guesses in different ways for different measurements
    We should fix this, but that has to happen in the actual DAq code (i.e. in how we save these parameters)
    for now, I'll just define an object to load the guesses for Berry phase experiments
    '''
    def __init__(self, filename):
        self.guesses  = np.loadtxt(filename)[:,0:7]
        self.f1g, self.f2g, self.gam1g, self.gam2g, self.eta, self.bf1, self.bf2 = self.guesses[:,0], self.guesses[:,1], self.guesses[:,2], self.guesses[:,3], self.guesses[:,4], self.guesses[:,5], self.guesses[:,6]
        # f_i is the lab frame eigenvalue, except it's boosted up bu (bf1 + bf2)/2 for some reason
        # Everything is in Hz... except gamma_i, which is in rad/s
        # this function loadss the guesses as they are in the file into an object
        self.times = np.loadtxt(filename)[:,9]
        return 
    def ith_guess(self, i):
        # This function returns a guess for a single run, more amenable to the fitting routine
        bfbar = (self.bf1[i] + self.bf2[i])/2
        self.init_guess = [self.f1g[i] - bfbar, self.f2g[i] - bfbar, self.gam1g[i]/(2*np.pi), self.gam2g[i]/(2*np.pi), self.bf1[i], self.bf2[i], self.eta[i]]
        return self.init_guess

    def plot_bare(self, plot = False):
        #Edit the font, font size, and axes width
        mpl.rcParams['font.family'] = 'Arial'
        mpl.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.linewidth'] = 2

        #Create figure object and store it in a variable called 'fig'
        fig, [ax1, ax2] = plt.subplots(nrows = 1, ncols =2, figsize=(12, 4),facecolor='white')

        #Edit the major and minor ticks of the x and y axes
        ax1.xaxis.set_tick_params(which='major', size=5, width=2, direction='in', top='on')
        ax1.xaxis.set_tick_params(which='minor', size=3, width=2, direction='in', top='on')
        ax1.yaxis.set_tick_params(which='major', size=5, width=2, direction='in', right='on')
        ax1.yaxis.set_tick_params(which='minor', size=3, width=2, direction='in', right='on')

        ax1.xaxis.set_tick_params(which='major', size=5, width=2, direction='in', top='on')
        ax1.xaxis.set_tick_params(which='minor', size=3, width=2, direction='in', top='on')
        ax1.yaxis.set_tick_params(which='major', size=5, width=2, direction='in', right='on')
        ax1.yaxis.set_tick_params(which='minor', size=3, width=2, direction='in', right='on')

        #plot
        ax1.plot(self.times , self.bf1 - self.bf1[0], 'o', color='forestgreen', label='mode 1', markersize = 10)
        ax1.plot(self.times , self.bf2 - self.bf2[0], 'o', color='darkorange', label='mode 2', markersize = 10)
        ax2.plot(self.times , self.gam1g - self.gam1g[0], 'o', color='forestgreen', label='mode 1', markersize = 10)
        ax2.plot(self.times , self.gam2g - self.gam2g[0], 'o', color='darkorange', label='mode 2', markersize = 10)
        # Add the x and y-axis labels
        ax1.set_xlabel('DAq time (hr)', labelpad=10)
        ax1.set_ylabel('deviation in bare mode frequency (Hz)', labelpad=10)
        ax1.legend(fontsize = 16, loc = 'best', frameon = True, fancybox = True)
        ax1.grid(True)

        ax2.set_xlabel('DAq time (hr)', labelpad=10)
        ax2.set_ylabel('deviation in damping (Hz)', labelpad=10)
        ax2.legend(fontsize = 16, loc = 'best', frameon = True, fancybox = True)
        ax2.grid(True)

        #plot or not
        if plot == False:
            plt.close()


class load_data(object):
    '''
    Child class for loading ringdown data from one file
    Fot the data considered in this fitting routine, we're not doing loops
    So there's no need to differentiate between loops and ringdowns
    '''
    def __init__(self, filename):
        self.filename = filename
        header0 = pd.read_csv(filename, nrows = 12)
        header1 = (header0.to_numpy())[:,0]
        header1_elements = [i for i in range(len(header1))]
        for e in header1_elements:
            header1[e] = float((header1[e].split())[-1])
            
        #distribute data specs
        [self.demod2f, self.demod5f, setdrivetime, filterorder, BWLIA, self.tLIA, t_wait] = header1[[0,1,2,3,5,6,11]]
        t_wait = 0 # bandage, since there's no loops and I just want T=0 to be when the drive turns off
        self.tauLIA = bw2tc(BWLIA, int(filterorder))
        self.setlooptime = 0
        
        #load data
        dat0 = pd.read_csv(filename, skiprows = 14, dtype = float)
        endv = 4
        dat1 = (dat0.to_numpy())[0:-endv]
        
        #make columns
        #time and TTL
        self.time = dat1[:,0]
        TTL = dat1[:,1]

        #demod 2 simple ringdown: abs, arg, C, X, Y
        abs2 = dat1[:,2]
        arg2 = 2*np.pi*dat1[:,3]
        self.C2 = abs2*np.exp(+1j*arg2)
        
        #demod 5 simple ringdown: abs, arg, C, X, Y
        abs5 = dat1[:,4]
        arg5 = 2*np.pi*dat1[:,5]
        self.C5 = abs5*np.exp(+1j*arg5)

        '''
        find switch position and other markers
        '''
        #array index
        self.iloopstart = np.where(np.diff(TTL).astype(int) == -1)[0][0] 

        #timestamps
        self.tloopstart = self.time[self.iloopstart] + t_wait
        self.tloopend = self.tloopstart + self.setlooptime
        self.tLIAstart = self.tloopstart
        self.tLIAend = self.tLIAstart + self.tLIA
        self.tDAqend = self.tloopend + setdrivetime

        #array indices
        self.iloopend = ift(self.time, self.tloopend)
        self.iLIAstart = ift(self.time, self.tLIAstart)
        self.iLIAend = ift(self.time, self.tLIAend)
        self.iDAqend = ift(self.time, self.tDAqend)

        return
    



def make_id(idtab):
    '''
    unrandomize ringdown measurements for fitting
    '''
    idtab = np.int32(idtab)
    uniques, out = np.unique(idtab, return_inverse=True)
    inds = idtab.argsort()
    sortedtab = uniques[inds]
    
    return sortedtab


def load_tap(filetap):
    '''
    to load taps
    '''

    #import relevant tap csv file
    tapdata = np.loadtxt(filetap)
    #distribute values
    eta = tapdata[0,6]

    measP1 = tapdata[:,7]
    measP2 = tapdata[:,8]
    setdelta = tapdata[:,3]
    measP4 = tapdata[:,11]
    setDelta4 = tapdata[:,5]

    return [eta, measP1, measP2, setdelta, measP4, setDelta4]

def save_fit_report(filename, foldername, FRs, save = True):
    #save fit report in a text file
    save_filename, _ = file_handling.make_filename(filename, foldername, 'txt')
    if save == True:
        with open(save_filename, 'w') as fh:
            fh.write('simple ringdown fit \n')
            fh.write(FRs[0].fit_report())

            fh.close()



'''
fit functions
'''
def single_osc_demod_response(t, A, phi, w1, gam1, demodf, tau):
    '''
    A in mV, phi in rad modulo 2pi, t in seconds, w1 in 2pi*Hz, gam1 in 2pi*Hz, demodf in Hz, tau in seconds
    '''
    filterfunc = 1/(1+1j*(w1 - (2*np.pi*demodf))*tau)
    oscfunc = np.abs(A)*np.exp(+1j*(phi%(2*np.pi)))*np.exp(-(np.abs(gam1)*t)/2)*np.exp(+1j*(w1 - (2*np.pi*demodf))*t)
    fullfunc = filterfunc*oscfunc
    
    return fullfunc

def single_osc_demod_response2(t, Are, Aim, w1, gam1, demodf, tau):
    '''
    Are, Aim in mV, t in seconds, w1 in 2pi*Hz, gam1 in 2pi*Hz, demodf in Hz, tau in seconds
    '''
    filterfunc = 1/(1+1j*(w1 - (2*np.pi*demodf))*tau)
    oscfunc = (Are + 1j*Aim)*np.exp(-(np.abs(gam1)*t)/2)*np.exp(+1j*(w1 - (2*np.pi*demodf))*t)
    fullfunc = filterfunc*oscfunc
    
    return fullfunc

def single_osc_demod_response_fit(t, A, phi, w1, gam1, dw1, demodf, tau, bkgR, bkgI):
   
    fullfunc = single_osc_demod_response(t, A, phi, 2*np.pi*w1 + 2*np.pi*dw1, 2*np.pi*gam1, demodf, tau)

    model = fullfunc + bkgR + 1j*bkgI 
    
    return model

single_osc_demod_response_fit_model = lmfit.Model(single_osc_demod_response_fit, independent_vars=['t'])

def two_osc_demod_response(t, A1, A2, phi1, phi2, w1, w2, gam1, gam2, dw1, dw2, demodf, tau, bkgR, bkgI, scaleR, scaleI):
    '''
    Units same as: single_osc_demod_response()
    '''
    osc1 = single_osc_demod_response(t, A1, phi1, 2*np.pi*w1 + 2*np.pi*dw1, 2*np.pi*gam1, demodf, tau)
    osc2 = single_osc_demod_response(t, A2, phi2, 2*np.pi*w2 + 2*np.pi*dw2, 2*np.pi*gam2, demodf, tau)
    initmodel = osc1 + osc2 + bkgR + 1j*bkgI 
    model = (np.real(initmodel)/scaleR) + 1j*(np.imag(initmodel)/scaleI)
    
    return model

two_osc_demod_response_model = lmfit.Model(two_osc_demod_response, independent_vars=['t'])

def two_prep_one_demod_response(t_1, t_2, A1_1, A2_1, A1_2, A2_2, phi1_1, phi2_1, phi1_2, phi2_2, w1_1, w2_1, w1_2, w2_2, gam1, gam2, 
                               dw1, dw2, demodf1, demodf2, tau, bkgR_1, bkgI_1, bkgR_2, bkgI_2, scaleR_1, scaleI_1, scaleR_2, scaleI_2):
    '''
    Units same as: single_osc_demod_response()
    This is for *two* separate preparations, where I'm fitting a single demod per preparation
    the preperation index is given after the underscore, i.e. varibles related to experiment 1 are given by _1 and vice verse
    Shared between the two preparations are the shift frequencies, decay rates, demod frequencies and time constant
    '''
    # prep 1
    osc1demod_1 = single_osc_demod_response(t_1, A1_1, phi1_1, 2*np.pi*w1_1 + 2*np.pi*dw1, 2*np.pi*gam1, demodf1, tau)
    osc2demod_1 = single_osc_demod_response(t_1, A2_1, phi2_1, 2*np.pi*w2_1 + 2*np.pi*dw2, 2*np.pi*gam2, demodf1, tau)

    # prep 2
    osc1demod_2 = single_osc_demod_response(t_2, A1_2, phi1_2, 2*np.pi*w1_2 + 2*np.pi*dw1, 2*np.pi*gam1, demodf2, tau)
    osc2demod_2 = single_osc_demod_response(t_2, A2_2, phi2_2, 2*np.pi*w2_2 + 2*np.pi*dw2, 2*np.pi*gam2, demodf2, tau)


    initmodel_1 = osc1demod_1 + osc2demod_1 + bkgR_1 + 1j*bkgI_1
    initmodel_2 = osc1demod_2 + osc2demod_2 + bkgR_2 + 1j*bkgI_2

    model1 = (np.real(initmodel_1)/scaleR_1) + 1j*(np.imag(initmodel_1)/scaleI_1)
    model2 = (np.real(initmodel_2)/scaleR_2) + 1j*(np.imag(initmodel_2)/scaleI_2)
    
    return np.concatenate([model1, model2])

two_prep_one_demod_response_model = lmfit.Model(two_prep_one_demod_response, independent_vars=['t_1', 't_2'])

def mode_one_prep_two_demod_response(t, A, B, C, phiA, phiB, phiC, w1, w2, fbar, df, gammabar, dgamma, 
                                     demodf2, demodf5, tau, bkgR2, bkgI2, bkgR5, bkgI5, scaleR2, scaleI2, scaleR5, scaleI5):
    '''
    This is for when we prepare the (1,0) mode (which, for the purposes of this, is the mode near demod 2) and fit both demods.
    w1 and w2: frequency to boost rotating frame eigenvalue to lab frame eigenvalue (still in Hz), which I'll assume is shared for a single demod
    '''

    # Oscillators at demod 2
    osc1demod2 = single_osc_demod_response(t, A, phiA, 2*np.pi*(w1 + fbar + df/2), 2*np.pi*(gammabar + dgamma/2), demodf2, tau)
    osc2demod2 = single_osc_demod_response(t, B, phiB, 2*np.pi*(w1 + fbar + df/2), 2*np.pi*(gammabar - dgamma/2), demodf2, tau)

    # Osicllators at demod 5

    osc1demod5 = single_osc_demod_response(t, C, phiC, 2*np.pi*(w2 + fbar - df/2), 2*np.pi*(gammabar + dgamma/2), demodf5, tau)
    osc2demod5 = -1*single_osc_demod_response(t, C, phiC, 2*np.pi*(w2 + fbar - df/2), 2*np.pi*(gammabar - dgamma/2), demodf5, tau)

    # Combine the two demods
    initmodel_2 = osc1demod2 + osc2demod2 + bkgR2 + 1j*bkgI2
    initmodel_5 = osc1demod5 + osc2demod5 + bkgR5 + 1j*bkgI5

    model2 = (np.real(initmodel_2)/scaleR2) + 1j*(np.imag(initmodel_2)/scaleI2)
    model5 = (np.real(initmodel_5)/scaleR5) + 1j*(np.imag(initmodel_5)/scaleI5)

    return np.concatenate([model2, model5])

mode_one_prep_two_demod_response_model = lmfit.Model(mode_one_prep_two_demod_response, independent_vars=['t'])

def two_prep_two_demod_response(t_1, t_2, L1, L2, K1, K2, phiL1, phiL2, phiK1, phiK2, phiKrel, w1_1, w2_1, w1_2, w2_2, gammabar, dgamma, 
                               fbar, df, demodf1_1, demodf2_1, demodf1_2, demodf2_2, tau, bkgR1_1, bkgI1_1, bkgR2_1, bkgI2_1, 
                               bkgR1_2, bkgI1_2, bkgR2_2, bkgI2_2): # , scaleR_1, scaleI_1, scaleR_2, scaleI_2): remove scales for now
    '''
    Units same as: single_osc_demod_response()
    This is for *two* separate preparations, where I'm fitting both demods in each experiment
    the preperation index is given after the underscore, i.e. varibles related to experiment 1 are given by _1 and vice verse
    Shared between the two preparations are the shift frequencies, decay rates, demod frequencies and time constant.
    This assumes we're initializing in (1,0) and (0,1), respectively
    '''

    # Prepare amplitudes
    L1c = np.abs(L1)*np.exp(1j*phiL1)
    L2c = np.abs(L2)*np.exp(1j*phiL2)
    K1c = np.abs(K1)*np.exp(1j*phiK1)
    K2c = np.abs(K2)*np.exp(1j*phiK2)
    A1 = -L1c*K2c/K1c
    B1 = L1c
    C1 = -L1c*K2c
    C1m = -C1
    A2 = L2c/K1c*np.exp(1j*phiKrel) # account for misalignment of the demods between the two measurmement
    A2m = -A2
    B2 = L2c
    C2 = -L2c*K2c/K1c

    # prep 1, individual ringing decays
    osc1demod1_1 = single_osc_demod_response2(t_1, A1.real, A1.imag, 2*np.pi*w1_1 + 2*np.pi*(fbar + df/2), 2*np.pi*(gammabar + dgamma/2), demodf1_1, tau)
    osc2demod1_1 = single_osc_demod_response2(t_1, B1.real, B1.imag, 2*np.pi*w1_1 + 2*np.pi*(fbar - df/2), 2*np.pi*(gammabar - dgamma/2), demodf1_1, tau)

    osc1demod2_1 = single_osc_demod_response2(t_1, C1.real, C1.imag, 2*np.pi*w2_1 + 2*np.pi*(fbar + df/2), 2*np.pi*(gammabar + dgamma/2), demodf2_1, tau)
    osc2demod2_1 = single_osc_demod_response2(t_1, C1m.real, C1m.imag, 2*np.pi*w2_1 + 2*np.pi*(fbar - df/2), 2*np.pi*(gammabar - dgamma/2), demodf2_1, tau)

    # prep 2, individual ringing decays
    osc1demod1_2 = single_osc_demod_response2(t_2, A2.real, A2.imag, 2*np.pi*w1_2 + 2*np.pi*(fbar + df/2), 2*np.pi*(gammabar + dgamma/2), demodf2_2, tau)
    osc2demod1_2 = single_osc_demod_response2(t_2, A2m.real, A2m.imag, 2*np.pi*w1_2 + 2*np.pi*(fbar - df/2), 2*np.pi*(gammabar - dgamma/2), demodf2_2, tau)

    osc1demod2_2 = single_osc_demod_response2(t_2, B2.real, B2.imag, 2*np.pi*w2_2 + 2*np.pi*(fbar + df/2), 2*np.pi*(gammabar + dgamma/2), demodf1_2, tau)
    osc2demod2_2 = single_osc_demod_response2(t_2, C2.real, C2.imag, 2*np.pi*w2_2 + 2*np.pi*(fbar - df/2), 2*np.pi*(gammabar - dgamma/2), demodf1_2, tau)
    

    # prep 1, full response from each demod
    model1_1 = osc1demod1_1 + osc2demod1_1 + bkgR1_1 + 1j*bkgI1_1
    model1_2 = osc1demod2_1 + osc2demod2_1 + bkgR2_1 + 1j*bkgI2_1

    # prep 2, full response from each demod
    model2_1 = osc1demod1_2 + osc2demod1_2 + bkgR1_2 + 1j*bkgI1_2
    model2_2 = osc1demod2_2 + osc2demod2_2 + bkgR2_2 + 1j*bkgI2_2

    # model1 = (np.real(initmodel_1)/scaleR_1) + 1j*(np.imag(initmodel_1)/scaleI_1)
    # model2 = (np.real(initmodel_2)/scaleR_2) + 1j*(np.imag(initmodel_2)/scaleI_2)
    
    return np.concatenate([model1_1, model1_2, model2_2, model2_1])

two_prep_two_demod_response_model = lmfit.Model(two_prep_two_demod_response, independent_vars=['t_1', 't_2'])


'''
fit all class
'''
class fit_one():
    def __init__(self, file,  initguess, shift = [0,0], scaleop = 0):
        '''
        This class is an intermediate step, designed for fitting both demods in an experiment 
        where we drive on mode 1, preparing (A,0)

        file contains the data that we take after driving near mode 1

        These data get loaded into an object, which is a child of this class, and called exp1 

        shift is a "cluge" to escape non-linear ringdowns. Right now the second one variable doesn't do anything
        In the future, we may iplement different shifts for demod 2 and demod 5

        #scaleop is for putting both mode data at same footing.

        '''
        
        #load dynamics data: file1a, mode 1
        exp1 = load_data(file)


        exp1.tLIA = exp1.tLIA + shift[0]*1e-3 # Shift start time to escape non-linear ringdown 
        # Propagate shifts downwards
        exp1.tLIAend = exp1.tLIAstart + exp1.tLIA
        exp1.iLIAend = ift(exp1.time, exp1.tLIAend)


        #load guesses that are already compensated for drifts, in Hz units
        self.f1g, self.f2g, self.gam1g, self.gam2g, self.bf1, self.bf2, self.eta = initguess

        if exp1.demod2f < exp1.demod5f:
            self.w11 = (self.bf1 + self.eta/2) # mode 1 boost from rotating to lab
            self.w22 = (self.bf2 - self.eta/2) # mode 2 boost from rotating to lab
        else:
            self.w11 = (self.bf2 - self.eta/2) # mode 2 boost from rotating to lab
            self.w22 = (self.bf1 + self.eta/2) # mode 1 boost from rotating to lab

        
        self.scale1s = bool(scaleop == 0)*1 + bool(scaleop == 1)*1*max(np.abs(exp1.C2[exp1.iLIAend:-1]))
        self.scale2s = self.scale1s
        self.scale3s = bool(scaleop == 0)*1 + bool(scaleop == 1)*1*max(np.abs(exp1.C5[exp1.iLIAend:-1]))
        self.scale4s = self.scale3s
        
        # I dont want to carry around all these selfs
        self.exp1 = exp1

        return
    
    def prepare_model(self, guess_array, modeorder, maxdev = 200):
        '''
        Unpacks the guess array and prepares the model for fitting

        modeorder: booleanif for if the experiment order matches demods sorted by frequence
        currently: True if exp1.demod2f < exp2.demod2f, False if exp1.demod2f > exp2.demod2f
        Experiment 1 is at the lower frequency mode, true, else false
        If you pass eigenvalue guesses, it incorporates them
        Otherwise it takes the guessed values from self.f1g, ...
        '''

        # I dont want to carry around all these selfs
        exp1 = self.exp1

        Ag, Bg, Cg, phAg, phBg, phCg, *backgrounds_eigenvalues = guess_array

        model = mode_one_prep_two_demod_response_model

        #set fixed parameters in the model
        model.set_param_hint('w1', value = self.w11, vary = False)
        model.set_param_hint('w2', value = self.w22, vary = False)
        model.set_param_hint('tau', value = exp1.tauLIA, vary = False)

        if modeorder:
            # set model parameters to the correct demod frequency and all the scalings correctly
            model.set_param_hint('scaleR2', value = self.scale1s, vary = False)
            model.set_param_hint('scaleI2', value = self.scale2s, vary = False)
            model.set_param_hint('scaleR5', value = self.scale3s, vary = False)
            model.set_param_hint('scaleI5', value = self.scale4s, vary = False)

            model.set_param_hint('demodf2', value = exp1.demod2f, vary = False)
            model.set_param_hint('demodf5', value = exp1.demod5f, vary = False)            

        # else:
        #     model.set_param_hint('scaleR2', value = self.scale3s, vary = False)
        #     model.set_param_hint('scaleI2', value = self.scale4s, vary = False)
        #     model.set_param_hint('scaleR5', value = self.scale1s, vary = False)
        #     model.set_param_hint('scaleI5', value = self.scale2s, vary = False)

        #     model.set_param_hint('demodf2', value = exp1.demod2f, vary = False)
        #     model.set_param_hint('demodf5', value = exp1.demod5f, vary = False)

        # Provide intial guesses to variable parameters
        model.set_param_hint('A', value = Ag)
        model.set_param_hint('B', value = Bg)
        model.set_param_hint('C', value = Cg)

        model.set_param_hint('phiA', value = phAg)
        model.set_param_hint('phiB', value = phBg) 
        model.set_param_hint('phiC', value = phCg)


        if len(backgrounds_eigenvalues) == 0:
            # If we're fitting from scratch
            model.set_param_hint('bkgR2', value = 0)
            model.set_param_hint('bkgI2', value = 0)
            model.set_param_hint('bkgR5', value = 0)
            model.set_param_hint('bkgI5', value = 0)
            fbarg = (self.f1g + self.f2g)/2
            dfg = (self.f2g - self.f1g)
            gammabarg = (self.gam1g + self.gam2g)/2
            dgammag = (self.gam2g - self.gam1g)
            model.set_param_hint('fbar', value = fbarg, min = -maxdev + fbarg, max = maxdev + fbarg, vary = True)#200 Hz
            model.set_param_hint('df', value = dfg, min = -maxdev + dfg, max = maxdev + dfg, vary = True)
            model.set_param_hint('gammabar', value = gammabarg, min = -maxdev + gammabarg, max = maxdev + gammabarg, vary = True)
            model.set_param_hint('dgamma', value = dgammag, min = -maxdev + dgammag, max = maxdev + dgammag, vary = True)       

        elif len(backgrounds_eigenvalues) == 8:
            # If we've already got a guess for the background and eigenvalues from a previous fit
            bkgR2g, bkgI2g, bkgR5g, bkgI5g, fbarg, dfg, gammabarg, dgammag = backgrounds_eigenvalues
            model.set_param_hint('bkgR2', value = bkgR2g)
            model.set_param_hint('bkgI2', value = bkgI2g)
            model.set_param_hint('bkgR5', value = bkgR5g)
            model.set_param_hint('bkgI5', value = bkgI5g)
            model.set_param_hint('fbar', value = fbarg, min = -maxdev + fbarg, max = maxdev + fbarg, vary = True)#200 Hz
            model.set_param_hint('df', value = dfg, min = -maxdev + dfg, max = maxdev + dfg, vary = True)
            model.set_param_hint('gammabar', value = gammabarg, min = -maxdev + gammabarg, max = maxdev + gammabarg, vary = True)
            model.set_param_hint('dgamma', value = dgammag, min = -maxdev + dgammag, max = maxdev + dgammag, vary = True)

        else:
            print(backgrounds_eigenvalues)
            raise ValueError("Guess array is an invalid length")
        
        return model

    def dofits(self, guesss, maxdev = 200):
        '''
        Having loaded all the data into the object: actually fit the data
        '''
        # Again, being lazy with the selfs
        exp1 = self.exp1

        # Prepare lmfit model
        # if exp1.demod2f < exp2.demod2f:
        model= self.prepare_model(guesss, True, maxdev)

        # Get the data into the right format
        fitdat1 = (np.real(exp1.C2[exp1.iLIAend:-1])/self.scale1s) + 1j*(np.imag(exp1.C2[exp1.iLIAend:-1])/self.scale2s) # data from exp 1, lower frequency (3,3)
        fitdat2 = (np.real(exp1.C5[exp1.iLIAend:-1])/self.scale3s) + 1j*(np.imag(exp1.C5[exp1.iLIAend:-1])/self.scale4s) # data from exp 2, higher frequency (5,3)
        fitdat = np.concatenate([fitdat1, fitdat2])
        t1axis = exp1.time[exp1.iLIAend:-1]-exp1.time[exp1.iLIAstart]

        # else:
        #     model= self.prepare_model(guesss, False, maxdev)

        #     # Get the data into the right format
        #     fitdat2 = (np.real(exp1.C2[exp1.iLIAend:-1])/self.scale1s) + 1j*(np.imag(exp1.C2[exp1.iLIAend:-1])/self.scale2s) # data from exp 1, higher frequency (5,3)
        #     fitdat1 = (np.real(exp2.C2[exp2.iLIAend:-1])/self.scale3s) + 1j*(np.imag(exp2.C2[exp2.iLIAend:-1])/self.scale4s) # data from exp 2, lower frequency (3,3)
        #     fitdat = np.concatenate([fitdat1, fitdat2])
        #     t2axis = exp1.time[exp1.iLIAend:-1]-exp1.time[exp1.iLIAstart]
        #     t1axis = exp2.time[exp2.iLIAend:-1]-exp2.time[exp2.iLIAstart]

        # Fit the data formated above using the perapred model
        params = model.make_params()
        FR = model.fit(fitdat, params, t = t1axis, max_nfev=1000000)
        
        return FR 
    
    def run_fit_generate_data(self, guesss):
        '''
        Essentially a wrapper around dofits
        Also generates:
        - Theory curves and experimental data for plotting
        - The fit report, as self.fit_report
        - An array with relevant parameters and the best-fit values, as self.return_array
        - A guess array to feed forward to the next fit, as self.next_guess
        '''

        # Again, being lazy with the selfs
        exp1 = self.exp1

        #do initial fitting
        FRsi = self.dofits(guesss)

        #unpack simple ringdown
        Ai = FRsi.params['A'].value
        Bi = FRsi.params['B'].value
        Ci = FRsi.params['C'].value
        phiAi = FRsi.params['phiA'].value
        phiBi = FRsi.params['phiB'].value
        phiCi = FRsi.params['phiC'].value
        bkgR2i = FRsi.params['bkgR2'].value
        bkgI2i = FRsi.params['bkgI2'].value
        bkgR5i = FRsi.params['bkgR5'].value
        bkgI5i = FRsi.params['bkgI5'].value
        fbari = FRsi.params['fbar'].value
        dfi = FRsi.params['df'].value
        gammabari = FRsi.params['gammabar'].value
        dgammai = FRsi.params['dgamma'].value

        guesssi = [np.abs(Ai), np.abs(Bi), np.abs(Ci), phiAi%(2*np.pi), phiBi%(2*np.pi), phiCi%(2*np.pi), 
                   bkgR2i, bkgI2i, bkgR5i, bkgI5i, fbari, dfi, gammabari, dgammai]
        

        #do fitting again to reduce fitting failures

        FRs = self.dofits(guesssi)
        As = FRs.params['A'].value
        Bs = FRs.params['B'].value
        Cs = FRs.params['C'].value
        phiAs = FRs.params['phiA'].value
        phiBs = FRs.params['phiB'].value
        phiCs = FRs.params['phiC'].value
        bkgR2s = FRs.params['bkgR2'].value
        bkgI2s = FRs.params['bkgI2'].value
        bkgR5s = FRs.params['bkgR5'].value
        bkgI5s = FRs.params['bkgI5'].value
        fbars = FRs.params['fbar'].value
        dfs = FRs.params['df'].value
        gammabars = FRs.params['gammabar'].value
        dgammas = FRs.params['dgamma'].value

        #make theory curves
        self.fit_t = np.linspace(exp1.time[exp1.iLIAend], exp1.time[-1], 500)
        self.fit_C2 = two_osc_demod_response(self.fit_t - exp1.time[exp1.iLIAstart], As, Bs, phiAs, phiBs, self.w11, self.w11, gammabars-dgammas/2, gammabars+dgammas/2,
                                            fbars-dfs/2, fbars+dfs/2, exp1.demod2f, exp1.tauLIA, bkgR2s, bkgI2s, 1, 1)
        self.fit_C5 = two_osc_demod_response(self.fit_t - exp1.time[exp1.iLIAstart], Cs, Cs, phiCs, np.pi + phiCs, self.w22, self.w22, gammabars-dgammas/2, gammabars+dgammas/2,
                                            fbars-dfs/2, fbars+dfs/2, exp1.demod5f, exp1.tauLIA, bkgR5s, bkgI5s, 1, 1)
        
        self.plotter = Plotter(self.exp1, None, [self.fit_t, self.fit_C2, self.fit_C5])

        self.fit_report = FRs

        self.return_array = np.array([exp1.demod2f, exp1.demod5f, fbars + dfs/2, fbars - dfs/2, gammabars + dgammas/2, gammabars - dgammas/2, 
                                      self.w11, self.w22, self.bf1, self.bf2, self.eta, np.abs(As), np.abs(Bs), np.abs(Cs), phiAs%(2*np.pi), phiBs%(2*np.pi), phiCs%(2*np.pi),
                                      bkgR2s, bkgI2s, bkgR5s, bkgI5s, self.scale1s, self.scale2s, self.scale3s, self.scale4s]) # for saving the fit results
        
        self.return_array_labels = ['demod2f (Hz)', 'demod5f (Hz)', 'lab frame f1 (Hz)', 'lab frame f2 (Hz)', 'gamma1 (Hz)', 'gamma2 (Hz)', 'bf1 + eta/2', 'bf2 - eta/2',
                                    'baref1 (Hz)', 'baref2 (Hz)', 'eta (Hz)', 'A1 (mV)', 'B (mV)', 'C (mV)', 'phiA (rad)', 'phiB (rad)', 'phiC (rad)', 
                                    'bkgR2 (mV)', 'bkgI2 (mV)', 'bkgR5 (mV)', 'bkgI5 (mV)', 'scale1 (mV)', 'scale2 (mV)', 'scale3 (mV)', 'scale4 (mV)']

        self.next_guess = [np.abs(As), np.abs(Bs), np.abs(Cs), phiAs%(2*np.pi), phiBs%(2*np.pi), phiCs%(2*np.pi), bkgR2s, bkgI2s, bkgR5s, bkgI5s, fbars, dfs, gammabars, dgammas]

        

        return



class fit_all():
    def __init__(self, file1a, file1b, initguess, shift = [0,0], scaleop = 0):
        '''
        The idea here is that we drive at two different configurations: (a) on mode 1, preparing (A,0)
        and (b) on mode 2, preparing (0,B). This gets us two different ringdowns.

        file1a contains the data that we take after driving near mode 1
        file1b contains the data that we take after driving near mode 2

        These data get loaded into two objects, which are children of this class, and called exp1 and exp2 respectively

        shift is a "cluge" to escape non-linear ringdowns. Given separately for demod 2 and demod 5, in units ms

        #scaleop is for putting both mode data at same footing. I'm going to comment it out for now, we can add id back
        in if it turns out to be necessary 

        '''
        
        #load dynamics, and make sure exp1 is when you're driving at the lower frequency mode
        exp1 = load_data(file1a)
        if exp1.demod2f < exp1.demod5f:
            exp2 = load_data(file1b)
        else:
            exp2 = load_data(file1a)
            exp1 = load_data(file1b)


        exp1.tLIA = exp1.tLIA + shift[0]*1e-3 # Shift start time to escape non-linear ringdown 
        # Propagate shifts downwards
        exp1.tLIAend = exp1.tLIAstart + exp1.tLIA
        exp1.iLIAend = ift(exp1.time, exp1.tLIAend)


        exp2.tLIA = exp2.tLIA + shift[0]*1e-3 # Shift start time to escape non-linear ringdown 
        # Propagate shifts downwards
        exp2.tLIAend = exp2.tLIAstart + exp2.tLIA
        exp2.iLIAend = ift(exp2.time, exp2.tLIAend)


        #load guesses that are already compensated for drifts, in Hz units
        self.f1g, self.f2g, self.gam1g, self.gam2g, self.bf1, self.bf2, self.eta = initguess

        self.w11 = (self.bf1 + self.eta/2) # near demod 2
        self.w21 = (self.bf2 - self.eta/2) # near demod 5
        self.w12 = (self.bf1 + self.eta/2) # same as w11
        self.w22 = (self.bf2 - self.eta/2) # same as w22

        
        # self.scale1s = bool(scaleop == 0)*1 + bool(scaleop == 1)*1*max(np.abs(exp1.C2[exp1.iLIAend:-1]))
        # self.scale2s = self.scale1s
        # self.scale3s = bool(scaleop == 0)*1 + bool(scaleop == 1)*1*max(np.abs(exp2.C2[exp2.iLIAend:-1]))
        # self.scale4s = self.scale3s
        
        # I dont want to carry around all these selfs
        self.exp1 = exp1
        self.exp2 = exp2

        return
    
    def prepare_model(self, guess_array, maxdev = 200):
        '''
        Unpacks the guess array and prepares the model for fitting

        modeorder: booleanif for if the experiment order matches demods sorted by frequence
        currently: True if exp1.demod2f < exp2.demod2f, False if exp1.demod2f > exp2.demod2f
        Experiment 1 is at the lower frequency mode, true, else false
        If you pass eigenvalue guesses, it incorporates them
        Otherwise it takes the guessed values from self.f1g, ...
        '''

        # I dont want to carry around all these selfs
        exp1 = self.exp1
        exp2 = self.exp2

        # A1sg, A2sg, phi1sg, phi2sg, B1sg, B2sg, qhi1sg, qhi2sg, *backgrounds_eigenvalues = guess_array
        L1g, L2g, K1g, K2g, phiL1g, phiL2g, phiK1g, phiK2g, phiKdiffg, *backgrounds_eigenvalues = guess_array

        model = two_prep_two_demod_response_model

        #set fixed parameters in the model
        model.set_param_hint('w1_1', value = self.w11, vary = False)
        model.set_param_hint('w2_1', value = self.w21, vary = False)
        model.set_param_hint('w1_2', value = self.w12, vary = False)
        model.set_param_hint('w2_2', value = self.w22, vary = False)
        model.set_param_hint('tau', value = exp1.tauLIA, vary = False)

        model.set_param_hint('demodf1_1', value = exp1.demod2f, vary = False)
        model.set_param_hint('demodf2_1', value = exp1.demod5f, vary = False)   
        model.set_param_hint('demodf1_2', value = exp2.demod2f, vary = False)
        model.set_param_hint('demodf2_2', value = exp2.demod5f, vary = False)


        # Provide intial guesses to variable parameters
        model.set_param_hint('L1', value = L1g)
        model.set_param_hint('L2', value = L2g)
        model.set_param_hint('phiL1', value = phiL1g)
        model.set_param_hint('phiL2', value = phiL2g)

        model.set_param_hint('K1', value = K1g)
        model.set_param_hint('K2', value = K2g)
        model.set_param_hint('phiK1', value = phiK1g)
        model.set_param_hint('phiK2', value = phiK2g)
        model.set_param_hint('phiKrel', value = phiKdiffg)

        if len(backgrounds_eigenvalues) == 0:
            # If we're fitting from scratch
            model.set_param_hint('bkgR1_1', value = 0)
            model.set_param_hint('bkgI1_1', value = 0)
            model.set_param_hint('bkgR2_1', value = 0)
            model.set_param_hint('bkgI2_1', value = 0)
            model.set_param_hint('bkgR1_2', value = 0)
            model.set_param_hint('bkgI1_2', value = 0)
            model.set_param_hint('bkgR2_2', value = 0)
            model.set_param_hint('bkgI2_2', value = 0)

            gambarg = (self.gam1g + self.gam2g)/2
            dgamg = (self.gam2g - self.gam1g)
            fbarg = (self.f1g + self.f2g)/2
            dfg = (self.f2g - self.f1g)
            model.set_param_hint('fbar', value = fbarg, min = -maxdev + fbarg, max = maxdev + fbarg, vary = True)#200 Hz
            model.set_param_hint('df', value = dfg, min = -maxdev + dfg, max = maxdev + dfg, vary = True)
            model.set_param_hint('gammabar', value = gambarg, min = -maxdev + gambarg, max = maxdev + gambarg, vary = True)
            model.set_param_hint('dgamma', value = dgamg, min = -maxdev + dgamg, max = maxdev + dgamg, vary = True)       

        elif len(backgrounds_eigenvalues) == 12:
            # If we've already got a guess for the background and eigenvalues from a previous fit
            bkgR1_1g, bkgI1_1g, bkgR1_2g, bkgI1_2g, bkgR2_1g, bkgI2_1g, bkgR2_2g, bkgI2_2g = backgrounds_eigenvalues[:8]
            fbarg, dfg, gambarg, dgamg = backgrounds_eigenvalues[8:]

            model.set_param_hint('bkgR1_1', value = bkgR1_1g)
            model.set_param_hint('bkgI1_1', value = bkgI1_1g)
            model.set_param_hint('bkgR2_1', value = bkgR2_1g)
            model.set_param_hint('bkgI2_1', value = bkgI2_1g)
            model.set_param_hint('bkgR1_2', value = bkgR1_2g)
            model.set_param_hint('bkgI1_2', value = bkgI1_2g)
            model.set_param_hint('bkgR2_2', value = bkgR2_2g)
            model.set_param_hint('bkgI2_2', value = bkgI2_2g)

            model.set_param_hint('fbar', value = fbarg, min = -maxdev + fbarg, max = maxdev + fbarg, vary = True)#200 Hz
            model.set_param_hint('df', value = dfg, min = -maxdev + dfg, max = maxdev + dfg, vary = True)
            model.set_param_hint('gammabar', value = gambarg, min = -maxdev + gambarg, max = maxdev + gambarg, vary = True)
            model.set_param_hint('dgamma', value = dgamg, min = -maxdev + dgamg, max = maxdev + dgamg, vary = True)     


        else:
            print(backgrounds_eigenvalues)
            raise ValueError("Guess array is an invalid length")
        # self.t1 = exp1.time[exp1.iLIAend:-1]-exp1.time[exp1.iLIAstart]
        # self.t2 = exp2.time[exp2.iLIAend:-1]-exp2.time[exp2.iLIAstart]
        self.t1 = exp1.time[exp1.iLIAend:-1]-exp1.time[exp1.iloopend]
        self.t2 = exp2.time[exp2.iLIAend:-1]-exp1.time[exp2.iloopend]

        return model

    def dofits(self, guesss, maxdev = 200):
        '''
        Having loaded all the data into the object: actually fit the data
        '''
        # Again, being lazy with the selfs
        exp1 = self.exp1
        exp2 = self.exp2

        # Prepare lmfit model
        model= self.prepare_model(guesss, maxdev)

        # Get the data into the right format
        # fitdat1 = (np.real(exp1.C2[exp1.iLIAend:-1])/self.scale1s) + 1j*(np.imag(exp1.C2[exp1.iLIAend:-1])/self.scale2s) # data from exp 1, lower frequency (3,3)
        # fitdat2 = (np.real(exp2.C2[exp2.iLIAend:-1])/self.scale3s) + 1j*(np.imag(exp2.C2[exp2.iLIAend:-1])/self.scale4s) # data from exp 2, higher frequency (5,3)
        fitdat1 = exp1.C2[exp1.iLIAend:-1]
        fitdat2 = exp1.C5[exp1.iLIAend:-1]
        fitdat3 = exp2.C2[exp2.iLIAend:-1]
        fitdat4 = exp2.C5[exp2.iLIAend:-1]
        fitdat = np.concatenate([fitdat1, fitdat2, fitdat3, fitdat4])
        t1axis = self.t1
        t2axis = self.t2

        # Fit the data formated above using the perapred model
        params = model.make_params()
        FR = model.fit(fitdat, params, t_1 = t1axis, t_2 = t2axis, max_nfev=1000000)
        
        return FR, model
    
    def run_fit_generate_data(self, guesss):
        '''
        Essentially a wrapper around dofits
        Also generates:
        - Theory curves and experimental data for plotting
        - The fit report, as self.fit_report
        - An array with relevant parameters and the best-fit values, as self.return_array
        - A guess array to feed forward to the next fit, as self.next_guess
        '''

        # Again, being lazy with the selfs
        exp1 = self.exp1
        exp2 = self.exp2

        #do initial fitting
        FR_initial, _ = self.dofits(guesss)

        #unpack simple ringdown

        guess_1 = unpack_FR(FR_initial)

        #do fitting again to reduce fitting failures
        self.fit_report, self.modelF = self.dofits(guess_1, maxdev = 1000)
        self.next_guess = unpack_FR(self.fit_report)

        # Make theory curves
        n = 500
        self.exp1_fit_t = np.linspace(exp1.time[exp1.iLIAend], exp1.time[-1], n) 
        self.exp2_fit_t = np.linspace(exp2.time[exp2.iLIAend], exp2.time[-1], n)
        dtemp = self.modelF.eval(self.fit_report.params, t_1 = self.exp1_fit_t - exp1.time[exp1.iloopend], 
                                                    t_2 = self.exp2_fit_t - exp2.time[exp2.iloopend])
        fitarray_temp = [self.exp1_fit_t, dtemp[:n], self.exp2_fit_t, dtemp[2*n:3*n]]
        self.plotter = Plotter(self.exp1, self.exp2, fitarray_temp)
        
        header, values = unpack_FR_return(self.fit_report)

        return header, values

'''
Plotters
'''

class Plotter:
    def __init__(self, exp1, exp2, fitarray):
        self.exp1 = exp1
        self.exp2 = exp2
        self.fitarray = fitarray
        self.exp1_fit_t, self.exp1_fit_C2, self.exp2_fit_t, self.exp2_fit_C2 = fitarray

    def make_theory_curves(self, model = None, FR = None):
        if model == None or FR == None:
            raise ValueError("Need to provide either params or fit report")
        else:
            n = 500
            # self.exp1_fit_t = np.linspace(self.exp1.time[self.exp1.iLIAend], self.exp1.time[-1], n) 
            # self.exp2_fit_t = np.linspace(self.exp2.time[self.exp2.iLIAend], self.exp2.time[-1], n)
            self.exp1_fit_t = np.linspace(self.exp1.time[self.exp1.iLIAstart], self.exp1.time[-1], n) 
            self.exp2_fit_t = np.linspace(self.exp2.time[self.exp2.iLIAstart], self.exp2.time[-1], n)
            dtemp = model.eval(FR.params, t_1 = self.exp1_fit_t - self.exp1.time[self.exp1.iLIAstart],
             t_2 = self.exp2_fit_t - self.exp2.time[self.exp2.iLIAstart])
            # dtemp = model.eval(FR.params, t_1 = self.exp1_fit_t - self.exp1.time[self.exp1.iLIAend],
            #  t_2 = self.exp2_fit_t - self.exp2.time[self.exp2.iLIAend])
            self.exp1_fit_C2 = dtemp[0:n]
            self.exp1_fit_C5 = dtemp[n:2*n]
            self.exp2_fit_C2 = dtemp[2*n:3*n]
            self.exp2_fit_C5 = dtemp[3*n:4*n]
            #filler
        # filler 
        return

    def plot_params(self, ax, fsz, wid, tpad, lpad, y_label = None, x_label = None, title = None):
        ax.xaxis.set_tick_params(which='major', size=5, width=wid, direction='in', labelsize = fsz, pad = tpad)
        ax.yaxis.set_tick_params(which='major', size=5, width=wid, direction='in', labelsize = fsz, pad = tpad)
        ax.autoscale(enable=True, axis='both', tight=False)
        ax.grid(True, color = 'lightgray', linewidth = wid)
        ax.minorticks_off()
        ax.margins(0.025, 0.05)
        if y_label is not None:
            ax.set_ylabel(y_label, labelpad=lpad, fontsize = fsz)
        if x_label is not None:
            ax.set_xlabel(x_label, labelpad=lpad, fontsize = fsz)
        if title is not None:
            ax.set_title(title, fontsize = fsz) 
        return

    def plot_driven(self, plotname, savefolder, show = False, save = True):
        # plots the driven demods
        # start plotting
        fsz = 10
        wid = 0.8
        lwid = 1
        plwid = 4.5
        thwid = wid
        lpad = 6
        pcol = 'darkorange'
        fcol = 'maroon'
        pcol1 = '#03A89E'
        fcol1 = '#191970'
        fwid = 1.5
        LIAcol = '#ADD8E6'
        loopcol = '#B5B5B5'
        tpad = 5
        mpl.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = fsz
        plt.rcParams['axes.linewidth'] = wid
        mpl.rcParams['axes.formatter.useoffset'] = False

        fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(16,9),facecolor='white',sharex='col',
                                gridspec_kw=dict({'height_ratios': [1, 1, 1, 1]}, hspace=0.0,wspace = 0.1))

        # both demods abs simple log  
        axs[0,0].plot(self.exp1.time - self.exp1.tLIAstart, np.abs(self.exp1.C2), color=pcol, linewidth = plwid)
        axs[0,0].plot(self.exp2.time - self.exp2.tLIAstart, np.abs(self.exp2.C2), color=pcol1, linewidth = plwid)
        axs[0,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[0,0].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C2), color=fcol1, linewidth = fwid)
        axs[0,0].axvspan(0, self.exp1.tLIAend - self.exp1.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5, label = 'LIA window')
        axs[0,0].legend(fontsize = fsz, loc = 'best', frameon = True)
        axs[0,0].set_yscale("log")
        self.plot_params(axs[0,0], fsz, wid, tpad, lpad, y_label = "Abs[response] (mV)", title = "simple ringdown")

        # both demods abs simple linear
        axs[1,0].plot(self.exp1.time - self.exp1.tLIAstart, np.abs(self.exp1.C2), color=pcol, linewidth = plwid, label = 'data 1')
        axs[1,0].plot(self.exp2.time - self.exp2.tLIAstart, np.abs(self.exp2.C2), color=pcol1, linewidth = plwid, label = 'data 2')
        axs[1,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C2), color=fcol, linewidth = fwid, label = 'fit 1')
        axs[1,0].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C2), color=fcol1, linewidth = fwid, label = 'fit 2')
        axs[1,0].axvspan(0, self.exp1.tLIAend - self.exp1.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5)
        axs[1,0].legend(fontsize = fsz, loc = 'best', frameon = True)
        self.plot_params(axs[1,0], fsz, wid, tpad, lpad, y_label = "Abs[response] (mV)")

        # both demods real simple
        axs[2,0].plot(self.exp1.time - self.exp1.tLIAstart, np.real(self.exp1.C2), color=pcol, linewidth = plwid)
        axs[2,0].plot(self.exp2.time - self.exp2.tLIAstart, np.real(self.exp2.C2), color=pcol1, linewidth = plwid)
        axs[2,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.real(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[2,0].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.real(self.exp2_fit_C2), color=fcol1, linewidth = fwid)
        axs[2,0].axvspan(0, self.exp1.tLIAend - self.exp1.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5)
        self.plot_params(axs[2,0], fsz, wid, tpad, lpad, y_label = "in-phase response (mV)")

        # both demods imaginary simple
        axs[3,0].plot(self.exp1.time - self.exp1.tLIAstart, np.imag(self.exp1.C2), color=pcol, linewidth = plwid)
        axs[3,0].plot(self.exp2.time - self.exp2.tLIAstart, np.imag(self.exp2.C2), color=pcol1, linewidth = plwid)
        axs[3,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.imag(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[3,0].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.imag(self.exp2_fit_C2), color=fcol1, linewidth = fwid)
        axs[3,0].axvspan(0, self.exp1.tLIAend - self.exp1.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5)
        self.plot_params(axs[3,0], fsz, wid, tpad, lpad, y_label = "quadrature response (mV)", x_label = "time (s)")

        # both demods abs loop log
        axs[0,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.abs(self.exp1.C2[self.exp1.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[0,1].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.abs(self.exp2.C2[self.exp2.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[0,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[0,1].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C2), color=fcol1, linewidth = fwid)
        axs[0,1].set_yscale("log")
        self.plot_params(axs[0,1], fsz, wid, tpad, lpad)

        # both demods abs loop linear
        axs[1,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.abs(self.exp1.C2[self.exp1.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[1,1].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.abs(self.exp2.C2[self.exp2.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[1,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[1,1].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C2), color=fcol1, linewidth = fwid)
        self.plot_params(axs[1,1], fsz, wid, tpad, lpad)
    
        # both demods real loop
        axs[2,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.real(self.exp1.C2[self.exp1.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[2,1].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.real(self.exp2.C2[self.exp2.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[2,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.real(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[2,1].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.real(self.exp2_fit_C2), color=fcol1, linewidth = fwid)
        self.plot_params(axs[2,1], fsz, wid, tpad, lpad)

        # both demods imaginary loop
        axs[3,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.imag(self.exp1.C2[self.exp1.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[3,1].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.imag(self.exp2.C2[self.exp1.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[3,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.imag(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[3,1].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.imag(self.exp2_fit_C2), color=fcol1, linewidth = fwid)
        self.plot_params(axs[3,1], fsz, wid, tpad, lpad, x_label = "time (s)")
        # ... (omitted for brevity)

        if save:
            plotname, _ = file_handling.make_plotfilename(plotname, savefolder)
            plt.savefig(plotname, dpi = 100,bbox_inches='tight') # ALMOST THE ENTIRE RUN TIME IS SPENT SAVING
        if not show:
            plt.close()


    def plot_all(self, plotname, savefolder, show = False, save = True):
        # plots all four demods
        # start plotting
        fsz = 10
        wid = 0.8
        lwid = 1
        plwid = 4.5
        thwid = wid
        lpad = 6
        pcol = 'darkorange'
        fcol = 'maroon'
        pcol1 = '#03A89E'
        fcol1 = '#191970'
        fwid = 1.5
        LIAcol = '#ADD8E6'
        loopcol = '#B5B5B5'
        tpad = 5
        mpl.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = fsz
        plt.rcParams['axes.linewidth'] = wid
        mpl.rcParams['axes.formatter.useoffset'] = False

        fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(20,9),facecolor='white',sharex='col',
                                gridspec_kw=dict({'height_ratios': [1, 1, 1, 1]}, hspace=0.0,wspace = 0.1))


        # experiment 1

        # both demods abs simple log  
        axs[0,0].plot(self.exp1.time - self.exp1.tLIAstart, np.abs(self.exp1.C2), color=pcol, linewidth = plwid)
        axs[0,0].plot(self.exp1.time - self.exp1.tLIAstart, np.abs(self.exp1.C5), color=pcol1, linewidth = plwid)
        axs[0,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[0,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C5), color=fcol1, linewidth = fwid)
        axs[0,0].axvspan(0, self.exp1.tLIAend - self.exp1.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5, label = 'LIA window')
        axs[0,0].legend(fontsize = fsz, loc = 'best', frameon = True)
        axs[0,0].set_yscale("log")
        self.plot_params(axs[0,0], fsz, wid, tpad, lpad, y_label = "Abs[response] (mV)", title = "simple ringdown, drive (3,3)")

        # both demods abs simple linear
        axs[1,0].plot(self.exp1.time - self.exp1.tLIAstart, np.abs(self.exp1.C2), color=pcol, linewidth = plwid, label = 'demod 2')
        axs[1,0].plot(self.exp1.time - self.exp1.tLIAstart, np.abs(self.exp1.C5), color=pcol1, linewidth = plwid, label = 'demod 5')
        axs[1,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C2), color=fcol, linewidth = fwid, label = 'fit d2')
        axs[1,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C5), color=fcol1, linewidth = fwid, label = 'fit d5')
        axs[1,0].axvspan(0, self.exp1.tLIAend - self.exp1.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5)
        axs[1,0].legend(fontsize = fsz, loc = 'best', frameon = True)
        self.plot_params(axs[1,0], fsz, wid, tpad, lpad, y_label = "Abs[response] (mV)")

        # both demods real simple
        axs[2,0].plot(self.exp1.time - self.exp1.tLIAstart, np.real(self.exp1.C2), color=pcol, linewidth = plwid)
        axs[2,0].plot(self.exp1.time - self.exp1.tLIAstart, np.real(self.exp1.C5), color=pcol1, linewidth = plwid)
        axs[2,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.real(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[2,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.real(self.exp1_fit_C5), color=fcol1, linewidth = fwid)
        axs[2,0].axvspan(0, self.exp1.tLIAend - self.exp1.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5)
        self.plot_params(axs[2,0], fsz, wid, tpad, lpad, y_label = "in-phase response (mV)")

        # both demods imaginary simple
        axs[3,0].plot(self.exp1.time - self.exp1.tLIAstart, np.imag(self.exp1.C2), color=pcol, linewidth = plwid)
        axs[3,0].plot(self.exp1.time - self.exp1.tLIAstart, np.imag(self.exp1.C5), color=pcol1, linewidth = plwid)
        axs[3,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.imag(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[3,0].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.imag(self.exp1_fit_C5), color=fcol1, linewidth = fwid)
        axs[3,0].axvspan(0, self.exp1.tLIAend - self.exp1.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5)
        self.plot_params(axs[3,0], fsz, wid, tpad, lpad, y_label = "quadrature response (mV)", x_label = "time (s)")

        # both demods abs loop log
        axs[0,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.abs(self.exp1.C2[self.exp1.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[0,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.abs(self.exp1.C5[self.exp1.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[0,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[0,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C5), color=fcol1, linewidth = fwid)
        axs[0,1].set_yscale("log")
        self.plot_params(axs[0,1], fsz, wid, tpad, lpad)

        # both demods abs loop linear
        axs[1,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.abs(self.exp1.C2[self.exp1.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[1,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.abs(self.exp1.C5[self.exp1.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[1,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[1,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.abs(self.exp1_fit_C5), color=fcol1, linewidth = fwid)
        self.plot_params(axs[1,1], fsz, wid, tpad, lpad)
    
        # both demods real loop
        axs[2,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.real(self.exp1.C2[self.exp1.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[2,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.real(self.exp1.C5[self.exp2.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[2,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.real(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[2,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.real(self.exp1_fit_C5), color=fcol1, linewidth = fwid)
        self.plot_params(axs[2,1], fsz, wid, tpad, lpad)

        # both demods imaginary loop
        axs[3,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.imag(self.exp1.C2[self.exp1.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[3,1].plot(self.exp1.time[self.exp1.iLIAend:-1] - self.exp1.tLIAstart, np.imag(self.exp1.C5[self.exp1.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[3,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.imag(self.exp1_fit_C2), color=fcol, linewidth = fwid)
        axs[3,1].plot(self.exp1_fit_t - self.exp1.tLIAstart, np.imag(self.exp1_fit_C5), color=fcol1, linewidth = fwid)
        self.plot_params(axs[3,1], fsz, wid, tpad, lpad, x_label = "time (s)")




        # experiment 2

                # both demods abs simple log  
        axs[0,2].plot(self.exp2.time - self.exp2.tLIAstart, np.abs(self.exp2.C2), color=pcol, linewidth = plwid)
        axs[0,2].plot(self.exp2.time - self.exp2.tLIAstart, np.abs(self.exp2.C5), color=pcol1, linewidth = plwid)
        axs[0,2].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C2), color=fcol, linewidth = fwid)
        axs[0,2].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C5), color=fcol1, linewidth = fwid)
        axs[0,2].axvspan(0, self.exp2.tLIAend - self.exp2.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5, label = 'LIA window')
        axs[0,2].legend(fontsize = fsz, loc = 'best', frameon = True)
        axs[0,2].set_yscale("log")
        self.plot_params(axs[0,2], fsz, wid, tpad, lpad, y_label = "Abs[response] (mV)", title = "simple ringdown, drive (5,3)")

        # both demods abs simple linear
        axs[1,2].plot(self.exp2.time - self.exp2.tLIAstart, np.abs(self.exp2.C2), color=pcol, linewidth = plwid, label = 'demod 2')
        axs[1,2].plot(self.exp2.time - self.exp2.tLIAstart, np.abs(self.exp2.C5), color=pcol1, linewidth = plwid, label = 'demod 5')
        axs[1,2].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C2), color=fcol, linewidth = fwid, label = 'fit d2')
        axs[1,2].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C5), color=fcol1, linewidth = fwid, label = 'fit d5')
        axs[1,2].axvspan(0, self.exp2.tLIAend - self.exp2.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5)
        axs[1,2].legend(fontsize = fsz, loc = 'best', frameon = True)
        self.plot_params(axs[1,2], fsz, wid, tpad, lpad, y_label = "Abs[response] (mV)")

        # both demods real simple
        axs[2,2].plot(self.exp2.time - self.exp2.tLIAstart, np.real(self.exp2.C2), color=pcol, linewidth = plwid)
        axs[2,2].plot(self.exp2.time - self.exp2.tLIAstart, np.real(self.exp2.C5), color=pcol1, linewidth = plwid)
        axs[2,2].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.real(self.exp2_fit_C2), color=fcol, linewidth = fwid)
        axs[2,2].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.real(self.exp2_fit_C5), color=fcol1, linewidth = fwid)
        axs[2,2].axvspan(0, self.exp2.tLIAend - self.exp2.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5)
        self.plot_params(axs[2,2], fsz, wid, tpad, lpad, y_label = "in-phase response (mV)")

        # both demods imaginary simple
        axs[3,2].plot(self.exp2.time - self.exp2.tLIAstart, np.imag(self.exp2.C2), color=pcol, linewidth = plwid)
        axs[3,2].plot(self.exp2.time - self.exp2.tLIAstart, np.imag(self.exp2.C5), color=pcol1, linewidth = plwid)
        axs[3,2].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.imag(self.exp2_fit_C2), color=fcol, linewidth = fwid)
        axs[3,2].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.imag(self.exp2_fit_C5), color=fcol1, linewidth = fwid)
        axs[3,2].axvspan(0, self.exp2.tLIAend - self.exp2.tLIAstart, facecolor=LIAcol, alpha=0.45, zorder=2.5)
        self.plot_params(axs[3,2], fsz, wid, tpad, lpad, y_label = "quadrature response (mV)", x_label = "time (s)")

        # both demods abs loop log
        axs[0,3].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.abs(self.exp2.C2[self.exp2.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[0,3].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.abs(self.exp2.C5[self.exp2.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[0,3].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C2), color=fcol, linewidth = fwid)
        axs[0,3].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C5), color=fcol1, linewidth = fwid)
        axs[0,3].set_yscale("log")
        self.plot_params(axs[0,3], fsz, wid, tpad, lpad)

        # both demods abs loop linear
        axs[1,3].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.abs(self.exp2.C2[self.exp2.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[1,3].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.abs(self.exp2.C5[self.exp2.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[1,3].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C2), color=fcol, linewidth = fwid)
        axs[1,3].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.abs(self.exp2_fit_C5), color=fcol1, linewidth = fwid)
        self.plot_params(axs[1,3], fsz, wid, tpad, lpad)
    
        # both demods real loop
        axs[2,3].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.real(self.exp2.C2[self.exp2.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[2,3].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.real(self.exp2.C5[self.exp2.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[2,3].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.real(self.exp2_fit_C2), color=fcol, linewidth = fwid)
        axs[2,3].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.real(self.exp2_fit_C5), color=fcol1, linewidth = fwid)
        self.plot_params(axs[2,3], fsz, wid, tpad, lpad)

        # both demods imaginary loop
        axs[3,3].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.imag(self.exp2.C2[self.exp2.iLIAend:-1]), color=pcol, linewidth = plwid)
        axs[3,3].plot(self.exp2.time[self.exp2.iLIAend:-1] - self.exp2.tLIAstart, np.imag(self.exp2.C5[self.exp2.iLIAend:-1]), color=pcol1, linewidth = plwid)
        axs[3,3].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.imag(self.exp2_fit_C2), color=fcol, linewidth = fwid)
        axs[3,3].plot(self.exp2_fit_t - self.exp2.tLIAstart, np.imag(self.exp2_fit_C5), color=fcol1, linewidth = fwid)
        self.plot_params(axs[3,3], fsz, wid, tpad, lpad, x_label = "time (s)")





        if save:
            plotname, _ = file_handling.make_plotfilename(plotname, savefolder)
            plt.savefig(plotname, dpi = 100,bbox_inches='tight') # ALMOST THE ENTIRE RUN TIME IS SPENT SAVING
        if not show:
            plt.close()














