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
import fitting.standard_fit_routines as fit
import fitting.Rotating_frame_fit_routines as rotfit
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


def load_tap(foldertap, itap):
    '''
    to load taps
    '''

    #import relevant tap csv file
    tapdata = np.loadtxt(foldertap + "summary of ringdown_{}.csv".format(itap))
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





'''
fit all class
'''
class fit_all():
    def __init__(self, file1a, file1b, initguess, shift = [0,0], scaleop = 0):
        '''
        The idea here is that we drive at two different configurations: (a) on mode 1, preparing (A,0)
        and (b) on mode 2, preparing (0,B). This gets us two different ringdowns.

        file1a contains the data that we take after driving near mode 1
        file1b contains the data that we take after driving near mode 2

        These data get loaded into two objects, which are children of this class, and called exp1 and exp2 respectively

        shift is a "cluge" to escape non-linear ringdowns. Given separately for demod 2 and demod 5, in units ms

        #scaleop is for putting both mode data at same footing.

        '''
        
        #load dynamics data: file1a, mode 1
        exp1 = load_data(file1a)
        exp2 = load_data(file1b)


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

        self.w11 = (self.bf1 + self.eta/2)
        self.w21 = (self.bf1 + self.eta/2) # near demod 1
        self.w12 = (self.bf2 - self.eta/2) # near demod 2
        self.w22 = (self.bf2 - self.eta/2)

        
        self.scale1s = bool(scaleop == 0)*1 + bool(scaleop == 1)*1*max(np.abs(exp1.C2[exp1.iLIAend:-1]))
        self.scale2s = self.scale1s
        self.scale3s = bool(scaleop == 0)*1 + bool(scaleop == 1)*1*max(np.abs(exp2.C2[exp2.iLIAend:-1]))
        self.scale4s = self.scale3s
        
        # I dont want to carry around all these selfs
        self.exp1 = exp1
        self.exp2 = exp2

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
        exp2 = self.exp2

        A1sg, A2sg, phi1sg, phi2sg, B1sg, B2sg, qhi1sg, qhi2sg, *backgrounds_eigenvalues = guess_array

        model = two_prep_one_demod_response_model

        #set fixed parameters in the model
        model.set_param_hint('w1_1', value = self.w11, vary = False)
        model.set_param_hint('w2_1', value = self.w21, vary = False)
        model.set_param_hint('w1_2', value = self.w12, vary = False)
        model.set_param_hint('w2_2', value = self.w22, vary = False)
        model.set_param_hint('tau', value = exp1.tauLIA, vary = False)

        if modeorder:
            # set model parameters to the correct demod frequency and all the scalings correctly
            model.set_param_hint('scaleR_1', value = self.scale1s, vary = False)
            model.set_param_hint('scaleI_1', value = self.scale2s, vary = False)
            model.set_param_hint('scaleR_2', value = self.scale3s, vary = False)
            model.set_param_hint('scaleI_2', value = self.scale4s, vary = False)

            model.set_param_hint('demodf1', value = exp1.demod2f, vary = False)
            model.set_param_hint('demodf2', value = exp2.demod2f, vary = False)            

        else:
            model.set_param_hint('scaleR_1', value = self.scale3s, vary = False)
            model.set_param_hint('scaleI_1', value = self.scale4s, vary = False)
            model.set_param_hint('scaleR_2', value = self.scale1s, vary = False)
            model.set_param_hint('scaleI_2', value = self.scale2s, vary = False)

            model.set_param_hint('demodf1', value = exp2.demod2f, vary = False)
            model.set_param_hint('demodf2', value = exp1.demod2f, vary = False)

        # Provide intial guesses to variable parameters
        model.set_param_hint('A1_1', value = A1sg)
        model.set_param_hint('A2_1', value = A2sg)
        model.set_param_hint('phi1_1', value = phi1sg)
        model.set_param_hint('phi2_1', value = phi2sg)        

        model.set_param_hint('A1_2', value = B1sg)
        model.set_param_hint('A2_2', value = B2sg)
        model.set_param_hint('phi1_2', value = qhi1sg)
        model.set_param_hint('phi2_2', value = qhi2sg)

        if len(backgrounds_eigenvalues) == 0:
            # If we're fitting from scratch
            model.set_param_hint('bkgR_1', value = 0)
            model.set_param_hint('bkgI_1', value = 0)
            model.set_param_hint('bkgR_2', value = 0)
            model.set_param_hint('bkgI_2', value = 0)
            model.set_param_hint('dw1', value = self.f1g, min = -maxdev + self.f1g, max = maxdev + self.f1g, vary = True)#200 Hz
            model.set_param_hint('dw2', value = self.f2g, min = -maxdev + self.f2g, max = maxdev + self.f2g, vary = True)
            model.set_param_hint('gam1', value = self.gam1g, min = -maxdev + self.gam1g, max = maxdev + self.gam1g, vary = True)
            model.set_param_hint('gam2', value = self.gam2g, min = -maxdev + self.gam2g, max = maxdev + self.gam2g, vary = True)       

        elif len(backgrounds_eigenvalues) == 8:
            # If we've already got a guess for the background and eigenvalues from a previous fit
            bkgR1g, bkgI1g, bkgR2g, bkgI2g, f1sg, f2sg, g1sg, g2sg = backgrounds_eigenvalues
            model.set_param_hint('bkgR_1', value = bkgR1g)
            model.set_param_hint('bkgI_1', value = bkgI1g)
            model.set_param_hint('bkgR_2', value = bkgR2g)
            model.set_param_hint('bkgI_2', value = bkgI2g)
            model.set_param_hint('dw1', value = f1sg, min = -maxdev + f1sg, max = maxdev + f1sg, vary = True)#200 Hz
            model.set_param_hint('dw2', value = f2sg, min = -maxdev + f2sg, max = maxdev + f2sg, vary = True)
            model.set_param_hint('gam1', value = g1sg, min = -maxdev + g1sg, max = maxdev + g1sg, vary = True)
            model.set_param_hint('gam2', value = g2sg, min = -maxdev + g2sg, max = maxdev + g2sg, vary = True)

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
        exp2 = self.exp2

        # Prepare lmfit model
        if exp1.demod2f < exp2.demod2f:
            model= self.prepare_model(guesss, True, maxdev)

            # Get the data into the right format
            fitdat1 = (np.real(exp1.C2[exp1.iLIAend:-1])/self.scale1s) + 1j*(np.imag(exp1.C2[exp1.iLIAend:-1])/self.scale2s) # data from exp 1, lower frequency (3,3)
            fitdat2 = (np.real(exp2.C2[exp2.iLIAend:-1])/self.scale3s) + 1j*(np.imag(exp2.C2[exp2.iLIAend:-1])/self.scale4s) # data from exp 2, higher frequency (5,3)
            fitdat = np.concatenate([fitdat1, fitdat2])
            t1axis = exp1.time[exp1.iLIAend:-1]-exp1.time[exp1.iLIAstart]
            t2axis = exp2.time[exp2.iLIAend:-1]-exp2.time[exp2.iLIAstart]

        else:
            model= self.prepare_model(guesss, False, maxdev)

            # Get the data into the right format
            fitdat2 = (np.real(exp1.C2[exp1.iLIAend:-1])/self.scale1s) + 1j*(np.imag(exp1.C2[exp1.iLIAend:-1])/self.scale2s) # data from exp 1, higher frequency (5,3)
            fitdat1 = (np.real(exp2.C2[exp2.iLIAend:-1])/self.scale3s) + 1j*(np.imag(exp2.C2[exp2.iLIAend:-1])/self.scale4s) # data from exp 2, lower frequency (3,3)
            fitdat = np.concatenate([fitdat1, fitdat2])
            t2axis = exp1.time[exp1.iLIAend:-1]-exp1.time[exp1.iLIAstart]
            t1axis = exp2.time[exp2.iLIAend:-1]-exp2.time[exp2.iLIAstart]

        # Fit the data formated above using the perapred model
        params = model.make_params()
        FR = model.fit(fitdat, params, t_1 = t1axis, t_2 = t2axis, max_nfev=1000000)
        
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
        exp2 = self.exp2

        #do initial fitting
        FRsi = self.dofits(guesss)

        #unpack simple ringdown
        A1si = FRsi.params['A1_1'].value
        A2si = FRsi.params['A2_1'].value
        phi1si = FRsi.params['phi1_1'].value
        phi2si = FRsi.params['phi2_1'].value
        dw1si = FRsi.params['dw1'].value
        dw2si = FRsi.params['dw2'].value
        gam1si = FRsi.params['gam1'].value
        gam2si = FRsi.params['gam2'].value
        bkgR1si = FRsi.params['bkgR_1'].value
        bkgI1si = FRsi.params['bkgI_1'].value

        B1si = FRsi.params['A1_2'].value
        B2si = FRsi.params['A2_2'].value
        qhi1si = FRsi.params['phi1_2'].value
        qhi2si = FRsi.params['phi2_2'].value
        bkgR2si = FRsi.params['bkgR_2'].value
        bkgI2si = FRsi.params['bkgI_2'].value

        guesssi = [np.abs(A1si), np.abs(A2si), phi1si%(2*np.pi), phi2si%(2*np.pi), np.abs(B1si), np.abs(B2si), qhi1si%(2*np.pi), qhi2si%(2*np.pi),
                    bkgR1si, bkgI1si, bkgR2si, bkgI2si, dw1si, dw2si, gam1si, gam2si]  

        #do fitting again to reduce fitting failures
        FRs = self.dofits(guesssi, maxdev = 1000)
        A1s = FRs.params['A1_1'].value
        A2s = FRs.params['A2_1'].value
        phi1s = FRs.params['phi1_1'].value
        phi2s = FRs.params['phi2_1'].value
        dw1s = FRs.params['dw1'].value
        dw2s = FRs.params['dw2'].value
        gam1s = FRs.params['gam1'].value
        gam2s = FRs.params['gam2'].value
        bkgR1s = FRs.params['bkgR_1'].value
        bkgI1s = FRs.params['bkgI_1'].value

        B1s = FRs.params['A1_2'].value
        B2s = FRs.params['A2_2'].value
        qhi1s = FRs.params['phi1_2'].value
        qhi2s = FRs.params['phi2_2'].value
        bkgR2s = FRs.params['bkgR_2'].value
        bkgI2s = FRs.params['bkgI_2'].value



        #make theory curves
        if exp1.demod2f < exp2.demod2f:
            # for experiment 1 (data from file1a)
            self.exp1_fit_t = np.linspace(exp1.time[exp1.iLIAend], exp1.time[-1], 500)
            self.exp1_fit_C2 = two_osc_demod_response(self.exp1_fit_t - exp1.time[exp1.iLIAstart], A1s, A2s, phi1s, phi2s, self.w11, self.w21, gam1s, gam2s, 
                                        dw1s, dw2s, exp1.demod2f, exp1.tauLIA, bkgR1s, bkgI1s, 1, 1)
            # for experiment 2 (data from file1b)
            self.exp2_fit_t = np.linspace(exp2.time[exp2.iLIAend], exp2.time[-1], 500)
            self.exp2_fit_C2 = two_osc_demod_response(self.exp2_fit_t - exp2.time[exp2.iLIAstart], B1s, B2s, qhi1s, qhi2s, self.w12, self.w22, gam1s, gam2s, 
                                        dw1s, dw2s, exp2.demod2f, exp1.tauLIA, bkgR2s, bkgI2s, 1, 1)
            
            self.plotter = Plotter(self.exp1, self.exp2, [self.exp1_fit_t, self.exp1_fit_C2, self.exp2_fit_t, self.exp2_fit_C2])

        elif exp1.demod2f > exp2.demod2f:
            # for experiment 2 (data from file1b)
            self.exp1_fit_t = np.linspace(exp2.time[exp2.iLIAend], exp2.time[-1], 500)
            self.exp1_fit_C2 = two_osc_demod_response(self.exp1_fit_t - exp2.time[exp2.iLIAstart], A1s, A2s, phi1s, phi2s, self.w11, self.w21,  gam1s, gam2s,
                                        dw1s, dw2s,  exp2.demod2f, exp2.tauLIA, bkgR1s, bkgI1s, 1, 1)
            # for experiment 1 (data from file1a)
            self.exp2_fit_t = np.linspace(exp1.time[exp1.iLIAend], exp1.time[-1], 500)
            self.exp2_fit_C2 = two_osc_demod_response(self.exp2_fit_t - exp1.time[exp1.iLIAstart], B1s, B2s, qhi1s, qhi2s, self.w12, self.w22, gam1s, gam2s, 
                                        dw1s, dw2s, exp1.demod2f, exp1.tauLIA, bkgR2s, bkgI2s, 1, 1)
            
            self.plotter = Plotter(self.exp2, self.exp1, [self.exp1_fit_t, self.exp1_fit_C2, self.exp2_fit_t, self.exp2_fit_C2])
        
        # ret1 = [self.exp1_fit_t, self.exp1_fit_C2, self.exp2_fit_t, self.exp2_fit_C2]#for plotting Defunct

        # ret2 = [FRs]#for saving individual fit results
        self.fit_report = FRs

        self.return_array = np.array([exp1.demod2f, exp2.demod2f, self.w11 + dw1s, self.w22 + dw2s, gam1s, gam2s , self.bf1, self.bf2, self.eta,
                np.abs(A1s), np.abs(A2s), phi1s%(2*np.pi), phi2s%(2*np.pi), bkgR1s, bkgI1s, self.scale1s, self.scale2s,
                np.abs(B1s), np.abs(B2s), qhi1s%(2*np.pi), qhi2s%(2*np.pi), bkgR2s, bkgI2s, self.scale3s, self.scale4s])#for saving collated fit results
        
        self.return_array_labels = ['demod2f (Hz)', 'demod5f (Hz)', 'f1 (Hz)', 'f2 (Hz)', 'gamma1 (Hz)', 'gamma2 (Hz)', 'baref1 (Hz)', 'baref2 (Hz)', 'eta (Hz)',
                                    'A1 (mV)', 'A2 (mV)', 'phi1 (rad)', 'phi2 (rad)', 'bkgR1 (mV)', 'bkgI1 (mV)', 'scale1 (mV)', 'scale2 (mV)',
                                    'B1 (mV)', 'B2 (mV)', 'qhi1 (rad)', 'qhi2 (rad)', 'bkgR2 (mV)', 'bkgI2 (mV)', 'scale3 (mV)', 'scale4 (mV)']

        self.next_guess = [np.abs(A1s), np.abs(A2s), phi1s%(2*np.pi), phi2s%(2*np.pi), np.abs(B1s), np.abs(B2s), qhi1s%(2*np.pi), qhi2s%(2*np.pi),
                bkgR1s, bkgI1s,  bkgR2s, bkgI2s, dw1s, dw2s, gam1s, gam2s]#for guesss
        
        

        return

'''
Plotters
'''

class Plotter:
    def __init__(self, exp1, exp2, fitarray):
        self.exp1 = exp1
        self.exp2 = exp2
        self.fitarray = fitarray
        self.exp1_fit_t, self.exp1_fit_C2, self.exp2_fit_t, self.exp2_fit_C2 = fitarray

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

    def plot_all(self, plotname, savefolder, show = False, save = True):
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


