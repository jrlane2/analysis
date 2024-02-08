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
    
    def __init__(self, data, dataspecs, savefolder, stack_spec = [0.01, 0.01, 0, 0]):
        '''
        This gets all data and data specs in-situ, makes a folder to save C-avg-ed data.
        '''

        #distribute dataspecs
        [self.k1, self.k2, self.c1, self.c2] = stack_spec
        self.demod2f = dataspecs[0]
        self.demod5f = dataspecs[1]
        self.setlooptime = dataspecs[2]
        self.setdrivetime = dataspecs[3]
        self.setresttime = dataspecs[4]
        self.bandwidth = dataspecs[5]
        self.order = dataspecs[6]
        self.tLIA = self.settling_time(self.bandwidth, self.order, signal_level = 99)
        self.t_wait = dataspecs[8]
        self.shift = self.c1*self.setlooptime + self.c2*self.tLIA + 1*self.t_wait
        self.iters = dataspecs[7]
        self.setpreptime = dataspecs[9]
        self.meastimeproxy = self.iters*2*(self.setlooptime + self.setdrivetime + self.setresttime + self.setpreptime)*2
        #load data and kill nans
        self.dat = data
        [self.minv, self.maxv] = self.killnans2() # What does killnans actually return? The first and last indicies of the data to be averaged?
        #make arrays to operate on
        self.time = self.dat[:,0] - self.dat[0,0]#timestamps
        self.TTL1 = self.dat[:,1]#booled TTL1 - for control loops
        self.TTL2 = self.dat[:,2]#booled TTL2 - for simple ringdown
        self.demod2X = self.dat[:,3]*1e3
        self.demod2Y = self.dat[:,4]*1e3
        self.demod5X = self.dat[:,5]*1e3
        self.demod5Y = self.dat[:,6]*1e3
        self.C2 = self.demod2X + 1j*self.demod2Y
        self.C5 = self.demod5X + 1j*self.demod5Y
        
        
        #distribute save folder
        self.save_folder = savefolder
    
        
        return
    

    def killnans2(self):
        '''
        Function for removing nan-s
        Pervious version fails is there's data revival (i.e. nans before valid data)
        This version grabs the largest single chunk of non nan-y data
        '''
        start, end = self.largest_non_nan_chunk() # get the first and last indicies of largest non-NaN block
        self.dat = self.dat[start:end, :]
        print("DEBUG KillNaN2: ", [start, end])
        if start == 0 and end == len(self.dat[:,0]):
            self.check3 = 0
        else:
            self.check3 = 1
        return [1, end-start+1]




    def simple_ringdown(self):
        '''
        Function to assemble all simple ringdown data appropriately and do C-avg-ing.
        '''

        #sort out remnant bugs in data
        self.tempS = np.where((np.diff(self.TTL2) == -1)*1 == 1)[0] #Indicies of TTL2 lowering edgge
        self.tempS2 = self.ift(self.time, self.tfi(self.time, self.tempS, self.minv, self.maxv)-self.k1, self.minv, self.maxv) # is self.minv the new zero?
        self.starterS = self.tempS2[self.tempS2 > 0] # so that we only count indices after self.minv?
        self.casesS = np.where(self.tempS2 < 0)[0] 
        
        #markers for simple ringdowns
        self.tempS3 = np.where((np.diff(self.TTL2) == -1)*1 == 1)[0] + self.minv -1 -1
        self.iloopstartS = np.delete(self.tempS3, self.casesS)
        self.tloopstartS = self.tfi(self.time, self.iloopstartS, self.minv, self.maxv) + self.t_wait
        self.tloopendS = self.tloopstartS + self.setpreptime
        self.tLIAstartS = self.tloopendS
        self.tLIAendS = self.tLIAstartS + self.tLIA
        self.t1stackS = self.tfi(self.time, self.starterS, self.minv, self.maxv)
        self.i1stackS = self.starterS
        self.tDAqendS = self.tloopstartS + self.setlooptime + self.setresttime + self.setpreptime
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
        
        self.elements = [i for i in range(1, self.looplen)]
        for e in self.elements:
            self.stackS[e,:,2] = self.initstackS[self.i1stackS[e]:self.i2stackS[e] + 1, 2]
            self.stackS[e,:,3] = self.initstackS[self.i1stackS[e]:self.i2stackS[e] + 1, 3]
        for e in self.elements:
            self.stackS[e,:,2] = self.stackS[e,:,2]/(np.exp(+1j*np.angle(self.stackS[e,self.ift(self.time, self.k1 - self.time[0] + self.shift, self.minv, self.maxv),2])))
            self.stackS[e,:,3] = self.stackS[e,:,3]/(np.exp(+1j*np.angle(self.stackS[e,self.ift(self.time, self.k1 - self.time[0] + self.shift, self.minv, self.maxv),3])))
            self.phasestackS[e,:,1] = np.unwrap(np.angle(self.stackS[e,:,2]))
            self.phasestackS[e,:,1] = (self.phasestackS[e,:,1] - self.phasestackS[e,self.ift(self.time, self.k1 - self.time[0] + self.shift, self.minv, self.maxv) - self.minv + 1 + 1,1])/(2*np.pi)
            self.phasestackS[e,:,2] = np.unwrap(np.angle(self.stackS[e,:,3]))
            self.phasestackS[e,:,2] = (self.phasestackS[e,:,2] - self.phasestackS[e,self.ift(self.time, self.k1 - self.time[0] + self.shift, self.minv, self.maxv) - self.minv + 1 + 1,2])/(2*np.pi)
        
        #do C-avg
        self.meanmode1S = (np.sum(self.stackS[:,:,2], axis = 0))/(self.looplen - 1)
        self.absmode1S = np.abs(self.meanmode1S)
        self.argmode1S = np.unwrap(np.angle(self.meanmode1S))
        self.argmode1S = self.argmode1S - self.argmode1S[self.ift(self.time, self.k1 - self.time[0], self.minv, self.maxv) - self.minv + 1 + 1]
        self.meanmode2S = (np.sum(self.stackS[:,:,3], axis = 0))/(self.looplen - 1)
        self.absmode2S = np.abs(self.meanmode2S)
        self.argmode2S = np.unwrap(np.angle(self.meanmode2S))
        self.argmode2S = self.argmode2S - self.argmode2S[self.ift(self.time, self.k1 - self.time[0] + self.shift, self.minv, self.maxv) - self.minv + 1 + 1]
        
        temp = np.stack((np.abs(self.stackS[0,:,0]), np.abs(self.stackS[0,:,1]), self.absmode1S, self.argmode1S/ (2*np.pi), self.absmode2S, self.argmode2S/(2*np.pi)),axis = -1)
        return self.stackS, self.phasestackS, temp, self.elements
    


    def loop_ringdown(self):
        '''
        Function to assemble all simple ringdown data appropriately and do C-avg-ing.
        '''

        #sort out remnant bugs in data
        self.tempC = np.where((np.diff(self.TTL1) == -1)*1 == 1)[0]
        self.tempC2 = self.ift(self.time, self.tfi(self.time, self.tempC, self.minv, self.maxv)-self.k1, self.minv, self.maxv)
        self.starterC = self.tempC2[self.tempC2 > 0]
        self.casesC = np.where(self.tempC2 < 0)[0] 
        
        #markers for loop ringdowns
        self.tempC3 = np.where((np.diff(self.TTL1) == -1)*1 == 1)[0] + self.minv -1 -1
        self.iloopstartC = np.delete(self.tempC3, self.casesC)
        self.tloopstartC = self.tfi(self.time, self.iloopstartC, self.minv, self.maxv) + self.t_wait
        self.tloopendC = self.tloopstartC + self.setlooptime + self.setpreptime
        self.tLIAstartC = self.tloopendC
        self.tLIAendC = self.tLIAstartC + self.tLIA
        self.t1stackC = self.tfi(self.time, self.starterC, self.minv, self.maxv)
        self.i1stackC = self.starterC
        self.tDAqendC = self.tloopstartC + self.setlooptime + self.setresttime + self.setpreptime
        self.t2stackC = self.tDAqendC - self.k2
        self.i2stackC = self.ift(self.time, self.t2stackC, self.minv, self.maxv)
        #to avoid recent C-avg-ing BS
        self.i2stackC = self.i2stackC[self.i2stackC > self.i1stackC[0]]
        self.i2stackC = self.i2stackC[0:len(self.i1stackC)]

        self.devC = np.std(self.i2stackC - self.i1stackC)
        self.looplenC = len(self.i2stackC) - 3 # do not touch this line
        
        
   
        #stacking loop ringdowns
        self.initstackC = np.stack((self.time, self.TTL2, self.C2, self.C5),axis = -1)
        self.stackC = np.zeros((self.looplenC, (self.i2stackC - self.i1stackC)[0] + 1, 4), dtype = 'complex_')
        self.phasestackC = np.zeros((self.looplenC, (self.i2stackC - self.i1stackC)[0] + 1, 3))
        
        self.stackC[:,:,0] = self.initstackC[self.i1stackC[1]:self.i2stackC[1] + 1, 0]
        self.stackC[:,:,1] = self.initstackC[self.i1stackC[1]:self.i2stackC[1] + 1, 1]
        self.phasestackC[:,:,0] = np.real(self.initstackC[self.i1stackC[0]:self.i2stackC[0] + 1, 0])
        
        self.elements = [i for i in range(1, self.looplenC)]
        for e in self.elements:
            self.stackC[e,:,2] = self.initstackC[self.i1stackC[e+1]:self.i2stackC[e+1] + 1, 2]
            self.stackC[e,:,3] = self.initstackC[self.i1stackC[e+1]:self.i2stackC[e+1] + 1, 3]
        for e in self.elements:
            self.stackC[e,:,2] = self.stackC[e,:,2]/(np.exp(+1j*np.angle(self.stackC[e,self.ift(self.time, self.k1 - self.time[0] + self.shift, self.minv, self.maxv),2])))
            self.stackC[e,:,3] = self.stackC[e,:,3]/(np.exp(+1j*np.angle(self.stackC[e,self.ift(self.time, self.k1 - self.time[0] + self.shift, self.minv, self.maxv),3])))
            self.phasestackC[e,:,1] = np.unwrap(np.angle(self.stackC[e,:,2]))
            self.phasestackC[e,:,1] = (self.phasestackC[e,:,1] - self.phasestackC[e,self.ift(self.time, self.k1 - self.time[0] + self.shift, self.minv, self.maxv) - self.minv + 1 + 1,1])/(2*np.pi)
            self.phasestackC[e,:,2] = np.unwrap(np.angle(self.stackC[e,:,3]))
            self.phasestackC[e,:,2] = (self.phasestackC[e,:,2] - self.phasestackC[e,self.ift(self.time, self.k1 - self.time[0] + self.shift, self.minv, self.maxv) - self.minv + 1 + 1,2])/(2*np.pi)
        
        #do C-avg
        self.meanmode1C = (np.sum(self.stackC[:,:,2], axis = 0))/(self.looplenC - 1)
        self.absmode1C = np.abs(self.meanmode1C)
        self.argmode1C = np.unwrap(np.angle(self.meanmode1C))
        self.argmode1C = self.argmode1C - self.argmode1C[self.ift(self.time, self.k1 - self.time[0], self.minv, self.maxv) - self.minv + 1 + 1]
        self.meanmode2C = (np.sum(self.stackC[:,:,3], axis = 0))/(self.looplenC - 1)
        self.absmode2C = np.abs(self.meanmode2C)
        self.argmode2C = np.unwrap(np.angle(self.meanmode2C))
        self.argmode2C = self.argmode2C - self.argmode2C[self.ift(self.time, self.k1 - self.time[0] + self.shift, self.minv, self.maxv) - self.minv + 1 + 1]
        temp = np.stack((np.abs(self.stackC[0,:,0]), np.abs(self.stackC[0,:,1]), self.absmode1C, self.argmode1C/ (2*np.pi), self.absmode2C, self.argmode2C/(2*np.pi)),axis = -1)
        return self.stackC, self.phasestackC, temp, self.elements
    
     
    
    def return_Cavg_data(self, filename, savefolder, save = True):
        '''
        Calls simple_ringdown and loop_ringdown functions and collates both (simple and loop ringdowns) of the C-avg-ed data in an array. 
        This array is supplemented with relevant headers and saved.
        '''

        stackS, phasestackS, tempS, elementsS = self.simple_ringdown()
        stackC, phasestackC, tempC, elementsC = self.loop_ringdown()
        temptot = np.stack((tempS[:,0], tempS[:,1], tempS[:,2], tempS[:,3], tempS[:,4], tempS[:,5], 
                            tempC[:,0], tempC[:,1], tempC[:,2], tempC[:,3], tempC[:,4], tempC[:,5]),axis = -1)
        
        l1 = "demod2 (Hz) = {}".format(self.demod2f)+"\n"
        l2 = "demod5 (Hz) = {}".format(self.demod5f)+"\n" 
        l3 = "loop time (s) = {}".format(self.setlooptime)+"\n"
        l3a = "prep time (s) = {}".format(self.setpreptime)+"\n"
        l4 = "drive time (s) = {}".format(self.setdrivetime)+"\n" 
        l5 = "rest time (s) = {}".format(self.setresttime)+"\n"
        l6 = "LIA filter order = {}".format(int(self.order))+"\n"
        l7 = "LIA BW (Hz) = {}".format(self.bandwidth)+"\n"
        l8 = "LIA response time (s) = {}".format(self.tLIA)+"\n"
        l9 = "check 1 = {}".format(self.devS + self.devC)+"\n"
        l10 = "check 2 = {}".format(((len(self.casesS)>1)*1 + (len(self.casesC)>1)*1))+"\n"
        l10a = "check 3 = {}".format(self.check3)+"\n"
        l11 = "# valid loops = {}".format(self.looplen)+"\n"
        l11a = "t_wait (s) = {}".format(self.t_wait)+"\n"
        l12 = "time (simple, s), TTL (simple) , abs demod 2 (simple, mV), arg demod 2 (simple, 2pi*rad), abs demod 5 (simple, mV), arg demod 5 (simple. 2pi*rad)"
        l13 = "time (cloop, s), TTL (cloop) , abs demod 2 (cloop, mV), arg demod 2 (cloop, 2pi*rad), abs demod 5 (cloop, mV), arg demod 5 (cloop. 2pi*rad)"
        preamble = l1+l2+l3+l3a+l4+l5+l6+l7+l8+l9+l10+l10a+l11+l11a+l12+l13
        
        if save == True:
            save_filename, _ = file_handling.make_filename(filename, savefolder, 'csv')
            file_handling.savetxtdate(save_filename, temptot, delimiter = ',', header = preamble)
        
        #return temptot
        return
    


    def plot_all(self, filename, savefolder, plot = False, save = True):
        '''
        Plots everything in a "picture". A failed bid to reduce time taken for each iteration.
        '''
        
        #import all data
        stackS, phasestackS, tempS, elementsS = self.simple_ringdown()
        stackC, phasestackC, tempC, elementsC = self.loop_ringdown()
        
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
        f = plt.figure(figsize=(11,10),facecolor='white')
        gs0 = gridspec.GridSpec(2, 2, wspace = 0.125, hspace = 0.2, width_ratios=[1, 1])
        

        gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0,:-1], hspace=0)

        #demod 2 abs simple ringdown
        ax0 = f.add_subplot(gs00[0])
        ax0.tick_params(axis='both', which='major', direction='in', labelsize=fsz, size=3, width=wid, top='on', pad = 5, right='on')
        for e in elementsS:
             ax0.plot(tempS[:,0], np.abs(stackS[e,:,2]), color='darkturquoise', linewidth = thwid, alpha = 0.1)
        ax0.plot(tempS[:,0], tempS[:,2], color='teal', linewidth = plwid, label = 'demod 2 data')
        ax0.axvspan(self.tLIAstartS[0], self.tLIAendS[0], facecolor='#B0E2FF', alpha=0.45, zorder=2.5, label = 'LIA window')
        ax0.axvspan(self.tloopstartS[0], self.tloopendS[0], facecolor='lemonchiffon', alpha=0.45, zorder=2.5, label = 'prep window')
        ax0.autoscale(enable=True, axis='both', tight=False)
        ax0.grid(True, color = 'lightgray', linewidth = lwid)
        ax0.set_yscale("log")
        ax0.set_ylabel("Abs[response] (mV)", labelpad=8, fontsize = fsz+2)
        ax0.legend(fontsize = fsz, loc = 'best', frameon = True)
        ax0.set_title("prep ringdown", fontsize = fsz+2)
        ax0.minorticks_off()
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax0.margins(0.025, 0.05)
        
        #demod 2 arg simple ringdown 
        ax1 = f.add_subplot(gs00[1], sharex=ax0)
        ax1.tick_params(axis='both', which='major', direction='in', labelsize=fsz, size=3, width=wid, top='on', pad = 5, right='on')
        for e in elementsS:
            ax1.plot(tempS[:,0], phasestackS[e,:,1], color='darkturquoise', linewidth = thwid, alpha = 0.1)
        ax1.plot(tempS[:,0], tempS[:,3], color='teal', linewidth = plwid)
        ax1.axvspan(self.tLIAstartS[0], self.tLIAendS[0], facecolor='#B0E2FF', alpha=0.45, zorder=2.5)
        ax1.axvspan(self.tloopstartS[0], self.tloopendS[0], facecolor='lemonchiffon', alpha=0.45, zorder=2.5)
        ax1.autoscale(enable=True, axis='both', tight=False)
        ax1.grid(True, color = 'lightgray', linewidth = lwid)
        ax1.set_ylabel("Arg[response] (2$\pi$)", labelpad=8, fontsize = fsz+2)
        ax1.set_xlabel("time (s)", labelpad=8, fontsize = fsz+2)
        ax1.margins(0.025, 0.05)
    
        
        
        gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1,:-1], hspace=0)
        #demod 2 abs loop ringdown
        ax2 = f.add_subplot(gs01[0])
        ax2.tick_params(axis='both', which='major', direction='in', labelsize=fsz, size=3, width=wid, top='on', pad = 5, right='on')
        for e in elementsC:
            ax2.plot(tempC[:,0], np.abs(stackC[e,:,2]), color='darkturquoise', linewidth = thwid, alpha = 0.1)
        ax2.plot(tempC[:,0], tempC[:,2], color='teal', linewidth = plwid)
        ax2.axvspan(self.tLIAstartC[1], self.tLIAendC[1], facecolor='#B0E2FF', alpha=0.45, zorder=2.5)
        ax2.axvspan(self.tloopstartC[1] + self.setpreptime, self.tloopendC[1] , facecolor='lightgray', alpha=0.45, zorder=2.5, label = 'loop window')
        ax2.axvspan(self.tloopstartC[1] , self.tloopstartC[1] + self.setpreptime, facecolor='lemonchiffon', alpha=0.45, zorder=2.5)
        ax2.autoscale(enable=True, axis='both', tight=False)
        ax2.grid(True, color = 'lightgray', linewidth = lwid)
        ax2.set_yscale("log")
        ax2.set_ylabel("Abs[response] (mV)", labelpad=8, fontsize = fsz+2)
        ax2.legend(fontsize = fsz, loc = 'best', frameon = True)
        ax2.set_title("prep and loop ringdown", fontsize = fsz+2)
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.minorticks_off()
        ax2.margins(0.025, 0.05)
        
        #demod 2 arg loop ringdown
        ax3 = f.add_subplot(gs01[1], sharex=ax2)
        ax3.tick_params(axis='both', which='major', direction='in', labelsize=fsz, size=3, width=wid, top='on', pad = 5, right='on')
        for e in elementsC:
            ax3.plot(tempC[:,0], phasestackC[e,:,1], color='darkturquoise', linewidth = thwid, alpha = 0.1)
        ax3.plot(tempC[:,0], tempC[:,3], color='teal', linewidth = plwid)
        ax3.axvspan(self.tLIAstartC[1], self.tLIAendC[1], facecolor='#B0E2FF', alpha=0.45, zorder=2.5)
        ax3.axvspan(self.tloopstartC[1] + self.setpreptime, self.tloopendC[1] , facecolor='lightgray', alpha=0.45, zorder=2.5)
        ax3.axvspan(self.tloopstartC[1] , self.tloopstartC[1] + self.setpreptime, facecolor='lemonchiffon', alpha=0.45, zorder=2.5)
        ax3.autoscale(enable=True, axis='both', tight=False)
        ax3.grid(True, color = 'lightgray', linewidth = lwid)
        ax3.set_ylabel("Arg[response] (2$\pi$)", labelpad=8, fontsize = fsz+2)
        ax3.set_xlabel("time (s)", labelpad=8, fontsize = fsz+2)
        ax3.margins(0.025, 0.05)
        
        
        
        gs02 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[:1, -1], hspace=0)
        #demod 5 abs simple ringdown
        ax4 = f.add_subplot(gs02[0])
        ax4.tick_params(axis='both', which='major', direction='in', labelsize=fsz, size=3, width=wid, top='on', pad = 5, right='on')
        for e in elementsS:
            ax4.plot(tempS[:,0], np.abs(stackS[e,:,3]), color='darkorange', linewidth = thwid, alpha = 0.1)
        ax4.plot(tempS[:,0], tempS[:,4], color='sienna', linewidth = plwid, label = 'demod 5 data')
        ax4.axvspan(self.tLIAstartS[0], self.tLIAendS[0], facecolor='#B0E2FF', alpha=0.45, zorder=2.5)
        ax4.axvspan(self.tloopstartS[0], self.tloopendS[0], facecolor='lemonchiffon', alpha=0.45, zorder=2.5)
        ax4.autoscale(enable=True, axis='both', tight=False)
        ax4.grid(True, color = 'lightgray', linewidth = lwid)
        ax4.legend(fontsize = fsz, loc = 'best', frameon = True)
        ax4.set_yscale("log")
        ax4.minorticks_off()
        plt.setp(ax4.get_xticklabels(), visible=False)
        ax4.margins(0.025, 0.05)
        
        #demod 5 arg simple ringdown
        ax5 = f.add_subplot(gs02[1], sharex=ax4)
        ax5.tick_params(axis='both', which='major', direction='in', labelsize=fsz, size=3, width=wid, top='on', pad = 5, right='on')
        for e in elementsS:
            ax5.plot(tempS[:,0], phasestackS[e,:,2], color='darkorange', linewidth = thwid, alpha = 0.1)
        ax5.plot(tempS[:,0], tempS[:,5], color='sienna', linewidth = plwid)
        ax5.axvspan(self.tLIAstartS[0], self.tLIAendS[0], facecolor='#B0E2FF', alpha=0.45, zorder=2.5)
        ax5.axvspan(self.tloopstartS[0], self.tloopendS[0], facecolor='lemonchiffon', alpha=0.45, zorder=2.5)
        ax5.autoscale(enable=True, axis='both', tight=False)
        ax5.grid(True, color = 'lightgray', linewidth = lwid)
        ax5.set_xlabel("time (s)", labelpad=8, fontsize = fsz+2)
        ax5.margins(0.025, 0.05)
        
        
        
        gs03 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1:, -1], hspace=0)
        #demod 5 abs loop ringdown
        ax6 = f.add_subplot(gs03[0])
        ax6.tick_params(axis='both', which='major', direction='in', labelsize=fsz, size=3, width=wid, top='on', pad = 5, right='on')
        for e in elementsC:
            ax6.plot(tempC[:,0], np.abs(stackC[e,:,3]), color='darkorange', linewidth = thwid, alpha = 0.1)
        ax6.plot(tempC[:,0], tempC[:,4], color='sienna', linewidth = plwid)
        ax6.axvspan(self.tLIAstartC[1], self.tLIAendC[1], facecolor='#B0E2FF', alpha=0.45, zorder=2.5)
        ax6.axvspan(self.tloopstartC[1] + self.setpreptime, self.tloopendC[1] , facecolor='lightgray', alpha=0.45, zorder=2.5)
        ax6.axvspan(self.tloopstartC[1] , self.tloopstartC[1] + self.setpreptime, facecolor='lemonchiffon', alpha=0.45, zorder=2.5)
        ax6.autoscale(enable=True, axis='both', tight=False)
        ax6.grid(True, color = 'lightgray', linewidth = lwid)
        ax6.set_yscale("log")
        ax6.minorticks_off()
        plt.setp(ax6.get_xticklabels(), visible=False)
        ax6.margins(0.025, 0.05)
        
        #demod 5 arg loop ringdown
        ax7 = f.add_subplot(gs03[1], sharex=ax6)
        ax7.tick_params(axis='both', which='major', direction='in', labelsize=fsz, size=3, width=wid, top='on', pad = 5, right='on')
        for e in elementsC:
            ax7.plot(tempC[:,0], phasestackC[e,:,2], color='darkorange', linewidth = thwid, alpha = 0.1)
        ax7.plot(tempC[:,0], tempC[:,5], color='sienna', linewidth = plwid)
        ax7.axvspan(self.tLIAstartC[1], self.tLIAendC[1], facecolor='#B0E2FF', alpha=0.45, zorder=2.5)
        ax7.axvspan(self.tloopstartC[1] + self.setpreptime, self.tloopendC[1] , facecolor='lightgray', alpha=0.45, zorder=2.5)
        ax7.axvspan(self.tloopstartC[1] , self.tloopstartC[1] + self.setpreptime, facecolor='lemonchiffon', alpha=0.45, zorder=2.5)
        ax7.autoscale(enable=True, axis='both', tight=False)
        ax7.grid(True, color = 'lightgray', linewidth = lwid)
        ax7.set_xlabel("time (s)", labelpad=8, fontsize = fsz+2)
        ax7.margins(0.025, 0.05)
        
        f.suptitle("$T_{loop}$ = " + str(f'{self.setlooptime*1e3:.3f}') + " ms", size=fsz+3)
        gs0.tight_layout(f)
        
        if save == True:
            plotname, _ = file_handling.make_plotfilename(filename,savefolder)
            plt.savefig(plotname, dpi = 400,bbox_inches='tight')
        
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

    # def killnans(self):
    #     '''
    #     OG Function to remove nan-s and shenanigans.
    #     It failed when I had nan chunks in the middle of valid data...
    #     '''

    #     target = np.arange(len(self.dat[:,0]))
    #     nanrows1 = np.unique(np.array([i[0] for i in np.argwhere(np.isnan(self.dat))]))
    #     nanrows2 = np.where(self.dat[:,0] > self.meastimeproxy)
    #     nanrows = np.union1d(nanrows1, nanrows2)
    #     left = target
    #     self.check3 = 0
    #     if len(nanrows) > 0:
    #         left = np.delete(target, nanrows)
    #         print ('beep!')
    #         self.check3 = 1
    #         newdat = np.delete(self.dat, nanrows, axis = 0)
    #         self.dat = newdat
    #         del newdat
    #     print("DEBUG: ", [np.min(left)+1, np.max(left)+1])
    #     return [np.min(left)+1, np.max(left)+1]
    #     #return nanrows1, nanrows2, nanrows