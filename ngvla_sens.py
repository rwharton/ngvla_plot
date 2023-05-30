from __future__ import division
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import PathPatch
from collections import namedtuple
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'DeJavu Serif',
              'serif':['Computer Modern Roman']})
rc('text', usetex=True)

catfile = 'psrcat.txt'

def gaussian(t, W, P, norm=True):
    aa = 4.0 * np.log(2) / W**2.0
    if norm:
        gg = np.sqrt(aa/np.pi) * np.exp(-1.0 * aa * (t-0.5*P)**2)
    else:
        gg = np.exp(-1.0 * aa * (t-0.5*P)**2)
    return gg

def gauss_ft(ff, W, norm=True):
    aa = 4.0 * np.log(2) / W**2.0
    if norm:
        gg = np.exp(-1.0 * (np.pi * ff)**2.0 / aa)
    else:
        gg = np.sqrt(aa/np.pi) * np.exp(-1.0 * (np.pi * ff)**2.0 / aa)
    return gg
        
def scat_ft(ff, tsc):
    return 1.0 / (1 + 2j * np.pi * ff * tsc)
    
def fft_conv(a, b):
    cc = np.fft.irfft( np.fft.rfft(a) * np.fft.rfft(b) )
    cc /= np.sum(b)
    return cc

def get_hsig(pp):
    nhs = np.arange(len(pp))
    return nhs[1:], np.cumsum(pp[1:]) / np.sqrt(nhs[1:])
    #return nhs[1:], np.cumsum(pp[1:]) / nhs[1:]

def max_nh(nhs, hsig):
    xx = np.where(hsig==max(hsig))[0][-1]
    return nhs[xx], hsig[xx]

def get_closest(a, aval):
    xx = np.where( np.abs(a-aval) == min(np.abs(a-aval)) )[0]
    return xx[0]

def boxcar_ft(ff, dt):
    return np.sinc(ff*dt)

def dm_delay(fc, df, DM):
    """
    Calculate dispersive delay across bandwidth
    of 'df' centered on frequency 'fc' (in  MHz)
    at DM of 'DM'.  

    Returns delay in ms.
    """
    fl = fc - 0.5 * df
    fh = fc + 0.5 * df
    dt = 4148808.0 * (fl**-2.0 - fh**-2.0) * DM
    return dt

def Smin_vla(tmins, smin1=0.400):
    return smin1 / np.sqrt(tmins)

def get_maxD(xs, Smin1, hs, Lmin):
    xx = np.where( Smin1*xs**2.0 / hs <= Lmin )[0]
    if len(xx):
        lastx = xx[-1]
    else:
        lastx = 0
    return xs[lastx]

def get_maxDs(pers, tsc, dms, harms, xs, Lmin, tobs, Smin1, fc, df, dt, dc=0.05, hg=False):
    Dlist = []
    nhlist = []
    hslist = []
    Smin1 = Smin1 / np.sqrt(tobs / 60.0)
    if hg:
        Smin1 = 2.0 * Smin1
    for P in pers:
        W = dc * P 
        Pf = gauss_ft(harms/P, W)

        hmat = np.array( list(harms/P) * len(tsc) )
        hmat.shape = (len(tsc), len(harms))
        Hfmat = scat_ft(hmat.T, tsc)
        #Hfmat = gauss_ft(hmat.T, tsc)

        tdms = dm_delay(fc, df, dms)
        Dmf  = boxcar_ft(hmat.T, tdms)

        Dtf = boxcar_ft(harms/P, dt)

        Fprod = np.abs(Hfmat.T * Pf * Dtf * Dmf.T)
        hsig = np.cumsum(Fprod, axis=1) / np.sqrt(harms)
        
        nh_hs = np.array([max_nh(harms, hh) for hh in hsig])
        
        Dlist.append(get_maxD(xs, Smin1, nh_hs[:,1], Lmin))
        nhlist.append( max(nh_hs[:,0]) )
        hslist.append( max(nh_hs[:,1]) )
    return np.array(Dlist), np.array(nhlist), np.array(hslist)

def get_smin(pers, DM, tsc, harms, Smin1, fc, df, dt, dc=0.05):
    smin_list = []
    nh_list = []
    hsig_list = []
    for P in pers:
         W = dc * P
         Pf = gauss_ft(harms/P, W)

         hms = harms / P
         ## New ##
         fmax = 1.0 / (2 * dt)
         xx = np.where(hms > fmax)[0]

         Hf = scat_ft(hms, tsc)
         
         tdm = dm_delay(fc, df, DM)
         Dmf  = boxcar_ft(hms, tdm)

         #Dtf = boxcar_ft(harms/P, dt)
         Dtf = boxcar_ft(hms, dt)
         
         Fprod = np.abs(Hf * Pf * Dtf * Dmf)
         if len(xx):
             Fprod[xx] = 0
         hsig = np.cumsum(Fprod) / np.sqrt(harms)
         #hsig = np.cumsum(Fprod) / harms 
         nh_hs = max_nh(harms, hsig)
         
         nh_list.append(nh_hs[0])
         hsig_list.append(nh_hs[1])

    hsigs = np.array(hsig_list)
    nhs = np.array(nh_list)
    smins = Smin1 / hsigs
    #smins = Smin1 / np.sqrt(hsigs)
    return smins, hsigs, nhs 

def ret_exists(valstr):
    if valstr == '-999':
        return None
    else:
        return float(valstr)

def get_gain(Nant, ap_eff):
    return 0.178 * Nant * ap_eff

def get_gain2(Nant, ap_eff):
    return 0.178 * ap_eff * np.sqrt(Nant * (Nant - 1.0))

def get_sefd(band, Nant):
    tsys = {'L':26, 'S':(29 + 70.), 'C':31, 'X':34, 'Ku':39, 'K':54}
    ap_eff = {'L':0.45, 'S':0.62, 'C':0.60, 'X':0.56, 'Ku':0.54, 'K':0.51}
    return tsys[band] / get_gain2(Nant, ap_eff[band])

def get_smin1(band, Nant, bw, tint, snr=10.0, npol=2.0):
    """
    Get smin1, return in mJy

    tint in sec, bw in MHz
    """
    smin1 = snr * get_sefd(band, Nant) / np.sqrt(npol * tint * bw)
    return smin1

def get_gbt_smin1(tsys, G, bw, tint, snr=10.0, npol=2.0):
    smin1 = snr * tsys / (G * np.sqrt(npol * tint * bw))
    return smin1

def get_Lmin(Smin1, fc, pers, DM, tsc1, harms=np.arange(1, 33, 1), 
             df=0.1, dt=0.064, dc=0.10, dgc=8.5, f0=1400.0):
    tsc = tsc1 * (fc/1000.0)**-4.0
    smins, hsigs, nhs = get_smin(pers, DM, tsc, harms, Smin1, fc, df, dt, dc)
    lmins = smins * dgc**2.0 * (fc / f0)**1.6
    return lmins

def easy_get_Lmin(pars, Nant, tint, pers, DM, tsc1, fac=1.0, 
                  snr=10.0, npol=2.0, f0=1400.0, dc=0.05, dgc=8.5):
    smin1 = get_smin1(pars['band'], Nant, pars['bw'], tint, snr=snr, npol=npol)  
    lmins = get_Lmin(smin1 / fac, fc=pars['fc'], pers=pers, DM=DM, tsc1=tsc1,
                      df=pars['df'], dt=pars['dt'], dc=dc, dgc=dgc, f0=f0)
    return lmins

def get_psrlist(infile=catfile):
    Pulsar = namedtuple('PSR', 'name ra dec p pdot dm s1400 spindex rlum1400')
    psrlist = []
    for line in open(infile, 'r'):
        cols = line.split()
        if len(cols) != 10:
            continue
        else:
            psrlist.append( Pulsar(name=cols[1],
                                   ra = ret_exists(cols[2]),
                                   dec = ret_exists(cols[3]),
                                   p = ret_exists(cols[4]),
                                   pdot = ret_exists(cols[5]),
                                   dm = ret_exists(cols[6]),
                                   s1400 = ret_exists(cols[7]),
                                   spindex = ret_exists(cols[8]),
                                   rlum1400 = ret_exists(cols[9]) ))
    return psrlist

def get_det_psr(psrP, psrL, surveyP, surveyL):
    """
    return the number of pulsars with psrL > surveyL
    """
    surveyL_interp = np.interp(psrP, surveyP, surveyL)
    xx = np.where( psrL > surveyL_interp )[0]
    ndet = len(xx)
    npsr = len(psrP)
    fdet = ndet / float(npsr)
    return ndet, npsr, fdet

def get_det_psr_Pcut(psrP, psrL, surveyP, surveyL, Pcut, cut='>'):
    if cut == ">":
        psrP_p = psrP[psrP > Pcut]
        psrL_p = psrL[psrP > Pcut]
        surveyP_p = surveyP[surveyP > Pcut]
        surveyL_p = surveyL[surveyP > Pcut]
    else:
        psrP_p = psrP[psrP < Pcut]
        psrL_p = psrL[psrP < Pcut]
        surveyP_p = surveyP[surveyP < Pcut]
        surveyL_p = surveyL[surveyP < Pcut]
    
    surveyL_interp = np.interp(psrP_p, surveyP_p, surveyL_p)
    xx = np.where( psrL_p > surveyL_interp )[0]
    ndet = len(xx)
    npsr = len(psrP_p)
    fdet = ndet / float(npsr)
    return ndet, npsr, fdet

def det_psr_summary(psrP, psrL, surveyP, surveyL, Pcut):
    # ALL PULSARS
    ndet, npsr, fdet = get_det_psr(psrP, psrL, surveyP, surveyL)
    print("  ALL PSR  %d/%d (%.3f)" %(ndet, npsr, fdet)) 

    # JUST CP
    ndet_cp, npsr_cp, fdet_cp = get_det_psr_Pcut(psrP, psrL, surveyP, surveyL, 
                                                 Pcut, cut='>')
    print("  P>%d ms  %d/%d (%.3f)" %(Pcut * 1e3, ndet_cp, npsr_cp, fdet_cp)) 
    
    # JUST MSP
    ndet_msp, npsr_msp, fdet_msp = get_det_psr_Pcut(psrP, psrL, surveyP, surveyL, 
                                                    Pcut, cut='<')
    print("  P<%d ms  %d/%d (%.3f)" %(Pcut * 1e3, ndet_msp, npsr_msp, fdet_msp)) 
    return

def get_dt_string(dt):
    if dt < 0.1:
        return r"\delta t = %.0f\,\mu\rm s" %(dt * 1000.0)
    else:
        return r"\delta t = %.1f\,\rm ms" %dt

def myLogFormat(y,pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


def get_vla_label(par, tint):
    vlabel=r"VLA: $\nu=%.1f \/ {\rm GHz}, \/ \Delta \nu = %.1f \/{\rm GHz}, \/ T = %.1f\/{\rm hr}, \/ %s$"\
           %(par['fc']/1e3, par['bw']/1e3, tint/3600.,  get_dt_string(par['dt']))
    return vlabel


def get_label(par, tint):
    vlabel=r"$\nu=%.1f \/ {\rm GHz}, \/ \Delta \nu = %.1f \/{\rm GHz}, \/ T = %.1f\/{\rm hr}$"\
           %(par['fc']/1e3, par['bw']/1e3, tint/3600.)
    return vlabel


def make_plot(parlist):
    psrlist = get_psrlist()
    psrP = np.array([ ii.p for ii in psrlist if ii.p and ii.rlum1400 ])
    psrL = np.array([ ii.rlum1400 for ii in psrlist if ii.p and ii.rlum1400 ])

    tint = 6.0 * 3600.0
    pers = np.logspace(-1, 4, 1000)
    DM = 1778.0
    Nant = 25.0

    tscA = 1e3 * 1000    # 1 GHz scattering time (ms)
    tscB = 1.3 * 1000.0   # 1 GHz scattering time (ms)
    Pcut = 30e-3 # sec (boundary for ~MSPs)

    fig = plt.figure(figsize=(8,10))
    ax = fig.add_subplot(111)

    fsize = 24
    tfsize = 20

    ###############################
    ###        PULSARS          ###
    ###############################
    plt.scatter(psrP, psrL, c='0.6', s=10, edgecolors='none', zorder=0)

    ###############################
    ###  Sensitivity (x VLA)    ###
    ###############################
    #faclist = [1, 3, 10]
    faclist = [10]

    ###############################
    ###     LINE PROPERTIES     ###
    ###############################
    lw_list = np.linspace(3, 5, len(parlist))
    ls_list = ['--', '-']
    hatch_list = ['||', '--', '//']
    cvals = [0.25, 0.5, 0.75]

    for ii, par in enumerate(parlist):
        for jj, fac in enumerate(faclist): 
            lminA = easy_get_Lmin(par, Nant, tint, pers, DM, tscA, fac=fac)
            lminB = easy_get_Lmin(par, Nant, tint, pers, DM, tscB, fac=fac)

            lw  = lw_list[ii] 
            ls  = "-" #ls_list[jj % len(ls_list)]
            hh = hatch_list[ii]
            if jj == len(faclist)-1:
                figlab = get_label(par, tint)
            else:
                figlab = None
    
            col = plt.cm.magma(cvals[ii])

            ax.plot(pers*0.001, lminA, ls='-', c=col, lw=lw, zorder=100+ii,
                    label=figlab)
            ax.plot(pers*0.001, lminB, ls='--', c=col, lw=lw, zorder=100+ii)
    
            #ax.fill_between(pers*0.001, lminA, lminB, edgecolor=col, 
            #                facecolor='none', hatch=hh)
            
            #ax.fill_between(pers*0.001, lminA, lminB, color=col, 
            #                alpha=0.3, zorder=100+ii)

            print("%s (VLA x %d): " %(par['band'], fac))
        
            print(" HIGH SCATTER (%.1f s at 1 GHz):" %(tscA/1e3))
            det_psr_summary(psrP, psrL, pers*1e-3, lminA, Pcut)
        
            print(" MAG SCATTER (%.1f s at 1 GHz):" %(tscB/1e3))
            det_psr_summary(psrP, psrL, pers*1e-3, lminB, Pcut)

            print("\n")

    # Plot Pulsars w/ pb factor
    p_names = np.array([ "1745-2910", "1745-2912", "1746-2849", "1746-2850", "1746-2856"])
    p_kpsr = np.array([ 0.982, 0.187, 1.48, 1.08, 0.945 ])
    spb_kpsr = np.array([ 25, 24, 20, 70, 61 ]) * 1e-3
    pb_kpsr = np.array([ 0.25, 0.20, 0.20, 0.23, 0.08 ])
    s_kpsr = spb_kpsr / pb_kpsr
    ax.plot(p_kpsr, s_kpsr * (1.4 / 3.0)**-1.6 * 8.5**2.0, marker='s', c='r', ls='', mec='r', ms=8, zorder=200)

    ax.plot(3.76, 600.0, marker='d', c='r', ls='', mec='r', ms=8, zorder=200)

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    xt = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    ax.set_xticks(xt)
    
    ax.set_xlim(8e-4, 10)
    ax.set_ylim(1e-2, 5000)
    ax.set_xlabel("Spin Period (s)", fontsize=fsize)
    ax.set_ylabel("1400 MHz Pseudo-Luminosity " + r"$({\rm mJy\ kpc}^2)$", fontsize=fsize)

    ax.set_xticklabels(xt)
    
    plt.setp(ax.get_xticklabels(), fontsize=tfsize)
    plt.setp(ax.get_yticklabels(), fontsize=tfsize)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))

    plt.legend(loc='lower center', fontsize=18)
    plt.show()
    
    return 


#####################
# VLA Obs Paramters #
#####################


Spar = {'band' : 'S', 
        'df' : 1., 
        'dt' : 0.064, 
        'bw' : 1500.0, 
        'fc' : 3e3}

Cpar = {'band': 'C', 
        'df' : 4.,
        'dt' : 0.064, 
        'bw' : 4e3,
        'fc' : 6e3}

Xpar = {'band' : 'X', 
        'df' : 4.,
        'dt' : 0.064,
        'bw' : 4e3, 
        'fc' : 10e3}

Kupar = {'band' : 'Ku', 
         'df' : 6.,
         'dt' : 0.064,
         'bw' : 6e3, 
         'fc' : 15e3}

Kpar = {'band' : 'K', 
        'df' : 16.,
        'dt' : 0.064, 
        'bw' : 6e3, 
        'fc' : 22e3}


####################
#  MAKE PLOTS      #
####################

parlist = [Spar, Xpar, Kupar]

make_plot(parlist)


