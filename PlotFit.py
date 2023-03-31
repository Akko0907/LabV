import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def Fit(x,y,func=None,sigy=None,kicks=None):    
    # storages the parameters in popt
    # and the cov matrix in pcov. 
    popt, pcov = curve_fit(func,x,y,p0=kicks)
    
    # We calculate the deviations and 
    # uncertaints in the parameters and 
    # print it out.
    print(' Parameters calculated: ')
    i = 0
    dic = 'abcdefg' 
    while i < len(popt):
        k = popt[i]
        sigma_k = (pcov[i][i]**0.5)
        print(f'{dic[i]} = {k} +- {sigma_k}')
        i+=1
    print()
    print(pcov)

    # calculates the absolute residuals
    # and the chi_square of the fit, along
    # with its degrees of freedom ( needed to
    # avaliate how good is the fit )
    r = ((func(x,*popt)-y))
    if sigy is not None:
        chiq = sum((r/sigy)**2)
        NGL = len(x)-len(popt)
        print(f'NGL = {NGL} \nchiq = {chiq}')
        return popt,pcov,r
    
    return popt,pcov,r

def Plot(x,y,sigy=None,
         func=None,kicks=None,
         xlabel='X',ylabel='Y',
         xlim=None,ylim=None,xtick=None,ytick=None,log=False):
    
    # Graph using some sort of fit
    if func!=None:

        popt,pcov,r = Fit(x,y,func=func,sigy=sigy,kicks=kicks)
        
        if sigy!=None:
            fig,ax = plt.subplots(2,figsize=(8,6),sharex=True,gridspec_kw={'height_ratios': [3, 1],'hspace':0.05})
                
            ax[0].errorbar(x,y,sigy,fmt='s',markerfacecolor='none',markeredgecolor='k')
            ax[0].plot(x,func(x,*popt),c='r')
            ax[1].errorbar(x,r,sigy,fmt='s',markerfacecolor='none',markeredgecolor='k')
        
            # scales types
            if log:
                ax[0].set_xscale('log')
                ax[1].set_xscale('log')
            if xlim is not None:
                ax[0].xlim(xlim)
            if ylim is not None:
                ax[0].ylim(ylim)
            if xtick is not None:
                ax[0].xticks(xtick)
            if ytick is not None:
                ax[0].yticks(ytick)
    
            ax[0].grid()
            ax[1].grid()
            ax[0].set_ylabel(ylabel,size=12)
            ax[1].set_ylabel('$Residues$',size=12)

        else:
            fig,ax = plt.subplots(figsize=(8,6))
                
            ax.scatter(x,y,marker='s',facecolors='none',edgecolors='k',s=12)
            ax.plot(x,func(x,*popt),c='r')

            # scales types
            if log:
                ax.set_xscale('log')
            if xlim is not None:
                ax.xlim(xlim)
            if ylim is not None:
                ax.ylim(ylim)
            if xtick is not None:
                ax.xticks(xtick)
            if ytick is not None:
                ax.yticks(ytick)
    
            ax.grid()
            ax.set_ylabel(ylabel,size=12)
    
        plt.xlabel(xlabel,size=12)
                  
        return (ax,popt,pcov)
    
    # Graph without fit
    else:        
        fig,ax = plt.subplots()
        
        ax.scatter(x,y,marker='.',c='k') 
        
        # scale types
        if log:
            ax.set_xscale('log')
        if xlim is not None:
            ax.xlim(xlim)
        if ylim is not None:
            ax.ylim(ylim)
        if xtick is not None:
            ax.xticks(xtick)
        if ytick is not None:
            ax.yticks(ytick)
        
        ax.grid()
        ax.set_ylabel(ylabel,size=12)
        ax.set_xlabel(xlabel,size=12)
        
def Gradient(vals,func):
  '''
    Returns the Gradient of a function at a specific point
  '''
  argc = len(vals)

  M = np.array( [np.linspace(val-0.1*val,val+0.1*val,200) for val in vals] )
  h = np.array( [M[i][1]-M[i][0] for i in range(argc)] )
  
  F2 = np.array( [func(*M[:,i]) for i in range(1,200)] )
  F1 = np.zeros((argc,199))
  for i in range(argc):
    for j in range(1,200):
      coord = [ M[k][j-1] if k==i else M[k][j] for k in range(argc) ]
      F1[i][j-1] = func(*coord)
    
  df = (F2-F1)
  diff = df[:,99]/h
  
  return diff

def Propagate_sig(vals,sigs,func):
  grad = Gradient(vals,func)
  sigs = np.array(sigs)
  sigf = np.sqrt( np.sum( grad**2 * sigs**2 ) )

  return sigf


def Z_score(x,mu,sigx,sigmu=0):
  sig = (sigx**2 + sigmu**2)**0.5
  Z = abs(x-mu)/sig

  return Z
