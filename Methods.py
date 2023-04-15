import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def Fit(x, y, func=None, sigy=None, p0=None):    
     
    popt, pcov = curve_fit(func,x,y,p0=p0)
    
    # PRINTING PARAMETERS
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

    # RESIDUES AND CHI^2
    r = ((func(x,*popt)-y))
    if sigy is not None:
        chiq = sum((r/sigy)**2)
        NGL = len(x)-len(popt)
        print(f'NGL = {NGL} \nchiq = {chiq}')
        return popt,pcov,r
    
    return popt,pcov,r

def Plot(x, y, sigy=None, func: 'function'=None, p0=None,
         xlabel: str='X', ylabel: str='Y', log: bool=False, marker: str='s', 
         markersize: float=12, markeredge: str='k', markerface: str='none'):
    
    popt,pcov,r = Fit(x,y,func,sigy,p0)
        
    if sigy!=None:
        fig,ax = plt.subplots(2,figsize=(8,6),sharex=True,gridspec_kw={'height_ratios': [3, 1],'hspace':0.05})
                
        ax[0].errorbar(x, y, sigy, fmt=marker, ecolor=markeredge,
                       markerfacecolor=markerface, markeredgecolor=markeredge)
        ax[0].plot(x,func(x,*popt),c='r')
        ax[1].errorbar(x, r, sigy, fmt=marker, ecolor=markeredge,
                       markerfacecolor=markerface, markeredgecolor=markeredge)
        
        # scales types
        if log:
            ax[0].set_xscale('log')
            ax[1].set_xscale('log')
    
        ax[0].grid()
        ax[1].grid()
        ax[0].set_ylabel(ylabel,size=12)
        ax[1].set_ylabel('$Residues$',size=12)

    else:
        fig,ax = plt.subplots(figsize=(8,6))
                
        ax.scatter(x,y,marker=marker,facecolors=markerface,edgecolors=markeredge,s=markersize)
        ax.plot(x,func(x,*popt),c='r')

        # scales types
        if log:
            ax.set_xscale('log')
    
        ax.grid()
        ax.set_ylabel(ylabel,size=12)
    
    plt.xlabel(xlabel,size=12)
                  
    return (ax,popt,pcov)
        
def Grad(point: 'list|tuple', func: 'function', divs: int=1000):
    '''
        Returns the Gradient of a function at a specific point
    '''
    argc = len(point)

    M = np.array( [np.linspace(var-0.1*var, var+0.1*var, divs) for var in point] )
    h = np.array( [M[i][1]-M[i][0] for i in range(argc)] )
    
    F2 = np.array( [func(*M[:,i]) for i in range(1,divs)] )
    F1 = np.zeros((argc,divs-1))
    for i in range(argc):
        for j in range(1,divs):
            coord = [ M[k][j-1] if k==i else M[k][j] for k in range(argc) ]
            F1[i][j-1] = func(*coord)
        
    df = F2-F1
    diff = df[:,divs//2-1]/h
    
    return diff

def Prop_sig(vals: 'list|tuple', sigs: 'list|tuple', func: 'function'):
  ''' Propagate Uncertainties '''
  
  grad = Grad(vals,func)
  sigs = np.array(sigs)
  sigf = np.sqrt( np.sum( grad**2 * sigs**2 ) )

  return sigf

def Z_score(x: float, mu: float, sigx=float, sigmu: float=0):
  sig = (sigx**2 + sigmu**2)**0.5
  Z = abs(x-mu)/sig

  return Z
