import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
from numpy.fft import fft
import numpy as np

def Fit(x, y, func=None, sigy=None, p0=None, mute=True) -> 'tuple[np.ndarray]':    
     
    popt, pcov = curve_fit(func,x,y,p0=p0)
    
    # PRINTING PARAMETERS
    if not mute:
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
        return popt,pcov,r
        
    
    return popt,pcov,r

def PlotFit(x, y, sigy=None, func: 'function'=None, p0=None,
         xlabel: str='X', ylabel: str='Y', log: bool=False, marker: str='s', 
         markersize: float=12, markeredge: str='k', markerface: str='none'):
    
    popt,pcov,r = Fit(x,y,func,sigy,p0,mute=False)
        
    if sigy is not None:
        fig,ax = plt.subplots(2,figsize=(8,6),sharex=True,gridspec_kw={'height_ratios': [3, 1],'hspace':0.05})
                
        ax[0].errorbar(x, y, sigy, fmt=marker, ecolor=markeredge, markersize=markersize,
                       markerfacecolor=markerface, markeredgecolor=markeredge)
        if len(x)<50:
            start = x[0]
            finish = x[len(x)-1]
            u = np.linspace(start,finish,50)
        else:
            u = x
        ax[0].plot(u,func(u,*popt),c='r')
        ax[1].errorbar(x, r, sigy, fmt=marker, ecolor=markeredge, markersize=markersize,
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
        if len(x)<50:
            start = x[0]
            finish = x[len(x)-1]
            u = np.linspace(start,finish,50)
        else:
            u = x
        ax.plot(u,func(u,*popt),c='r')

        # scales types
        if log:
            ax.set_xscale('log')
    
        ax.grid()
        ax.set_ylabel(ylabel,size=12)
    
    plt.xlabel(xlabel,size=12)
                  
    return (ax,popt,pcov)

def Plot(x, y, sigy=None, xlabel: str='X', ylabel: str='Y', log: bool=False, 
         marker: str='s', markersize: float=12, markeredge: str='k', markerface: str='none') -> plt.Axes:
    
    if sigy!=None:
        fig,ax = plt.subplots()
        ax.errorbar(x, y, sigy, fmt=marker, ecolor=markeredge,
                       markerfacecolor=markerface, markeredgecolor=markeredge)
        # scales types
        if log:
            ax.set_xscale('log')
          
        ax.grid()
        ax.set_ylabel(ylabel,size=12)
        
    else:
        fig,ax = plt.subplots(figsize=(8,6))
                
        ax.scatter(x,y,marker=marker,facecolors=markerface,edgecolors=markeredge,s=markersize)
        
        # scales types
        if log:
            ax.set_xscale('log')
    
        ax.grid()
        ax.set_ylabel(ylabel,size=12)
    
    plt.xlabel(xlabel,size=12)
                  
    return ax

def Grad(point: 'list|tuple', func: 'function', divs: int=1000) -> np.ndarray:
    '''Returns the Gradient of a function at a specific point.'''
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

def Prop_sig(vals: 'list|tuple', sigs: 'list|tuple', func: 'function') -> float:
  ''' Propagate Uncertainties. Receives 2 lists/tuples of values representing
    the point/uncertainty, and the function relating the variables.

    Example: 
    
    f(x, y, z) receives the tuple vals=(0.2, 1, 3.75) and sigs=(0.05, 0.1, 0.32) where,

    x = 0.2+-0.05
   
    y = 1+-0.1

    z = 3.75+-0.32

    returns the uncertainty of f(0.2, 1, 3.75).  
  '''
  
  grad = Grad(vals,func)
  sigs = np.array(sigs)
  sigf = np.sqrt( np.sum( grad**2 * sigs**2 ) )

  return sigf

def Z_score(x: float, mu: float, sigx: float, sigmu: float=0) -> float:
  '''Returns The Z-score of X'''
  sig = (sigx**2 + sigmu**2)**0.5
  Z = abs(x-mu)/sig

  return Z

def FFTPlot(x, y, ylabel: str="FFT Amplitude", xlabel: str='Frequency',rel_height=1):
    N = len(x)
    Fs = N/(x[N-1]-x[0])
    Yk = fft(y)

    freq = np.arange(0,Fs/2,Fs/N)
    A = abs(Yk[:len(freq)])/(len(freq))

    peaks_index, _ = find_peaks(A, prominence=0.2*max(A))
    fpeaks = np.array(freq[peaks_index])
    Apeaks = np.array(A[peaks_index])

    width,height,left,right = peak_widths(A,peaks_index,rel_height=rel_height)
    right = right*Fs/N
    left = left*Fs/N
    width = right-left
    annots = [f"${round(f,3)}$" for f in fpeaks]
 

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(freq,A,c='k')
    ax.plot(fpeaks, Apeaks+0.015*Apeaks,"vr")
    for i, txt in enumerate(annots):
        ax.annotate(txt, (fpeaks[i], Apeaks[i]+0.03*Apeaks),fontsize=12,ha='center')
    ax.hlines(height,left,right, color="C3")


    ax.grid()
    ax.set_ylabel(ylabel,size=12)
    ax.set_xlabel(xlabel,size=12)

    return ax,fpeaks, [width,height,left,right]

def R_squared(y,yfit) -> float:
    r = y- yfit
    ss_res = np.sum(r**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared

def GenStatTable(function: str, y, yfit, popt, pcov, sigy=None):
    # FUNCTION TYPE
    txts = [function]
    table = ""

    # FITTED PARAMETERS
    i = 0
    dic = 'abcdefg' 
    while i < len(popt):
        k = popt[i]
        sigma_k = (pcov[i][i]**0.5)
        txts.append(f'${dic[i]} = {k:.4g} \pm {sigma_k:.4g}$')
        i+=1
    
    # R-SQUARED
    r2 = R_squared(y,yfit)
    txts.append(f"$R^2 = {r2:.5g}$")

    # CHI^2
    if sigy is not None:
        chiq = sum(((y-yfit)/sigy)**2)
        txts.append(f"$\chi^2 = {chiq:.5g}$")
        NGL = len(y)-len(popt)
        txts.append(f"$NGL = {NGL:.5g}$")

    for txt in txts:
        table += txt + "\n"
    
    return table[:-1]
    