import numpy as np
from rebin import find_common_idx_range
from scipy.optimize import curve_fit


def normalize(wave,spec1,spec2,WL_bands):
    '''
    given two spectra, rebinned to a common wavelength vector, matches the spectra with a first order fit
    
    params:
        wave - wavelength vector
        spec1 - first spectrum
        spec2 - second spectrum
        WL_bands - list of lists of len 2, each entry is [minWL,maxWL] for one band
    returns:
        m - slope of fit
        b - intercept of fit
    example:
        m,b=normalize(wave,spec1,spec2,[[4600,4650],[5100,5250]])
        spec1_norm=spec1+(m*wave+b)
        spec2_norm=spec2
    '''
    
    # Determine indices to include in fit
    idx_keep=np.asarray([])
    for i in WL_bands:
        idx_min,idx_max=find_common_idx_range(wave,i[0],i[1])
        idx_keep=np.append(idx_keep,np.arange(idx_min,idx_max))
        
    idx_keep=idx_keep.astype(int)    

    spec1_cut=spec1[idx_keep]
    spec2_cut=spec2[idx_keep]
    diff_cut=spec2_cut-spec1_cut
    
    wave_cut=wave[idx_keep]
    
    p=np.polyfit(wave_cut,diff_cut,deg=1)
    m=p[0]
    b=p[1]
    
    return m,b

def normalize_both(wave,spec1,spec2,WL_bands):
    '''
    given two spectra, rebinned to a common wavelength vector, fits both spectra with first order fits
    
    params:
        wave - wavelength vector
        spec1 - first spectrum
        spec2 - second spectrum
        WL_bands - list of lists of len 2, each entry is [minWL,maxWL] for one band
    returns:
        m1 - slope of fit for spec1
        b1 - intercept of fit for spec1
        m2 - slope of fit for spec2
        b2 - intercept of fit for spec2
    example:
        m1,b1,m2,b2=normalize_both(wave,spec1,spec2,[[4600,4650],[5100,5250]])
        spec1_norm=spec1-(m1*wave+b1)
        spec2_norm=spec2-(m2*wave+b2)
    '''
    
    # Determine indices to include in fit
    idx_keep=np.asarray([])
    for i in WL_bands:
        idx_min,idx_max=find_common_idx_range(wave,i[0],i[1])
        idx_keep=np.append(idx_keep,np.arange(idx_min,idx_max))
        
    idx_keep=idx_keep.astype(int)    

    spec1_cut=spec1[idx_keep]
    spec2_cut=spec2[idx_keep]
    
    wave_cut=wave[idx_keep]
    
    p1=np.polyfit(wave_cut,spec1_cut,deg=1)
    p2=np.polyfit(wave_cut,spec2_cut,deg=1)
    
    m1=p1[0]
    b1=p1[1]
    m2=p2[0]
    b2=p2[1]
    
    return m1,b1,m2,b2


def normalize_emission_lines(wave,spec1,spec2,WL_band):
    
    '''
    given two spectra (that should already be continuum subtracted and rebinned to a common wavelength vector),
    normalizes the two spectra with only a multiplicative factor. Finds f such that spec1=f*spec2.
    
    params:
        wave - wavelength vector
        spec1 - first spectrum
        spec2 - second spectrum
        WL_band - list of len 2, [minWL,maxWL] for emission line
    returns:
        f - multiplicative factor to match spec2 to spec1
    example:
        f=normalize_emission_lines(wave,spec1_norm,spec2_norm,[4770,4930])
        spec2_norm_match=f*spec2_norm
    '''
    
    
    idx_min,idx_max=find_common_idx_range(wave,WL_band[0],WL_band[1])
    
    spec1_cut=spec1[idx_min:idx_max]
    spec2_cut=spec2[idx_min:idx_max]

    def mult(x,factor):
        return factor*x
    
    popt, pcov = curve_fit(mult, spec2_cut, spec1_cut)
        
    return popt[0]
    