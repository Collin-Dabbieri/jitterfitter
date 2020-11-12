import numpy as np
from rebin import rebin
from rebin import find_common_idx_range
from normalize import normalize
from normalize import normalize_both
from normalize import normalize_emission_lines
import warnings


def calculate_shifted_wave(wave,velocity):
    '''
    For a given velocity offset, doppler shifts a wavelength array
    '''
    c=299800.0 #km/s
    
    wave=np.asarray(wave)
    new_wave=wave+(velocity/c)*wave
    
    return new_wave

def calculate_chisq(spec1,spec2,err1,err2):
    '''
    given 2 spectra on a common wavelength vector, calculates chi squared
    '''
    
    chisq=0
    for i in range(len(spec1)):
        
        #sigma=sqrt(err1[i]^2+err2[i]^2)
        #sigma^2=err1[i]^2+err2[i]^2
        variance_quad=err1[i]**2+err2[i]**2
        chisq+=((spec1[i]-spec2[i])**2)/variance_quad
        
    
        
    return chisq

def calculate_min_velocity(velocities,chisqs):
    '''
    given radial velocities for shifts, and the chisq values for those radial velocities, finds the minimum velocity and chisq values. Achieves this by fitting a parabola to the 5 lowest points in chisqs.
    
    params:
        velocities - vector of radial velocity shifts
        chisqs - chi squared value evaluated at each velocity shift
    returns:
        vel_min - radial velocity value for which chisq is minimized by parabola fit
        chisq_min - minimum chisq value of parabola fit
    ex:
        vel_min,chisq_min=calculate_min_velocity(velocities,chisqs)
    '''
    
    #fit parabola to 5 lowest points to find minimum chisq value
    
    # first find velocities and chisqs of points with 5 lowest chi^2 values
    five_lowest_idx=np.argsort(chisqs)[:5]
    vel_low=velocities[five_lowest_idx]
    chisq_low=chisqs[five_lowest_idx]
    # next fit a parabola to those points
    p=np.polyfit(vel_low,chisq_low,deg=2)
    c=p[0]
    b=p[1]
    a=p[2]
    # find minimum value of that parabola
    # a+bx+cx^2=y
    # minimum is where derivative=0
    # b+2cx=0
    #x=-b/2c
    vel_min=-b/(2*c)
    # check that this derivative is a minimum
    # if c>0 -> minimum
    # if c<0 -> maximum
    if c<0:
        warnings.warn("Parabola fit does not return a minimum")
    # find the chisq value at the minimum velocity for the parabola of best fit
    chisq_min=a+b*vel_min+c*vel_min**2
    
    return vel_min,chisq_min
    
def calculate_velocity_CI(velocities,chisqs,vel_min,chisq_min):
    '''
    TODO: add argument for changing the desired confidence interval
    
    Utilizes tables for the CI created by a deviation in chi squared value from the minimum to find a 99% confidence interval
    for radial velocity.
    
    params:
        velocities - vector of radial velocity shifts
        chisqs - chi squared value evaluated at each velocity shift
        vel_min - radial velocity value for which chisq is minimized
        chisq_min - minimum chisq value
    returns:
        min_CI_vel - minimum radial velocity within the 99% confidence interval
        max_CI_vel - maximum radial velocity within the 99% confidence interval
    ex:
        min_CI_vel,max_CI_vel=calculate_velocity_CI(velocities,chisqs,vel_min,chisq_min)
    '''
    
    #TODO the 99% CI about the minimum for one degree of freedom is chi^2min+6.63
    chisq_max=chisq_min+6.63
    
    mask=np.where(chisqs<chisq_max)
    vels_CI=velocities[mask]
    
    return min(vels_CI),max(vels_CI)
    

def fit_radial_velocity(wave,spec1,spec2,err1,err2,minWL,maxWL,norm_bands,vmin=-1500,vmax=1500,numpoints=1000):
    '''
    given 2 spectra, and a wavelength range for fitting, fits the change in radial velocity from spec1 to spec2.
    Note: Blueshifts are negative, if an emission line gets more redshifted from spec1 to spec2, it will have a positive value.
    
    params:
        wave - vector of common wavelengths for both spectra
        spec1 - first spectrum, rebinned onto common wavelengths
        spec2 - second spectrum, rebinned onto common wavelengths
        err1 - errors for first spectrum, rebinned onto common wavelengths
        err2 - errors for second spectrum, rebinned onto common wavelengths
        minWL - minimum wavelength value for fitting chi^2
        maxWL - maximum wavelength value for fitting chi^2
        norm_bands - list of lists each entry is of length 2 and contains [minWL,maxWL] for a band of wavelengths used for normalization
        vmin - minimum velocity offset, or most blueshifted velocity for spec2 (km/s)
        vmax - maximum velocity offset, or most redshifted velocity for spec2 (km/s)
        numpoints - number of calculated velocity shifts
        
    returns:
        velocities - list of sampled radial velocities for spec2
        chisqs - list of chi^2 values for each radial velocity
        wave_fit - common sampled wavelengths for best fit offset
        spec1_fit - spec1_norm resampled at wave_fit
        spec2_fit - spec2_norm shifted to best fit velocity and resampled at wave_fit
    
    '''
    
    #first, calculate desired shifts in wavelength
    velocities=np.linspace(vmin,vmax,numpoints)
    chisqs=np.asarray([])
    
    for i in range(numpoints):
        
        #calculated shifted wavelengths for spec2
        wave2=calculate_shifted_wave(wave,velocities[i])
           
        #rebin both spectra onto common wave
        wave_iter,spec1_iter,err1_iter,spec2_iter,err2_iter=rebin(wave,wave2,spec1,spec2,err1,err2)
        
        #normalize spectra
        m1,b1,m2,b2=normalize_both(wave_iter,spec1_iter,spec2_iter,norm_bands)
        spec1_iter_norm=spec1_iter-(m1*wave_iter+b1)
        spec2_iter_norm=spec2_iter-(m2*wave_iter+b2)
        
        #normalize emission lines that will be fit
        f=normalize_emission_lines(wave_iter,spec1_iter_norm,spec2_iter_norm,[minWL,maxWL])
        spec2_iter_norm=f*spec2_iter_norm
        
        #find min and max idx for WL range for fitting
        min_idx,max_idx=find_common_idx_range(wave_iter,minWL,maxWL)
        
        #calculate chi^2 of shifted spectra
        spec1_iter_cut=spec1_iter_norm[min_idx:max_idx]
        spec2_iter_cut=spec2_iter_norm[min_idx:max_idx]
        err1_iter_cut=err1_iter[min_idx:max_idx]
        err2_iter_cut=err2_iter[min_idx:max_idx]
        chisq=calculate_chisq(spec1_iter_cut,spec2_iter_cut,err1_iter_cut,err2_iter_cut)
        chisqs=np.append(chisqs,chisq)
        
    # Find radial velocity that minimizes chisq
    vel_min,chisq_min=calculate_min_velocity(velocities,chisqs)
    
    # Calculate confidence interval for radial velocities
    min_CI_vel,max_CI_vel=calculate_velocity_CI(velocities,chisqs,vel_min,chisq_min)
    

    # For plotting, return spectra at best fit radial velocity
    best_fit_velocity=vel_min
    best_fit_wave2=calculate_shifted_wave(wave,best_fit_velocity)
    wave_fit,spec1_fit,err1_fit,spec2_fit,err2_fit=rebin(wave,best_fit_wave2,spec1,spec2,err1,err2)
    m1,b1,m2,b2=normalize_both(wave_fit,spec1_fit,spec2_fit,norm_bands)
    spec1_fit_norm=spec1_fit-(m1*wave_fit+b1)
    spec2_fit_norm=spec2_fit-(m2*wave_fit+b2)
    f=normalize_emission_lines(wave_fit,spec1_fit_norm,spec2_fit_norm,[minWL,maxWL])
    spec2_fit_norm_match=f*spec2_fit_norm
    
        
        
    return velocities,chisqs,wave_fit,spec1_fit_norm,spec2_fit_norm_match,err1_fit,err2_fit,vel_min,min_CI_vel,max_CI_vel
        
        
    
    
    
    
    
    