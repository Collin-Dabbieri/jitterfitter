import numpy as np
from rebin import rebin
from rebin import find_common_idx_range
from normalize import normalize
import warnings
import time


def calculate_shifted_wave(wave,velocity):
    '''
    For a given velocity offset, doppler shifts a wavelength array
    delta lambda / lambda = v/c
    new lambda = lambda + delta lambda
    new wave = wave + (v/c)*wave
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
        variance_quad=(err1[i]**2)+(err2[i]**2)
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
    #[0,1,2,3,4,5,6,7,8,9] length=10
    
    # first find velocities and chisqs of points with 5 lowest chi^2 values
    lowest_idx=int(np.argmin(chisqs))
    if lowest_idx<2:
        five_lowest_idx=[0,1,2,3,4]
    elif lowest_idx>len(chisqs)-3:
        five_lowest_idx=[len(chisqs)-5,len(chisqs)-4,len(chisqs)-3,len(chisqs)-2,len(chisqs)-1]
    else:
        five_lowest_idx=[int(lowest_idx-2),int(lowest_idx-1),int(lowest_idx),int(lowest_idx+1),int(lowest_idx+2)]
        
    vel_low=velocities[five_lowest_idx]
    chisq_low=chisqs[five_lowest_idx]
    # next fit a parabola to those points
    p=np.polyfit(vel_low,chisq_low,deg=2)
    c=p[0]
    b=p[1]
    a=p[2]
    
    if c>0:
        # our parabola has a mininum
        # find minimum value of that parabola within the defined vel range
        vel_para=np.linspace(min(vel_low),max(vel_low),num=1000)
        chisq_para=a+b*vel_para+c*vel_para**2
        vel_min=vel_para[np.argmin(chisq_para)]
        chisq_min=np.amin(chisq_para)
    else:
        #print("Min Chisq Parabola has no Minimum, Using Minimum Data Point")
        # our parabola doesn't have a minimum, take the minimum data point
        chisq_min=np.amin(chisqs)
        vel_min=velocities[np.argmin(chisqs)]
    
    return vel_min,chisq_min
    
def calculate_velocity_CI(velocities,chisqs,vel_min,chisq_min):
    '''
    TODO: add argument for changing the desired confidence interval
    
    Utilizes tables for the CI created by a deviation in chi squared value from the minimum to find an alpha% confidence interval
    for radial velocity.
    
    At a one-sigma confidence level, the max change in chisquared is 1, this small value means that the confidence interval is often narrower 
    than the sampling. To remedy this, interpolation is used to upsample the velocities to 10000 points, so sampling should be finer than the CI
    
    params:
        velocities - vector of radial velocity shifts
        chisqs - chi squared value evaluated at each velocity shift
        vel_min - radial velocity value for which chisq is minimized
        chisq_min - minimum chisq value
    returns:
        min_CI_vel - minimum radial velocity within the 99% confidence interval
        max_CI_vel - maximum radial velocity within the 99% confidence interval
    ex:
        min_CI_vel,max_CI_vel=calculate_velocity_CI(velocities,chisqs)
    '''
    
    #the 99% CI about the minimum for one degree of freedom is chi^2min+6.63
    chisq_max=chisq_min+6.63
    
    
    # the 68% CI (one sigma) about the minimum for one degree of freedom is chi^2_min+1
    #chisq_max=chisq_min+1
    
    # upsample with interpolation to find CI in cases where CI is narrower than sampling
    velocities_upsample=np.linspace(min(velocities),max(velocities),num=100000)
    chisqs_upsample=np.interp(velocities_upsample,velocities,chisqs)
    
    mask=np.where(chisqs_upsample<chisq_max)
    vels_CI=velocities_upsample[mask]
    if len(vels_CI)==0:
        #print("Warning: Velocity CI is Empty")
        vels_CI=[vel_min]
    
    #argmin=np.argmin(chisqs)
    #max_vel=velocities[-1]
    #for i in np.arange(argmin,len(chisqs)-1):
    #    if ((chisqs[i]<=chisq_max) and (chisqs[i+1]>=chisq_max)):
    #        max_vel=velocities[i]
    #        break
            
    #min_vel=velocities[0]
    #for i in np.arange(1,argmin):
    #    if ((chisqs[i]<=chisq_max) and (chisqs[i-1]>=chisq_max)):
    #        min_vel=velocities[i]
    #return min_vel,max_vel 
    
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
        NOTE: THIS SHOULD INCLUDE THE BAND FOR THE EMISSION LINE, AND THE EMISSION LINE BAND SHOULD BE LAST
        vmin - minimum velocity offset, or most blueshifted velocity for spec2 (km/s)
        vmax - maximum velocity offset, or most redshifted velocity for spec2 (km/s)
        numpoints - number of calculated velocity shifts
        
    returns:
        velocities - list of sampled radial velocities for spec2
        chisqs - list of chi^2 values for each radial velocity
        wave_fit - common sampled wavelengths for best fit offset
        spec1_fit_norm - spec1_norm resampled at wave_fit
        spec2_fit_norm - spec2_norm shifted to best fit velocity and resampled at wave_fit
        err1_fit_norm - errors for first spectrum resampled at wavefit and normalized
        err2_fit_norm - errors for second spectrum shifted to best fit velocity and resampled at wave_fit
        vel_min - best fit change in radial velocity
        min_CI_vel - lower limit for confidence interval for change in radial velocity
        max_CI_vel - upper limit for confidence interval for change in radial velocity
    
    '''
    
    #first, calculate desired shifts in wavelength
    velocities=np.linspace(vmin,vmax,numpoints)
    chisqs=np.asarray([])
    
    #calculate_shifted_wave_times=np.asarray([])
    #rebin_times=np.asarray([])
    #normalize_times=np.asarray([])
    #common_idx_range_times=np.asarray([])
    #calculate_chisq_times=np.asarray([])
    
    
    for i in range(numpoints):
        
        #time0=time.time()        
        #calculated shifted wavelengths for spec2
        wave2=calculate_shifted_wave(wave,velocities[i])
        #time1=time.time()
        #calculate_shifted_wave_times=np.append(calculate_shifted_wave_times,time1-time0)
           
        #time0=time.time()
        #rebin both spectra onto common wave
        wave_iter,spec1_iter,err1_iter,spec2_iter,err2_iter=rebin(wave,wave2,spec1,spec2,err1,err2)
        #time1=time.time()
        #rebin_times=np.append(rebin_times,time1-time0)
                
        #normalize spectra
        
        #time0=time.time()
        # use both the continuum bands and the band for the emission line for normalization
        # add a lazy catch for a one in a million error that comes from spec1_iter or spec2_iter being []
        try:
            a,b=normalize(wave_iter,spec1_iter,spec2_iter,norm_bands)
        except:
            return None,None,None,None,None,None,None,None,None,None
            
        spec1_iter_norm=a*spec1_iter+b
        spec2_iter_norm=spec2_iter

        # propogate uncertainty
        # z=ax+b dz=adx
        err1_iter_norm=a*err1_iter
        err2_iter_norm=err2_iter
        
        #time1=time.time()
        #normalize_times=np.append(normalize_times,time1-time0)
        
        #time0=time.time()
        #find min and max idx for WL range for fitting
        min_idx,max_idx=find_common_idx_range(wave_iter,minWL,maxWL)
        #time1=time.time()
        #common_idx_range_times=np.append(common_idx_range_times,time1-time0)
        
        
        
        #calculate chi^2 of shifted spectra
        #slicing the flux arrays to only include the wavelength region with the emission line
        spec1_iter_cut=spec1_iter_norm[min_idx:max_idx]
        spec2_iter_cut=spec2_iter_norm[min_idx:max_idx]
        err1_iter_cut=err1_iter_norm[min_idx:max_idx]
        err2_iter_cut=err2_iter_norm[min_idx:max_idx]
             
        #time0=time.time()
        # Error in quadrature of the two spectra is used for denominator of chisq sum
        chisq=calculate_chisq(spec1_iter_cut,spec2_iter_cut,err1_iter_cut,err2_iter_cut)
        chisqs=np.append(chisqs,chisq)
        #time1=time.time()
        #calculate_chisq_times=np.append(calculate_chisq_times,time1-time0)
        
        
    #time0=time.time()
    # Find radial velocity that minimizes chisq
    vel_min,chisq_min=calculate_min_velocity(velocities,chisqs)
    #time1=time.time()
    #find_min_vel_time=time1-time0
    
    #time0=time.time()
    # Calculate confidence interval for radial velocities
    min_CI_vel,max_CI_vel=calculate_velocity_CI(velocities,chisqs,vel_min,chisq_min)
    #time1=time.time()
    #find_ci_time=time1-time0
    
    # For plotting, return spectra at best fit radial velocity
    best_fit_velocity=vel_min
    best_fit_wave2=calculate_shifted_wave(wave,best_fit_velocity)
    wave_fit,spec1_fit,err1_fit,spec2_fit,err2_fit=rebin(wave,best_fit_wave2,spec1,spec2,err1,err2)
    a,b=normalize(wave_fit,spec1_fit,spec2_fit,norm_bands)
    #print("a: "+str(round(a,2)))
    #print("b: "+str(round(b,2)))
    spec1_fit_norm=a*spec1_fit+b
    spec2_fit_norm=spec2_fit
    #z=ax+b
    #dz=a*dx
    err1_fit_norm=a*err1_fit
    err2_fit_norm=err2_fit
    
    #print("Total Calculate Shifted Wave Time: {:.3f} s".format(sum(calculate_shifted_wave_times)))
    #print("Total Rebin Time: {:.3f} s".format(sum(rebin_times)))
    #print("Total Normalize Time: {:.3f} s".format(sum(normalize_times)))
    #print("Total Common idx Range Time: {:.3f} s".format(sum(common_idx_range_times)))
    #print("Total Calculate Chisq Time: {:.3f} s".format(sum(calculate_chisq_times)))
    #print("Calculate Min Velocity Time: {:.3f} s".format(find_min_vel_time))
    #print("Calculate CI Time: {:.3f} s".format(find_ci_time))
    #print("##################################################")
        
    return velocities,chisqs,wave_fit,spec1_fit_norm,spec2_fit_norm,err1_fit_norm,err2_fit_norm,vel_min,min_CI_vel,max_CI_vel
        
        
    
    
    
    
    
    