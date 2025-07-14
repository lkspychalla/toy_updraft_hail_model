'''
7/14/2025
grow_hail_master.py
By Lydia Spychalla

Refer to Spychalla and Kumjian (2025a) "An Analytic Approximation of Vertical Velocity and Liquid Water Content Profiles in Supercell Updrafts and Their Use in a Novel Idealized Hail Model" (Journal of Atmospheric Science). The equations used herein are taken from this study, where the hail model is described in detail.

This script will compute hailstone trajectories from specified parameters, or from parameters sampled randomly from coupled or uniform distributions [see the "SP", "MP", and "EX" Monte Carlo simulations in Spychalla and Kumjian (2025b) "Uncoupling the Impact of Environment and Updraft Quantities on Hail Growth using an Idealized Hail Model"]. The final size, total time aloft, and whether a hailstone was lofted or hit the ground will be returned corresponding to each hailstone.

Tasks:
- Look over comments to see if anything is incomplete
- Test speed. Feels a little slow and not sure if anything is gumming up the computation?

Tasks I think I've completed, but should check:
- delt = 1. Update so that a user may specify
- Preserve memory use: instead of maintaining all Ds and zs, track only the current values; create a new script that can return z, D time series given parameter values.
- Parallelization is still pretty new. Double check that it works identically to the serial implementation
- Fixed the EX sampling issue. Had negative d_lwcs instead of positive
- Clean up and break up the sampling function into smaller subfunctions
- Made a note that n_samples is not used when sampling_method='manual'
- chunks issue still happening
'''



#--------------------------------------------------Import Dependencies--------------------------------------------------
import numpy as np
from tqdm import tqdm # Can remove tqdm if desired. I like it for keeping track of progress
import xarray as xr
import multiprocessing
import warnings


#--------------------------------------------------Define Constants--------------------------------------------------
# (These constants are defined globally, so all functions may access them)

'''
Z-scores of 10th and 90th percentiles
'''
global Z90, Z10
Z90 = 1.282
Z10 = -1.282

'''
List of a_m and b_m from the empirical mass-maximum dimension power laws from the article "A Comprehensive Observational Study of Graupel and Hail Terminal Velocity, Mass Flux, and Kinetic Energy" by Heymsfield et al. (2018). These power laws are listed as (a_m, b_m) pairs of the best fit to the 10th, 25th, 50th, 75th, and 90th percentile of the data. The coefficient is represented by a. The exponent is represented by b.
'''
global AB_M_LIST
AB_M_LIST = [(0.19,2.78), (0.27,2.72), (0.37,2.69), (0.47,2.73), (0.53,2.82)]


'''
The a's and b's corresponding to the best fit power laws to the 10th and 90th percentile of the area/mass-maximum dimension relationships found by Heymsfield et al. (2018). The coefficient is represented by a. The exponent is represented by b.
'''
global A_AREA_P10, A_AREA_P90, B_AREA_P10, B_AREA_P90
A_AREA_P10 = 0.32
A_AREA_P90 = 0.81
B_AREA_P10 = 0.88
B_AREA_P90 = 0.94

'''
The a's and b's corresponding to the best fit power laws to the 10th and 90th percentile of the terminal velocity-maximum dimension  relationships found in "CORRIGENDUM", the companion article to Heymsfield et al (2018), written in 2020. The coefficient is represented by a. The exponent is represented by b.
'''
global A_FALL_P10, A_FALL_P90, B_FALL_P10, B_FALL_P90
A_FALL_P10 = 6.73
A_FALL_P90 = 11.58
B_FALL_P10 = 0.68
B_FALL_P90 = 0.58


'''
Function defining the hail embryo sampling distribution used for the SP simulation in Spychalla and Kumjian (2025b)
Outputs:
  lamb     (float) = the lambda value of the exponential distribution
'''
def graupel_distribution():
    rho_graupel          = 300   #kg/m^3
    n0_graupel           = 4e6   #m-4
    rho_air              = 1     #kg/m3
    mixing_ratio_graupel = 0.001 #kg/kg
    lamb = ((np.pi*rho_graupel*n0_graupel)/(rho_air*mixing_ratio_graupel))**0.25
    return lamb


#--------------------------------------------------Main Hail Growth Equations--------------------------------------------------

'''
Calculte the LWC available to the hailstone at the hailstone's current altitude

Variable Inputs:
  zi                (float array) = hailstone altitude [m]
  ti                      (float) = simulation time [s]
Parameter Inputs:
  z_lcl             (float array) = height of the LCL [m]
  z_fl              (float array) = height of the freezing level [m]
  d_lwc             (float array) = height of maximum aLWC relative to the midpoint of the potentially liquid cloud layer
                                    ( z_alwc_max - (z_gl - z_lcl)/2 ) [m]
  dz_hgz            (float array) = depth of the hail growth zone [m]
  lwc_max           (float array) = maximum value of LWC [kg/m^3]
  T                 (float array) = updraft duration [s]
  P_T               (float array) = proportion of the updraft after t = 0 s

Outputs:
  lwc_z_t           (float array) = LWC at (zi, ti) [kg/m^3]
'''
def LWC(zi, ti, z_lcl, z_fl, d_lwc, dz_hgz, lwc_max, T, P_T):

    # Calculate heights of T = -25 degC (z_25) and T = -38 degC (z_GL)
    z_25 = dz_hgz*(0.7127) + z_fl
    z_gl = dz_hgz + z_fl

    # Calculate the quadratic approximation of LWC without freezing. Catch any negatives.
    alwc_quad = lwc_max*(1-((2*zi-2*d_lwc-dz_hgz-z_fl-z_lcl)/(2*d_lwc+dz_hgz+z_fl-z_lcl))**2)
    alwc_quad[alwc_quad < 0] = 0
    
    # Define and multiply the freezing function
    f = np.zeros_like(alwc_quad)
    f[np.logical_and(zi >= z_fl, zi <= z_25)] = 1
    f[np.logical_and(zi <= z_gl, zi >= z_25)] = ((z_gl-zi)/(z_gl-z_25))[np.logical_and(zi <= z_gl, zi >= z_25)]
    lwc_z = alwc_quad*f
    # Catch any nan or negative LWCs
    if np.any(np.logical_or(~np.isfinite(lwc_z), lwc_z < 0)):
        raise Warning('Invalid LWC(z)')

    # Scale LWC in time by the updraft pass function
    scale = (1 - ( 2*ti/T - 2*P_T + 1)**2)
    scale[scale < 0] = 0
    lwc_z_t = lwc_z * scale

    # Check for nan or negative LWCs
    if np.any(np.logical_or(lwc_z_t > 0.014, lwc_z_t < 0)):
        raise Warning(f'Invalid LWC(z,t)={lwc_z_t[np.logical_or(lwc_z_t > 0.014, lwc_z_t < 0)]}')
    
    return lwc_z_t
    
    

    
 
'''
Calculte the maximum dimensions of the hailstones after the current timestep

Variable Inputs:
  Di                (float array) = hailstone maximum dimension [m]
  zi                (float array) = hailstone altitude [m]
  ti                      (float) = simulation time [s]
  vTi               (float array) = hailstone fall speed [m/s]
  delt                    (float) = integration timestep [s]
Parameter Inputs:
  a_m               (float array) = mass power law coefficient
  b_m               (float array) = mass power law exponent
  z_lcl             (float array) = height of the lcl [m]
  z_fl              (float array) = height of the freezing level [m]
  d_lwc             (float array) = height of maximum aLWC relative to the midpoint of the potentially liquid cloud layer
                                    ( z_alwc_max - (z_gl - z_lcl)/2 ) [m]
  dz_hgz            (float array) = depth of the hail growth zone [m]
  lwc_max           (float array) = maximum value of LWC [kg/m^3]
  T                 (float array) = updraft duration [s]
  P_T               (float array) = proportion of the updraft after t = 0 s

Outputs:
  Di1               (float array) = new hailstone maximum dimension
'''
def D(Di, zi, ti, vTi, delt, a_m, b_m, z_lcl, z_fl, d_lwc, dz_hgz, lwc_max, T, P_T):
    
    #Find the LWC at (zi,ti)
    lwci = LWC(zi, ti, z_lcl, z_fl, d_lwc, dz_hgz, lwc_max, T, P_T)
    #Update diameter
    Di1 = Di + 10*A(Di, a_m, b_m)*vTi*lwci/(a_m*b_m*(100*Di)**(b_m - 1))*delt

    return Di1



'''
Calculte the updraft speed experienced by hailstones at the current time

Variable Inputs:
  zi      (float array) = hailstone altitude [m]
  ti            (float) = simulation time [s]
Parameter Inputs:
  z_lfc   (float array) = height of the lifted condensation level [m]
  dz_buoy (float array) = depth of the positively buoyant layer (equilibrium level - level of free convection) [m]
  dz_over (float array) = depth of the overshoot layer (overshooting top level - equilibrium level) [m]
  alpha   (float array) = buoyancy shape parameter
  w_max   (float array) = maximum experienced vertical velocity [m/s]
  T       (float array) = updraft duration [s]
  P_T     (float array) = proportion of the updraft after t = 0 s

Outputs:
  w_z_t   (float array) = vertical velocity at (zi, ti)
'''
def w(zi, ti, z_lfc, dz_buoy, dz_over, alpha, w_max, T, P_T):

    # the vertical profile of w
    w_z = np.zeros_like(zi)
    # We're going to be taking the root of a negative number for zi outside the range [z_lfc, z_ot]
    valid_w = ~np.logical_or(zi < z_lfc, zi > z_lfc+dz_buoy+dz_over)
    w_z[valid_w] = w_max[valid_w]*((zi[valid_w]-z_lfc[valid_w])/dz_buoy[valid_w])**(alpha[valid_w]-1)*((z_lfc[valid_w]+dz_buoy[valid_w]+dz_over[valid_w]-zi[valid_w])/dz_over[valid_w])**((alpha[valid_w]-1)*(dz_over[valid_w]/dz_buoy[valid_w]))
    if np.any(np.logical_or(~np.isfinite(w_z), w_z<0)):
        raise Warning('Invalid w(z)')
    
    # Calculate w(zi,ti)
    scale = (1 - ( 2*ti/T - 2*P_T + 1)**2)
    scale[scale < 0] = 0
    w_z_t = w_z*scale

    # Check any nan or negative w values
    if np.any(np.logical_or(w_z_t > w_max, w_z_t < 0)):
        raise Warning(f'Invalid w(z,t)={w_z_t[np.logical_or(w_z_t > w_max, w_z_t < 0)]}')

    # Return w(z,t)
    return w_z_t



'''
Calculte the hailstone's terminal velocity [based on the Heymsfield et al. (2020) fall speed-maximum dimension power law relations with added stochasticity]
Inputs:
  Di      (float array) = hailstone maximum dimension [m]
Outputs:
  vti     (float array) = hailstone fall speed [m/s]
'''
def vt(Di):
    # Define the vT distribution for Di
    vt10 = A_FALL_P10*(100*Di)**B_FALL_P10
    vt90 = A_FALL_P90*(100*Di)**B_FALL_P90
    
    vt_std = np.abs((vt90 - vt10)/(Z90 - Z10))
    vt_mean = (1/2)*(vt90 - vt10) + vt10
    
    # Sample a normal distribution and transform to the vT mean and std
    random_normal_samples = np.random.normal(0, 1, Di.shape[0])
    vti = (random_normal_samples*vt_std**2) + vt_mean
    return vti





'''
Calculte the hailstone's cross-sectional area for collection [based on the Heymsfield et al. (2018) area/mass-maximum dimension power law relations with added stochasticity]

Variable Inputs:
  Di                (float array) = hailstone maximum dimension [m]
Parameter Inputs:
  a_m               (float array) = mass power-law coefficient
  b_m               (float array) = mass power-law exponent

Outputs:
  area              (float array) = hailstone cross-sectional area [m^2]
'''
def A(Di, a_m, b_m):
    # Define the A distribution at Di
    area10 = a_m*(100*Di)**b_m/(10000*A_AREA_P10*(100*Di)**B_AREA_P10)
    area90 = a_m*(100*Di)**b_m/(10000*A_AREA_P90*(100*Di)**B_AREA_P90)
    
    area_std = np.abs((area90 - area10)/(Z90 - Z10))
    area_mean = (1/2)*(area90 - area10) + area10
    
    # Sample a normal distribution and transform to the A mean and std
    random_normal_samples = np.random.normal(0, 1, Di.shape[0])
    area = (random_normal_samples*area_std**2) + area_mean
    return area







'''
Calculte the hailstone's current altitude

Variable Inputs:
  zi            (float array) = hailstone altitude [m]
  ti                  (float) = simulation time
  vTi           (float array) = hailstone fall speed [m/s]
  delt                (float) = integration timestep [s]
Parameter Inputs:
  z_lfc         (float array) = height of the lifted condensation level [m]
  dz_buoy       (float array) = depth of the positively buoyant layer (equilibrium level - level of free convection) [m]
  dz_over       (float array) = depth of the overshoot layer (overshooting top level - equilibrium level) [m]
  alpha         (float array) = buoyancy shape parameter
  w_max         (float array) = maximum experienced vertical velocity [m/s]
  T             (float array) = updraft duration [s]
  P_T           (float array) = proportion of the updraft after t = 0 s

Outputs:
  zi1           (float array) = new hailtone altitude [m]
'''
def z(zi, ti, vTi, delt, z_lfc, dz_buoy, dz_over, alpha, w_max, T, P_T):
    # Find w(zi,ti)
    wi = w(zi, ti, z_lfc, dz_buoy, dz_over, alpha, w_max, T, P_T) 
    # Update the hailstone's altitude
    zi1 = zi + (wi - vTi)*delt
    return zi1



#--------------------------------------------------Main Model Functions--------------------------------------------------


'''
Main integration function for hail growth. The input parameters are first loaded. The hailstones step forward in time until all have hit the ground or timed out. The total time aloft, final size, and final location are calculated and returned.

Inputs:
  chunk_idx  (int) = the chunk idx to compute here when parallelizing

Outputs:
  return_ds (list) = 
     [
        sample_idx         (int) = sample label
        realization_idx    (int) = realization label
        t_tot            (float) = total time spent aloft [s]
        D_max            (float) = final maximum dimension [m]
        lofted            (bool) = flag for whether the sampled hailstonse 
                                      hit the ground (False) or EL (True)
     ]
'''
def grow_hail(chunk_idx):
    
    # Load in the dataset with the input parameters
    idxs = np.arange(chunk_idx*chunk_size, min((chunk_idx+1)*chunk_size,samples))
    this_ds = parameters.isel(idx=idxs)
    
    # Pull out all the parameters and repeat according to the desired number of realizations per sample  
    z_lfc     = np.repeat(this_ds.z_lfc.values,     realizations)
    dz_buoy   = np.repeat(this_ds.dz_buoy.values,   realizations)
    dz_over   = np.repeat(this_ds.dz_over.values,   realizations)
    w_max     = np.repeat(this_ds.w_max.values,     realizations)
    alpha     = np.repeat(this_ds.alpha.values,     realizations)
    z_el      = z_lfc + dz_buoy
    
    z_lcl     = np.repeat(this_ds.z_lcl.values,     realizations)
    z_fl      = np.repeat(this_ds.z_fl.values,      realizations)
    dz_hgz    = np.repeat(this_ds.dz_hgz.values,    realizations)
    d_lwc     = np.repeat(this_ds.d_lwc.values,     realizations)
    lwc_max   = np.repeat(this_ds.lwc_max.values,   realizations)
    
    H         = np.repeat(this_ds.H.values,         realizations)
    T         = np.repeat(this_ds.T.values,         realizations)
    P_T       = np.repeat(this_ds.P_T.values,       realizations)
    
    a_m       = np.repeat(this_ds.a_m.values,       realizations)
    b_m       = np.repeat(this_ds.b_m.values,       realizations)
    
    # Hailstone initial conditions
    Di        = np.repeat(this_ds.D_0.values,       realizations)
    zi        = dz_hgz*H + z_fl
    ti = 0
    

    # Define an array arrays to hold the data to return including final size, final location, total time aloft, and a flag for trajectory completion
    D_max      = np.zeros_like(Di)*np.nan
    lofted     = np.zeros_like(Di).astype(bool)
    t_tot      = np.zeros_like(Di)*np.nan
    finished   = np.zeros_like(Di).astype(bool)
   

    # Main integration loop. Continue as long as we haven't timed out or finished computing all hail trajectories
    for ti in np.arange(0, np.max(T*P_T) + 1000, delt):
        if np.all(finished):
            break
            
        # Calculate the new D, z values for the next timestep
        vTi = vt(Di)
        Di = D(Di, zi, ti, vTi, delt, a_m, b_m, z_lcl, z_fl, d_lwc, dz_hgz, lwc_max, T, P_T)
        zi = z(zi, ti, vTi, delt, z_lfc, dz_buoy, dz_over, alpha, w_max, T, P_T)
        ti = ti + delt

        # # Record the current zi, ti
        # zs[:,ti] = zi
        # Ds[:,ti] = Di

        # # Set any completed hailstones to nan
        # zs[finished, ti] = np.nan
        # Ds[finished, ti] = np.nan

        # Check for hailstones hitting the top or bottom boundaries
        D_max[(zi > z_el)*(~finished)]    = Di[(zi > z_el)*(~finished)]
        t_tot[(zi > z_el)*(~finished)]    = ti
        lofted[(zi > z_el)*(~finished)]   = True
        finished[(zi > z_el)*(~finished)] = True
        
        D_max[(zi <= 0)*(~finished)]    = Di[(zi <= 0)*(~finished)]
        t_tot[(zi <= 0)*(~finished)]    = ti
        lofted[(zi <= 0)*(~finished)]   = False
        finished[(zi <= 0)*(~finished)] = True
            
    # Label the sample and realizations indexes for each hailstone
    sample_idx = np.repeat(idxs, realizations)
    realization_idx = np.tile(np.arange(realizations), idxs.shape[0])

    # Return a list of the hailstones' indexes, total time aloft, final size, and whether they were lofted or hit the ground
    return_list = [list(sample_idx), list(realization_idx), list(t_tot), list(D_max), list(lofted),]
    return return_list


'''
Run the hail model!

Inputs:
  n_samples                    (int) = number of distinct initial conditions to sample. Note that n_samples will not be used when sampling_method='manual'
  n_realizations               (int) = number of realizations to simulate per set of inputs
  n_chunks                     (int) = number of total chunks to divide the sampled hailstones into for parallel computation
  seed                         (int) = random seed to produce the model input parameters
  dt                         (float) = integration timestep [s]
  sampling_method           (string) = choice of {'sp', 'mp', 'ex', 'manual'}
  ds_hail               (xr.Dataset) = dataset containing coupled hail trajectory scaling parameters to use for input sampling
  ds_storm              (xr.Dataset) = dataset containing coupled environmental parameters to use for input sampling
  ds_all                (xr.Dataset) = dataset containing a complete set of coupled parameters to use for input sampling

    Input parameter dataset formatting:
            ds_hail:
                coords = ['idx'],
                dims   = ['idx'],
                data_vars = 
                
            ds_storm:
                coords = ['idx'],
                dims   = ['idx'],
                data_vars = 'z_lcl',    (idx),
                            'z_lfc',    (idx),
                            'dz_buoy',  (idx),
                            'dz_over',  (idx),
                            'z_fl',     (idx),
                            'dz_hgz',   (idx),
                            'lwc_max',  (idx),
                            'd_lwc',    (idx),
                            'w_max',    (idx),
                            'alpha',    (idx),
                            'P_T',      (idx),
                            'H',        (idx),
                            'D_0',      (idx),
                            'T',        (idx),
                            'a_m',      (idx),
                            'b_m',      (idx),
            ds_all:
                coords = ['idx'],
                dims   = ['idx'],
                data_vars = 

  

Outputs:
  parameters          (xr.Dataset) = dataset containing model input parameters with
                                     the corresponding size statistics, lofting flags
                                     and time statistics
'''
def hail_model(n_samples=1, n_realizations=1, n_chunks=1, seed=0, dt=1, sampling_method='ex', ds_hail=None, ds_storm=None, ds_all=None):

    # Set a random seed and define global variables
    np.random.seed(seed)
    global chunk_size, realizations, samples, delt

    # Makes the dataset parameters that is defined globally that contains all inputs
    sample_parameters(n_samples, sampling_method, ds_hail, ds_storm, ds_all)

    # If n_chunks is too large for total number of samples, change and throw warning
    if n_chunks > n_samples:
        warnings.warn(f'n_chunks too large for number of samples. Current n_samples = {n_samples} and n_chunks = {n_chunks}. Changing n_chunks = {n_samples}.')
        n_chunks = n_samples
    
    # Define chunk size for parallelization, and rename variables to globally-defined names
    chunk_size = np.ceil(n_samples/n_chunks).astype(int)
    # If n_chunks is too large and not a factor of n_samples, some of our chunks will be empty. Find out how many
    n_empty_chunks = np.floor((chunk_size*n_chunks - n_samples) / chunk_size).astype(int)
    if n_empty_chunks > 0:
    # Subtract the empty chunks from the given n_chunks
        warnings.warn(f'Removing {n_empty_chunks} from n_chunks because they would be empty.')
        n_chunks = n_chunks - n_empty_chunks
  
    realizations = n_realizations
    samples = n_samples
    delt = dt

    # Parallelization for hail trajectory computation. Chunk the computation into n_chunks groups
    # All hail trajectory results are saved into output_data_list
    with multiprocessing.Pool() as p:
        output_data_list = list(tqdm(p.imap(grow_hail, range(n_chunks)), total=n_chunks))
    p.close()

    # Parse the output_data_list into separate pieces of data
    sample_idx      = np.stack(sum([x[0] for x in output_data_list],[])).reshape(n_samples, n_realizations)
    realization_idx = np.stack(sum([x[1] for x in output_data_list],[])).reshape(n_samples, n_realizations)
    t_tot           = np.stack(sum([x[2] for x in output_data_list],[])).reshape(n_samples, n_realizations)
    D_max           = np.stack(sum([x[3] for x in output_data_list],[])).reshape(n_samples, n_realizations)
    lofted          = np.stack(sum([x[4] for x in output_data_list],[])).reshape(n_samples, n_realizations)
    realization_labels = realization_idx[0]
    sample_labels = sample_idx[:,0]

    # Add the model output to the parameters Dataset including: total time aloft, final maximum dimension, and whether the hailstones hit the ground or were lofted
    parameters['t_tot']   = xr.DataArray(dims=['idx', 'realization'], coords={'idx': sample_labels, 'realization': realization_labels}, data=t_tot, 
                                            attrs=dict(standard_name="Total_Time_Aloft", units="s"))
    parameters['D_max']   = xr.DataArray(dims=['idx', 'realization'], coords={'idx': sample_labels, 'realization': realization_labels}, data=D_max, 
                                            attrs=dict(standard_name="Final_Hailstone_Maximum_Dimension", units="m"))
    parameters['lofted']  = xr.DataArray(dims=['idx', 'realization'], coords={'idx': sample_labels, 'realization': realization_labels}, data=lofted.astype(bool), 
                                            attrs=dict(standard_name="Was Hailstone Lofted?", units=""))

    # Return the dataset with all the input parameters and model output data
    return parameters



'''
Pull out n-total random values from which to run the idealized updraft and hail model. This function will return a uniform distribution.
Inputs:
  low  (float) = the low limit to the distribution
  high (float) = the high limit to the distribution
  n      (int)   = number of random samples to generate
Outputs:
  A uniform distribution of size n between [low, high] as a numpy array
'''
def random_range(low, high, n):
    return (high - low)*np.random.random(n)+low


'''
Pull out n-total random environments from which to run the idealized updraft and hail model
Inputs:
  ds = xarray dataset from which to sample the storm environments
            ds:
                coords = ['idx'],
                dims   = ['idx'],
                data_vars = 'z_lcl',      (idx),
                            'z_lfc',      (idx),
                            'dz_buoy',    (idx),
                            'dz_over',    (idx),
                            'z_fl',       (idx),
                            'dz_hgz',     (idx),
                            'alwc_max',   (idx),
                            'd_lwc',      (idx),
                            'w_env_max',  (idx),
                            'alpha',      (idx),
                            
  n = number of random storms to generate
Outputs:
      z_lcl      (float array) = sampled LCL heights
      z_fl       (float array) = sampled freezing level heights
      dz_hgz     (float array) = sampled hail growth zone depths
      d_lwc      (float array) = sampled relative height of the adiabatic liquid water content maximum
      alwc_max   (float array) = sampled maximum adiabatic liquid water content
      z_lfc      (float array) = sampled LFC heights
      dz_buoy    (float array) = sampled depths of positive buoyancy
      dz_over    (float array) = sampled depths of overshoot
      w_env_max  (float array) = sampled thermodynamic maximum vertical velocity
      alpha      (float array) = sampled buoyancy shape parameter
      env_inds     (int array) = the sampled indexes of original ds.idx
'''
def random_storms(ds, n):

    #sample n total from the environments
    env_inds = np.random.choice(np.arange(ds.idx.shape[0]), n)
    
    # Pull out all storm datas and sample according to random env_inds
    z_lcl = ds.z_lcl.values[env_inds]
    z_fl = ds.z_fl.values[env_inds]
    dz_hgz = ds.dz_hgz.values[env_inds]
    d_lwc = ds.d_lwc.values[env_inds]
    alwc_max = ds.alwc_max.values[env_inds]
    
    w_env_max = ds.w_env_max.values[env_inds]
    alpha = ds.alpha.values[env_inds]
    z_lfc = ds.z_lfc.values[env_inds]
    dz_buoy = ds.dz_buoy.values[env_inds]
    dz_over = ds.dz_over.values[env_inds]

    # Solve the problem of some missing dz_over. Randomly sample from a normal distribution: N(3088.9, 523.0)
    # Continue to sample until all nans are filled with valid value
    problem_idxs = np.logical_or(np.isnan(dz_over), dz_over <= 0)
    while np.any(problem_idxs):
        dz_over[problem_idxs] = np.random.normal(3088.9, 523.0, problem_idxs.shape[0])[0]
        problem_idxs = np.logical_or(np.isnan(dz_over), dz_over <= 0)

    return z_lcl, z_fl, dz_hgz, d_lwc, alwc_max, z_lfc, dz_buoy, dz_over, w_env_max, alpha, env_inds 
    

'''
Pull out n-total random environments from which to run the idealized updraft and hail model
Inputs:
  ds = xarray dataset from which to sample the trajectory scaling parameters
            ds:
                coords = ['idx'],
                dims   = ['idx'],
                data_vars = 'w_traj_scale',    (idx),
                            'lwc_traj_scale',  (idx),
                            
  n = number of random storms to generate
Outputs:
  w_traj_scale       (float array) = sampled vertical velocity trajectory scaling parameters
  lwc_traj_scale     (float array) = sampled liquid water content trajectory scaling parameters
  hail_indxs           (int array) = the sampled indexes of original ds.idx
'''
def random_hailstones(ds, samples):
    
    #From our list of LWC_maxs and w_maxs, pull out n random samples
    
    hail_indxs = np.random.choice(np.arange(ds.idx.shape[0]), samples)
    
    w_traj_scale   = ds.w_traj_scale.values[hail_indxs]
    lwc_traj_scale = ds.lwc_traj_scale.values[hail_indxs]
    
    return w_traj_scale, lwc_traj_scale, hail_indxs




'''
Define the set of parameters for each trajectory to be calculated.

Inputs:
  sampling_method        (string)
  ds_hail        (xarray.Dataset)
  ds_storm       (xarray.Dataset) 
  ds_all         (xarray.Dataset)
          All parameters passed from hail_model(). See hail_model() documentation for input info

Outputs:
  Nothing! But we define and build the global variable "parameters":
  parameters  (xr.Dataset) = dataset of the input parameters for each trajectory
                             to be run in the model
'''
def sample_parameters(samples, sampling_method='ex', ds_hail=None, ds_storm=None, ds_all=None):

    # Defining parameters globally instead of passing as an input improves speed since its such a large dataset
    global parameters

    # Sample parameters identically to how they were sampled in Spychalla and Kumjian (2025b)
    if sampling_method.lower() in ['sp', 'mp', 'ex']:

        parameters = {}

        # Sample coupled parameters
        if sampling_method.lower() in ['sp', 'mp']:

            z_lcl, z_fl, dz_hgz, d_lwc, alwc_max, z_lfc, dz_buoy, dz_over, w_env_max, alpha, env_idx = random_storms(ds_storm, samples)
            w_traj_scale, lwc_traj_scale, hail_idx = random_hailstones(ds_hail, samples)
            w_max   =   w_traj_scale * w_env_max
            lwc_max = lwc_traj_scale *  alwc_max

            # Sample exponential hail embryo size distribution, with the constraint of no D_0 > 1 cm
            if sampling_method.lower() == 'sp':
                D_0 = np.random.exponential(scale=1/graupel_distribution(), size=samples)
                while D_0[D_0>0.01].shape[0] > 0:
                    D_0[D_0>0.01] = np.random.exponential(scale=1/graupel_distribution(), size=D_0[D_0>0.01].shape[0])

            # Sample uniform enbryo sizes
            else: #(sampling_method.lower() == 'mp')
                D_0 = random_range(0, 0.01, samples)

            # Save the subparameters of lwc_max and w_max to the parameters dataset (this is only relevant with sampling_method = 'sp' or 'mp'
            parameters['lwc_traj_scale'] = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': lwc_traj_scale, 
                                            'attrs': dict(standard_name="Scale_value_of_LWC_Experienced_by_KLTM_Hailstone", units="")}
            parameters['w_traj_scale']   = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': w_traj_scale, 
                                            'attrs': dict(standard_name="Scale_value_of_Vertical_Velocity_Experienced_by_KLTM_Hailstone", units="")}
            parameters['hail_idx']      = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': hail_idx, 
                                            'attrs': dict(standard_name='Hail_index_from_KLTM_dataset', units='')}
            parameters['alwc_max']       = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': alwc_max, 
                                            'attrs': dict(standard_name="Environment_Maximum_Adiabatic_Liquid_Water_Content", units="kg/m^3")}
            parameters['w_env_max']      = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': w_env_max, 
                                            'attrs': dict(standard_name="Environment_Maximum_Vertical_Velocity", units="m/s")}
            parameters['storm_idx']     = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': env_idx, 
                                            'attrs': dict(standard_name='Storm_index_from_parent_dataset', units='')}
                
        # Sample uniform parameters if using sampling_method = 'ex'
        else: #(sampling_method.lower() == 'ex')
            z_lcl   = random_range(     0,  4000, samples)
            z_lfc   = random_range(     0,  6000, samples)
            dz_buoy = random_range(  1000, 14000, samples)
            dz_over = random_range(   500,  7000, samples)
            z_fl    = random_range(  1500,  7500, samples)
            dz_hgz  = random_range(  4800,  5700, samples)
            d_lwc   = random_range(  1200,  2000, samples)
            alpha   = random_range(     1,     4, samples)
            w_max   = random_range(     0,   150, samples)
            lwc_max = random_range(     0, 0.014, samples)
            D_0     = random_range(     0,  0.01, samples)


        # For all sampling methods, sampling randomly mass-maximum dimension parameters, initial altitude, and updraft pass parameters
        ab_mass_idx = np.random.choice(np.arange(len(AB_M_LIST)), samples)
        a_m    = np.array(AB_M_LIST)[ab_mass_idx.astype(int)][:,0]
        b_m    = np.array(AB_M_LIST)[ab_mass_idx.astype(int)][:,1]
        
        H   = random_range(0, 1, samples)
        P_T = random_range(0.5, 1, samples)
        T   = random_range(0, 3600, samples)

        # Add all sampled parameters to the parameters dictionary
        parameters['H']        = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': H, 
                                 'attrs': dict(standard_name="Initial_Height_Normalized_By_Tropopause", units="")}
        parameters['P_T']      = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': P_T, 
                                  'attrs': dict(standard_name="Proportion_of_Updraft_Experienced_after_T=0", units="")}
        parameters['T']        = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': T, 
                                  'attrs': dict(standard_name="Duration_of_Hailstone's_Experience_of_Updraft", units="s")}
        parameters['D_0']      = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': D_0, 
                                  'attrs': dict(standard_name="Hail_Embryo_Maximum_Diameter", units="m")}
        parameters['a_m']      = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': a_m, 
                                  'attrs': dict(standard_name="Heymsfield_Mass_Coefficient", units="")}
        parameters['b_m']      = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': b_m, 
                                  'attrs': dict(standard_name="Heymsfield_Mass_Exponent", units="")}
        parameters['z_lcl']    = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': z_lcl, 
                                  'attrs': dict(standard_name="LCL_Height", units="m")}
        parameters['z_lfc']    = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': z_lfc, 
                                  'attrs': dict(standard_name="LFC_Height", units="m")}
        parameters['dz_buoy']  = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': dz_buoy, 
                                  'attrs': dict(standard_name="Positive_Buoyancy_Depth", units="m")}
        parameters['dz_over']  = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': dz_over, 
                                  'attrs': dict(standard_name="Overshoot_Depth", units="m")}
        parameters['z_fl']     = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': z_fl, 
                                  'attrs': dict(standard_name="Freezing_Height_in_Updraft", units="m")}
        parameters['dz_hgz']   = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': dz_hgz, 
                                  'attrs': dict(standard_name="Hepth_of_the_HGZ", units="m")}
        parameters['d_lwc']    = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': d_lwc, 
                                  'attrs': dict(standard_name="Relative_Height_of_LWC_max", units="m")}
        parameters['alpha']    = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': alpha, 
                                  'attrs': dict(standard_name="Buoyancy_Shape_Parameter", units="")}
        parameters['lwc_max'] = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': lwc_max, 
                                 'attrs': dict(standard_name="Maximum_Liquid_Water_Content", units="kg/m^3")}
        parameters['w_max']   = {'dims': ('idx'), 'coords': {'idx':np.arange(samples)}, 'data': w_max, 
                                'attrs': dict(standard_name="Maximum_Vertical_Velocity", units="m/s")}

        # Convert to an xarray dataset
        parameters = xr.Dataset.from_dict(parameters)

    # If choosing to manually pass parameters, rename the input dataset to parameters
    elif sampling_method.lower() == 'manual':
        
        parameters = ds_all

    # Catch any invalid values of sampling_method
    else:
        raise Exception("Please select sampling distribution method from {\'sp\', \'mp\', \'ex\', \'manual\'}")

    return


