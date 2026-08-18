import numpy as np

#Measure functions:

def calc_Phi(config):
    """
    Calculates the total field (sum) of a single lattice configuration.
    Matches the explicit definition of capital Phi in Durr & Kiel.

    Args:
        config (numpy.ndarray): The 2D lattice field configuration.
        
    Returns:
        float: The calculated Phi (sum).
    """
    return np.sum(config)


#Jackknife analysis:

def jackknife_error(data_array, bin_size, observable_func):
    """
    Performs Jackknife method to estimate the statistical error of an observable
    while accounting for autocorrelation in MCMC data.
    
    Args:
        data_array (numpy.ndarray): The equilibrated (thermalized) history of the HMC simulation.
        bin_size (int): The size of the blocks to group data into.
        observable_func (callable): The function used to calculate the observable.
        
    Returns:
        float: The Jackknife error estimate (returns NaN if bin_size is too large).
    """

    num_bins = int(data_array.size/bin_size)

    #We need to truncate the data so it matches perfectly in the number of bins
    truncated_data = data_array[:num_bins * bin_size]

    #If number of bins is less than 2, than bin_size is too big so return nan (not a number)
    if num_bins < 2:
        return np.nan


    #Calcualte the full measured value
    full_measurement = observable_func(truncated_data)

    #Now, we need the measured value with ith bin removed (I call this jack_observable)
    jack_observables = np.zeros(num_bins)
    
    for i in range(num_bins):

        left_side = truncated_data[:i*bin_size]
        right_side = truncated_data[(i+1)*bin_size:]

        #Combine to get data with ith bin removed
        combined_data = np.concatenate((left_side,right_side))

        jack_observables[i] = observable_func(combined_data)
    

    jack_mean = np.mean(jack_observables)
    jack_error = np.sqrt( (num_bins-1)/num_bins * np.sum((jack_observables - jack_mean)**2) ) 

    return full_measurement, jack_error

#Autocorrelation analysis:

def autocorrelation(data_array, max_lag):
    """
    Calculates the normalized autocorrelation function rho(t) for an observable

    Args:
        data_array (numpy.ndarray): The equilibrated (thermalized) data history of the HMC simulation.
        max_lag (int): Maximum lag time of trajectories to examine.
        
    Returns:
        rho(t) (numpy.ndarray): the normalized autocorrelation function.
    
    """

    N = len(data_array)
    mean = np.mean(data_array)
    var = np.var(data_array)
    
    rho = np.zeros(max_lag)
    for t in range(max_lag):
        #Calculate covariance between data_array[i] and data_array[i+t]
        cov = np.sum((data_array[:N-t] - mean) * (data_array[t:] - mean)) / (N - t)
        rho[t] = cov / var
    return rho

#Observables

def calc_chi(Phi_array):
    """
    Calculates the extensive magnetic susceptibility. Takes an array of total field sums.
    
    Args:
        Phi_array (numpy.ndarray): The 1D field sum array from an HMC run.
        
    Returns:
        float: The calculated extensive magnetic susceptibility (variance).
    """

    return np.mean(Phi_array**2) - np.mean(np.abs(Phi_array))**2

#Spatial Correlation

def measure_correlations(phi):
    """
    Measures the local magnetization and spatial products for r=0 to N/2.
    Returns a 1D array of length (N//2 + 2).
    - Index 0      : m_local = mean(phi)
    - Index 1 to end: spatial products for r=0 up to r=N/2
    """
    N = phi.shape[0]
    max_r = N // 2
    
    # Create an output array. For N=64, max_r=32. 
    # We need 1 spot for m_local + 33 spots for r=0 to 32. Total = 34.
    out = np.zeros(max_r + 2)
    
    # 1. Store the local magnetization at index 0
    out[0] = np.mean(phi)
    
    # 2. Calculate and store the spatial products
    for r in range(max_r + 1):
        # Shift along X
        shifted_x = np.roll(phi, r, axis=0)
        spatial_avg_x = np.mean(phi * shifted_x)
        
        # Shift along Y
        shifted_y = np.roll(phi, r, axis=1)
        spatial_avg_y = np.mean(phi * shifted_y)
        
        # Average the directions and store at index r+1
        out[r + 1] = 0.5 * (spatial_avg_x + spatial_avg_y)
        
    return out