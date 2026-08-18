import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def S(phi, mu_sq, lam):
    """
    Calculates the total action of the 2D scalar phi^4 theory.
    
    Args:
        phi (numpy.ndarray): The 2D lattice field configuration.
        mu_sq (float): The bare mass squared parameter.
        lam (float): The coupling constant (lambda).
        
    Returns:
        float: The total calculated action summed across all lattice sites.
    """

    #Mass and interaction terms
    mass_term = (2.0 + 0.5 * mu_sq) * phi**2
    interaction_term = (lam / 4.0) * phi**4
    
    #Calculate the kinetic term. 
    #We roll the array by 1 on the x-axis (axis=0) and the y-axis (axis=1).
    kinetic = -phi * (np.roll(phi, 1, axis=0) + np.roll(phi, 1, axis=1))
    
    #The total action is the sum of every lattice site
    return np.sum(kinetic + mass_term + interaction_term)

def dS(phi, mu_sq, lam):
    """
    Calculates the derivative of the action for the Leapfrog integrator.
    
    Args:
        phi (numpy.ndarray): The current 2D lattice field configuration.
        mu_sq (float): The bare mass squared parameter.
        lam (float): The coupling constant (lambda).
        
    Returns:
        numpy.ndarray: A 2D array of the forces at each lattice site.
    """

    #To find the derivative (the force), we need to sum over nearest neighbors from interaction term
    neighbors = (np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) + 
                 np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1))
    
    #Return the full force
    return -neighbors + (4.0 + mu_sq) * phi + lam * phi**3



def HMC(S,dS,n_traj,initial_config,mu_sq,lam,step_size,num_steps,measure_func,verbose = False, min_int = 2.0):
    """
    Executes the Hybrid Monte Carlo (HMC) algorithm for a 2D lattice.
    
    Args:
        S (callable): Function to calculate the action.
        dS (callable): Function to calculate the derivative of the action.
        n_traj (int): Number of HMC trajectories to generate.
        initial_config (numpy.ndarray): The starting 2D lattice configuration.
        mu_sq (float): The bare mass squared parameter.
        lam (float): The coupling constant (lambda).
        step_size (float): The leapfrog integration step size (dt).
        num_steps (int): Number of leapfrog steps per trajectory.
        measure_func (callable): Observable function to measure at each step.
        
    Returns:
        tuple:
            results (numpy.ndarray): History of the measured observable.
            exp_dh_list (numpy.ndarray): History of exp(-dH) to verify the Creutz equality.
            current_phi (numpy.ndarray): The final lattice field configuration.
            acceptance (float): The overall Metropolis acceptance rate percentage.
    """

    #Declare intial values and HMC paramters
    N = initial_config.shape[0]
    current_phi = np.copy(initial_config)
    
    sample_measurement = np.array(measure_func(initial_config))

    results = np.zeros((n_traj,) + sample_measurement.shape)

    exp_dh_list = np.zeros(n_traj)

    num_acc = 0 #Number of Metropolis accepted

    if verbose:
        print(f"""
    ========================================
        HMC Engine Initialization        
    ========================================
    Lattice Size (N)  : {N}x{N}
    Trajectories      : {n_traj}
    Bare Mass Sq      : {mu_sq}
    Coupling (Lambda) : {lam}
    Step Size (dt)    : {step_size}
    Leapfrog Steps    : {num_steps}
    ========================================
    """)
    for i in tqdm(range(n_traj), disable=not verbose, mininterval= min_int, bar_format="HMC Engine |{bar}| {percentage:3.0f}% Complete"):

        #Generate all conjugate momenta
        p_int = np.random.normal(loc=0,scale=1,size=(N,N))

        #Calulate intial Hamiltonian
        H_i = np.sum(0.5*p_int**2) + S(current_phi,mu_sq,lam)
        
        phi_new = np.copy(current_phi)
        p_new = np.copy(p_int)

        p_new = p_new - (step_size/2)*dS(phi_new,mu_sq,lam)

        #Leap Frog Integrator
        for j in range(num_steps):

            phi_new = phi_new + step_size*p_new

            if j != num_steps -1:
                p_new = p_new - (step_size)*dS(phi_new,mu_sq,lam)
            else:
                p_new = p_new - (step_size/2)*dS(phi_new,mu_sq,lam)


        #Calculator Final Hamiltonian
        H_f = np.sum(0.5*p_new**2) + S(phi_new,mu_sq,lam)


        dH = H_f - H_i
        r = np.random.rand()

        #Do Metropolis Check
        if r <= np.exp(-dH):
            current_phi = phi_new
            num_acc +=1
        
        results[i] = measure_func(current_phi)
        exp_dh_list[i] = np.exp(-dH)

    acceptance = num_acc/n_traj * 100
    if verbose:
        print(f"HMC Acceptance: {acceptance:.2f}%")

    return results, exp_dh_list, current_phi, acceptance

#Now, for Fourier Acceleration


def get_inverse_kernel(N, kappa):
    """
    Builds the momentum-space Kernel and the momentum-space Kernel inverse.
    """
    k = np.arange(N)
    k1, k2 = np.meshgrid(k, k, indexing='ij')
    
    K_tilde = 4.0 * kappa * (np.sin(np.pi * k1 / N)**2 + np.sin(np.pi * k2 / N)**2) + (1 - kappa)
    
    # The Acceleration Grid used to update the field
    K_tilde_inverse = 1.0 / K_tilde
    
    return K_tilde, K_tilde_inverse

def generate_pi_tilde(N,A):

    pi_tilde = np.zeros((N,N), dtype = 'complex')

    i, j = np.indices((N, N))

    base = (j <= N/2)
    bad_spots = (i>N/2) & ( (j==0) | (j==N/2))
    fundamental_domain = base & ~ bad_spots

    corners = (i == (N-i)%N) & (j == (N-j)%N)

    std_dev = N/np.sqrt(A)

    R= np.random.normal(0, std_dev / np.sqrt(2))
    I= np.random.normal(0, std_dev / np.sqrt(2))

    R[corners] = np.random.normal(0,std_dev[corners])
    I[corners] = 0

    pi_tilde[fundamental_domain] = R[fundamental_domain] + 1j*I[fundamental_domain]

    mirror_i = (N - i[fundamental_domain]) % N
    mirror_j = (N - j[fundamental_domain]) % N

    pi_tilde[mirror_i, mirror_j] = R[fundamental_domain] - 1j * I[fundamental_domain]

    return pi_tilde



def HMC_FA(S, dS, n_traj, initial_config, mu_sq, lam, step_size, num_steps, measure_func, kappa, verbose=False, min_int = 2.0):

    # Declare initial values and HMC parameters
    N = initial_config.shape[0]
    current_phi = np.copy(initial_config)
    
    #Take sample measurement to get the measure functions output shape
    sample_measurement = np.array(measure_func(initial_config))

    #This now insures that the results numpy array will be structered correctly
    results = np.zeros((n_traj,) + sample_measurement.shape)

    exp_dh_list = np.zeros(n_traj)
    num_acc = 0

    #Neccesary information when debugging
    if verbose:
        print(f"""
    ========================================
       HMC_FA Engine Initialization         
    ========================================
    Lattice Size (N)  : {N}x{N}
    Trajectories      : {n_traj}
    Bare Mass Sq      : {mu_sq}
    Coupling (Lambda) : {lam}
    Accel Param. (k) : {kappa}
    Step Size (dt)    : {step_size}
    Leapfrog Steps    : {num_steps}
    ========================================
    """)

    # Get Kernel and its inverse in momentum space.
    K_tilde, A = get_inverse_kernel(N, kappa)

    for i in tqdm(range(n_traj), disable=not verbose, mininterval= min_int, bar_format="HMC_FA Engine |{bar}| {percentage:3.0f}% Complete"):

        #Generate auxillary momenta in momentum space
        p_tilde = generate_pi_tilde(N,A)

        #Transform to real space
        p_int = np.real(np.fft.ifftn(p_tilde))

        H_i = (0.5/N**2) * np.sum(A * np.abs(p_tilde)**2) + S(current_phi, mu_sq, lam)
        
        phi_new = np.copy(current_phi)
        p_new = np.copy(p_int)

        #Accelerated Leapfrog Integrator

        #Half-step for momentum in real space
        p_new = p_new - (step_size / 2.0) * dS(phi_new, mu_sq, lam)
        
        for j in range(num_steps):

            #Transform to momentum space
            phi_tilde = np.fft.fftn(phi_new)
            p_tilde = np.fft.fftn(p_new)
   
            #Update field using inverse Kernel
            phi_tilde = phi_tilde + step_size * A * p_tilde

            #Bring field back to real space
            phi_new = np.real(np.fft.ifftn(phi_tilde))

            if j != num_steps-1:
                #Full-step for momentum in real space
                p_new = p_new - (step_size) * dS(phi_new, mu_sq, lam)

            else:
                #Half-step for momentum in real space
                p_new = p_new - (step_size / 2.0) * dS(phi_new, mu_sq, lam)


        H_f = (0.5/N**2) * np.sum(A * np.abs(np.fft.fftn(p_new))**2) + S(phi_new, mu_sq, lam)

        dH = H_f - H_i
        r = np.random.rand()

        #Metropolis Check
        if r <= np.exp(-dH):
            current_phi = phi_new
            num_acc += 1
        
        results[i] = measure_func(current_phi)
        exp_dh_list[i] = np.exp(-dH)

    acceptance = num_acc / n_traj * 100
    if verbose:
        print(f"HMC_FA Acceptance: {acceptance:.2f}%")

    return results, exp_dh_list, current_phi, acceptance