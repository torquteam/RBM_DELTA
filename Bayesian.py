import numpy as np
import ctypes
import functions as func
import time
import bulk2params as trans
import multiprocessing
import functools
import operator
import os
from sys import exit
import math

current_directory = os.getcwd()
print("Current Working directory:", current_directory)

# list of nuclei atomic masses and numbers (to be used in calibration)
A=[16,40,48,68,90,100,116,132,144,208]
Z=[8,20,20,28,40,50,50,50,62,82]

# import c functions for all nuclei (speeds up computation)
###############################################################################
###############################################################################
libraries = []

# Load libraries
for i in range(10):
    lib = ctypes.CDLL(f'./{A[i]},{Z[i]}/c_functions.so')
    libraries.append(lib)

# Set up argument types for all libraries
for lib in libraries:
    lib.c_function.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                               np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                               np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    lib.c_function.restype = None
    lib.compute_jacobian.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                      np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    lib.compute_jacobian.restype = None
    lib.BA_function.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'), 
                                np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    lib.BA_function.restype = ctypes.c_double

    lib.Wkskin.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    lib.Wkskin.restype = ctypes.c_double

    lib.Rch.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    lib.Rch.restype = ctypes.c_double


def c_function_wrapper(lib):
    def wrapper(x, params):
        y = np.empty_like(x, dtype=np.double)
        lib.c_function(x, y, params)
        return y
    return wrapper

def jacobian_wrapper(lib):
    def wrapper(x, params):
        jac = np.empty((len(x), len(x)), dtype=np.double)
        lib.compute_jacobian(x, jac.reshape(-1), params)
        return jac.T
    return wrapper

def BA_wrapper(lib):
    def wrapper(x,params):
        BA = lib.BA_function(x,params)
        return BA
    return wrapper

def Wkskin_wrapper(lib):
    def wrapper(x):
        res = lib.Wkskin(x)
        return res
    return wrapper

def Rch_wrapper(lib):
    def wrapper(x):
        res = lib.Rch(x)
        return res
    return wrapper


# Create wrapper functions for all libraries
wrapper_functions_list = []
jacobian_wrappers_list = []
BA_wrappers_list = []
Wkskin_wrappers_list = []
Rch_wrappers_list = []
for lib in libraries:
    wrapper_functions_list.append(c_function_wrapper(lib))
    jacobian_wrappers_list.append(jacobian_wrapper(lib))
    BA_wrappers_list.append(BA_wrapper(lib))
    Wkskin_wrappers_list.append(Wkskin_wrapper(lib))
    Rch_wrappers_list.append(Rch_wrapper(lib))
######################################################################################
######################################################################################

# define nstates for neutron and proton
nstates_n = [3,6,7,10,11,11,14,16,16,22]
nstates_p = [3,6,6,7,10,11,11,11,13,16]

# define the directories to retrieve the galerkin equations and basis numbers
dirs = []
for i in range(10):
    dir = f"{A[i]},{Z[i]}/{A[i]},{Z[i]},Data"
    dirs.append(dir)

# get number of basis states for each nuclei
num_basis_states_f_list = []
num_basis_states_g_list = []
num_basis_states_c_list = []
num_basis_states_d_list = []
num_basis_meson_list = []
for i in range(10):
    num_basis_states_f, num_basis_states_g, num_basis_states_c, num_basis_states_d, num_basis_meson = func.import_basis_numbers(A[i],Z[i])
    f_basis, g_basis, c_basis, d_basis, S_basis, V_basis, B_basis, A_basis = func.get_basis(A[i],Z[i],nstates_n[i],nstates_p[i])
    num_basis_states_f_list.append(num_basis_states_f)
    num_basis_states_g_list.append(num_basis_states_g)
    num_basis_states_c_list.append(num_basis_states_c)
    num_basis_states_d_list.append(num_basis_states_d)
    num_basis_meson_list.append(num_basis_meson)

# define the prior distribution
def compute_prior(bulks):
    prior_data = [-16.0, 0.15, 0.6, 230.0, 34.0, 80.0, 0.03, 500.0]
    prior_unct = [1.0  , 0.04, 0.1, 10.0 , 4.0 , 40.0, 0.03, 50.0 ]
    lkl= 1.0
    for i in range(len(prior_data)):
        lkl = lkl*np.exp(-0.5*(prior_data[i]-bulks[i])**2/prior_unct[i]**2)
    return lkl

# define the lkl function for a single nucleus
def compute_lkl(exp_data,BA_mev_th, Rch_th, FchFwk_th):
    lkl = np.exp(-0.5*(exp_data[0]-BA_mev_th)**2/exp_data[1]**2)
    if (exp_data[2] != -1):
        lkl = lkl*np.exp(-0.5*(exp_data[2]-Rch_th)**2/exp_data[3]**2)
    if (FchFwk_th != -1):
        #lkl = lkl*np.exp(-0.5*(exp_data[4]-FchFwk_th)**2/exp_data[5]**2)
        lkl = lkl*1.0
    return lkl

# define the total liklihood
def compute_nuclei_v2(num_nuclei,params,flag,n_energies_list,p_energies_list):
    lkl = 1.0
    if (flag == True):
        print("flagged")
        return 0.0
    for i in range(num_nuclei):
        BA_mev_th, Rch_th, Fch_Fwk_th, en_n, en_p = func.hartree_RBM(A[i],nstates_n[i],nstates_p[i],num_basis_states_f_list[i],num_basis_states_g_list[i],num_basis_states_c_list[i],num_basis_states_d_list[i],num_basis_meson_list[i],params,wrapper_functions_list[i],BA_wrappers_list[i],Rch_wrappers_list[i],Wkskin_wrappers_list[i],n_energies_list[i],p_energies_list[i],jac=jacobian_wrappers_list[i])
        n_energies_list[i] = en_n
        p_energies_list[i] = en_p
        #print(A[i],BA_mev_th,Rch_th)
        lkl = lkl*compute_lkl(exp_data[i,:],BA_mev_th,Rch_th,Fch_Fwk_th)
        if (abs(BA_mev_th) > 9.0 or abs(BA_mev_th) < 7.0):
            print("error: ",params)
            return 0.0

    return lkl

# define the metropolis hastings algorithm
def metropolis(post0, postp, bulks_0, bulks_p, acc_counts, index):
    # metroplis hastings step
    r = np.random.uniform(0,1)
    a = postp/post0
    if (a>1):
        a=1.0
    if (r <= a):
        post0 = postp
        for k in range(n_params):
            bulks_0[k] = bulks_p[k] # accept the proposed changes
        acc_counts[index]+=1   # count how many times the changes are accepted
    return post0

# adaptive MCMC method to reach acceptance rates
def adaptive_width(iter,n_check,arate,acc_counts,stds,agoal,index):
    if ((iter+1)%n_check == 0):
        arate[index] = acc_counts[index]/n_check
        acc_counts[index] = 0
        if (arate[index] < agoal):
            stds[index] = 0.9*stds[index]      # if acceptance rate is too low then decrease the range
        elif (arate[index] > agoal):
            stds[index] = 1.1*stds[index]      # if acceptance rate is too high then increase the range

# function to change a single parameter in MCMC
def param_change(n_params, bulks_0, bulks_p, stds, mw, mp, index):
    for k in range(n_params):
        bulks_p[k] = bulks_0[k] # copy old values
    bulks_p[index] = np.random.normal(bulks_0[index],stds[index])      # change one param
    params, flag = trans.get_parameters(bulks_p[0],bulks_p[1],bulks_p[2],bulks_p[3],bulks_p[4],bulks_p[5],bulks_p[6],bulks_p[7],mw,mp)
    return params, flag

# MCMC algorithm to be called in parallel
def MCMC_worker(args):
    iterations_burn, iterations_run, process_id, post0 = args
    np.random.seed(process_id)
    start_time = time.time()
    with open(f"burnin_out_{process_id}.txt", "w") as output_file:
        for i in range(iterations_burn):
            # one sweep
            for j in range(n_params):
                # get new proposed parameters
                params, flag = param_change(n_params,bulks_0,bulks_p,stds,mw,mp,j)

                # get the posterior
                prior = compute_prior(bulks_p)
                lklp = compute_nuclei_v2(n_nuclei,params,flag,energy_guess_n_list,energy_guess_p_list)
                postp = prior*lklp
                #print(prior,lklp,post0,postp)

                # metroplis hastings step
                post0 = metropolis(post0,postp,bulks_0,bulks_p,acc_counts,j)

                # rate monitoring to adjust the width of the sampling
                adaptive_width(i,n_check,arate,acc_counts,stds,agoal,j)
                if ((i+1)%n_check == 0):
                    print(arate,stds)

            # print MCMC sweep
            for k in range(n_params):
                print(f"{bulks_0[k]}",file=output_file, end='  ')
            print("",file=output_file)
            print(f"{i+1} completed")
    end_time = time.time()
    print("Burn in took:{:.4f} seconds".format((end_time-start_time)))
    print(stds)
    ##############################################
    # end of burn in

    # start MCMC runs
    with open(f"MCMC_{process_id}.txt", "w") as output_file:
        for i in range(iterations_run):
            # one sweep
            for j in range(n_params):
                # get new proposed parameters
                params, flag = param_change(n_params,bulks_0,bulks_p,stds,mw,mp,j)

                # get the posterior
                prior = compute_prior(bulks_p)
                lklp = compute_nuclei_v2(n_nuclei,params,flag,energy_guess_n_list,energy_guess_p_list)
                postp = prior*lklp

                # metroplis hastings step
                post0 = metropolis(post0,postp,bulks_0,bulks_p,acc_counts,j)
            
            # print MCMC sweep
            for k in range(n_params):
                print(f"{bulks_0[k]}",file=output_file, end='  ')
            print("",file=output_file)
            print(f"{i+1} completed")

def posterior_observables(posterior_file, n_energies_list, p_energies_list, n_params):
    posterior = np.loadtxt(posterior_file)
    n_samples = len(posterior)
    with open("Posterior.txt", "w") as output_file:
        for i in range(n_samples):
            bulks = posterior[i,:]
            params, flag = trans.get_parameters(bulks[0],bulks[1],bulks[2],bulks[3],bulks[4],bulks[5],bulks[6],bulks[7],mw,mp)
            for k in range(n_params):
                print(f"{bulks[k]}",file=output_file, end='  ')
            for j in range(n_nuclei):
                BA_mev_th, Rch_th, Fch_Fwk_th, en_n, en_p = func.hartree_RBM(A[j],nstates_n[j],nstates_p[j],num_basis_states_f_list[j],num_basis_states_g_list[j],num_basis_states_c_list[j],num_basis_states_d_list[j],num_basis_meson_list[j],params,wrapper_functions_list[j],BA_wrappers_list[j],Rch_wrappers_list[j],Wkskin_wrappers_list[j],n_energies_list[j],p_energies_list[j],jac=jacobian_wrappers_list[j])
                if (j == 2 or j == 9):
                    print(f"{BA_mev_th}  {Rch_th}  {Fch_Fwk_th}",file=output_file, end='  ')
                else:
                    print(f"{BA_mev_th}  {Rch_th}",file=output_file, end='  ')
            print("",file=output_file)

####################################################################################
####################################################################################
# import the experimental data and errors
exp_data_full = func.load_data("exp_data.txt")
exp_data = exp_data_full[:,2:]

# import state information (j, alpha, fill_frac, filetag)
state_info_n_list = []
state_info_p_list = []
energy_guess_p_list = []
energy_guess_n_list = []
for i in range(10):
    n_labels, state_file_n = func.load_spectrum( dirs[i] + "/neutron_spectrum.txt")
    p_labels, state_file_p = func.load_spectrum(dirs[i] + "/proton_spectrum.txt")
    energies_p = func.load_data(dirs[i] + "/proton/energies.txt")
    energies_n = func.load_data(dirs[i] + "/neutron/energies.txt")
    energy_guess_p = [50.0 for j in range(nstates_p[i])]
    energy_guess_n = [50.0 for j in range(nstates_n[i])]
    state_info_n = state_file_n[:,[3,4,5]]
    state_info_p = state_file_p[:,[3,4,5]]
    state_info_n_list.append(state_info_n)
    state_info_p_list.append(state_info_p)
    energy_guess_n_list.append(energy_guess_n)
    energy_guess_p_list.append(energy_guess_p)

# import start file
start_data = func.load_data("MCMC_startfile.txt")
bulks_0 = start_data[:,0]
#bulks_0 = [-16.17766627458662,  0.14918278902808854,  0.5404952093005428,  239.365472508626,  33.464072602438065,  60.318005807154094,  0.03605138672060682,  482.3147828259097]; 
stds = start_data[:,1]
bulks_p = np.empty_like(bulks_0)

#####################################################
# MCMC metrpolis hastings
#####################################################
nburnin = 10000
nruns = 62500
n_params = 8
mw = 782.5
mp = 763.0
n_nuclei = 10

# adaptive specifications for MCMC
acc_counts = [0]*n_params
n_check = 100
agoal = 0.3
arate = [0]*n_params

# initialize the starting point
'''
params, flag = trans.get_parameters(bulks_0[0],bulks_0[1],bulks_0[2],bulks_0[3],bulks_0[4],bulks_0[5],bulks_0[6],bulks_0[7],mw,mp)
#params, flag = trans.get_parameters(-16.20922384680473,  0.14874945980481055,  0.5637026253817242,  245.50998742105892,  32.68594158147973,  54.774894701933974,  0.028009904259910175,  497.52017153360873 ,mw,mp)
#params = np.array([496.939, 110.349, 187.695, 192.927, 3.26, -0.003551, 0.0235, 0.043377]) #garnet 1.5447531287323243e-16
#params = np.array([497.479, 108.0943, 183.7893, 80.4656, 3.0029, -0.000533, 0.0256, 0.000823]) #gold 4.2329717573521464e-17
prior = compute_prior(bulks_0)
lkl0 = compute_nuclei_v2(n_nuclei,params,flag,energy_guess_n_list,energy_guess_p_list)
post0 = prior*lkl0
print(lkl0)
print(params)

# parallelize the MCMC
n_processes = 4


inputs = [(nburnin, nruns, i, post0) for i in range(n_processes)]
with  multiprocessing.Pool(processes=n_processes) as pool:
    results = pool.map(MCMC_worker, inputs)
'''

#######################################################
# Posterior Observables
#######################################################
posterior_observables("MCMC_noskin.txt",energy_guess_n_list,energy_guess_p_list,n_params)

def error_sample(posterior_file, n_samples, n_params):
    posterior = np.loadtxt(posterior_file)
    n_rows, n_cols = np.shape(posterior)
    index_factor = math.floor(n_rows/n_samples)
    with open("RBM_samples.txt", "w") as output_file:
        for i in range(n_samples):
            bulks = posterior[(i+1)*index_factor-1,0:n_params]
            params, flag = trans.get_parameters(bulks[0],bulks[1],bulks[2],bulks[3],bulks[4],bulks[5],bulks[6],bulks[7],mw,mp)
            for k in range(n_params):
                print(f"{params[k]}",file=output_file, end='  ')
            for k in range(n_params,n_cols):
                print(f"{posterior[(i+1)*index_factor-1][k]}",file=output_file, end='  ')
            print("",file=output_file)

#error_sample("Posterior_noskin.txt",500,8)
