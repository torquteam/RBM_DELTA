from scipy.optimize import root
import numpy as np
import time
import ctypes
import sys
import os
sys.path.append('/Users/marcsalinas/Desktop/RBM')
import functions as func
from scipy.integrate import simpson as simps
import matplotlib.pyplot as plt

current_directory = os.getcwd()
print("Current Working directory:", current_directory)

# Specify the nucleus
##################################################################
nucleus = 3
##################################################################

# Specify the number of proton and neutron states
nstates_n_list = [3,6,7,10,11,11,14,16,16,22]
nstates_p_list = [3,6,6,7, 10,11,11,11,13,16]
nstates_n = nstates_n_list[nucleus]
nstates_p = nstates_p_list[nucleus]

# Specify the number of protons and neutrons (for file specification purposes)
A_list = [16,40,48,68,90,100,116,132,144,208]
Z_list = [8, 20,20,28,40,50, 50, 50, 62, 82]
A = A_list[nucleus]
Z = Z_list[nucleus]

mNuc_mev = 939

# Load C functions shared library
library_path = os.path.join(current_directory, f"{A},{Z}/c_functions_greedy.so")
lib = ctypes.CDLL(library_path)

# Define the argument and return types of the functions
lib.c_function.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                                          np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                                          np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                                          np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # num_basis_states_wf
                                                          np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')]
lib.c_function.restype = None

lib.BA_function.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'), 
                            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # num_basis_states_wf
                            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')]
lib.BA_function.restype = ctypes.c_double

lib.Wkskin.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.Wkskin.restype = ctypes.c_double

lib.Rch.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.Rch.restype = ctypes.c_double

# Define a wrapper function that calls the C functions
def c_function_wrapper(x,params,num_basis_states_wf,num_basis_states_meson):
    z = np.zeros(np.sum(num_basis_states_wf)+np.sum(num_basis_states_meson)+nstates_n+nstates_p, dtype=np.double)
    lib.c_function(x, z, params,num_basis_states_wf,num_basis_states_meson)
    return z

def c_wrapper2(w,params,num_basis_states_wf,num_basis_states_meson):
    nstates_wf = nstates_n*2 + nstates_p*2
    n_meson = 5
    n_total = nstates_wf*6 + n_meson*7 + nstates_n + nstates_p
    x = np.zeros(n_total, dtype=np.double)
    count = 0
    for i in range(nstates_wf):
        for j in range(0,num_basis_states_wf[i]):   
            x[6*i+j] = w[count+j]
        count = count + num_basis_states_wf[i]
    for i in range(n_meson):
        for j in range(0,num_basis_states_meson[i]):
            x[6*nstates_wf+7*i+j] = w[count+j]
        count = count + num_basis_states_meson[i]
    for i in range(nstates_n+nstates_p):
        x[6*nstates_wf+7*n_meson+i] = w[count+i]
    return c_function_wrapper(x,params,num_basis_states_wf,num_basis_states_meson)

def compute_jacobian_wrapper(x,params):
    jac = np.empty((len(x), len(x)), dtype=np.double)
    lib.compute_jacobian(x, jac.reshape(-1),params)
    return jac.T

def BA_function(x,params,num_basis_states_wf,num_basis_states_meson):
    nstates_wf = nstates_n*2 + nstates_p*2
    n_meson = 5
    n_total = nstates_wf*6 + n_meson*7 + nstates_n + nstates_p
    w = np.zeros(n_total, dtype=np.double)
    count = 0
    for i in range(nstates_wf):
        for j in range(0,num_basis_states_wf[i]):   
            w[6*i+j] = x[count+j]
        count = count + num_basis_states_wf[i]
    for i in range(n_meson):
        for j in range(0,num_basis_states_meson[i]):
            w[6*nstates_wf+7*i+j] = x[count+j]
        count = count + num_basis_states_meson[i]
    for i in range(nstates_n+nstates_p):
        w[6*nstates_wf+7*n_meson+i] = x[count+i]
    BA = lib.BA_function(w,params,num_basis_states_wf,num_basis_states_meson)
    return BA

def Wkskin(x):
    res = lib.Wkskin(x)
    return res

def Rch(x):
    res = lib.Rch(x)
    return res

# import Data
##################################################
dir = f"{A},{Z}/{A},{Z},Data"
num_basis_states_f, num_basis_states_g, num_basis_states_c, num_basis_states_d, num_basis_meson = func.import_basis_numbers(A,Z)

num_basis_states_wf = np.array(num_basis_states_f+num_basis_states_g+num_basis_states_c+num_basis_states_d, dtype=np.int32)
num_basis_states_meson = np.array(num_basis_meson, dtype=np.int32)
print(np.sum(num_basis_states_wf)+np.sum(num_basis_meson),"basis states")

param_set = func.load_data("validation_DINO_RBM.txt")
actual_results = func.load_data(dir + f"/{A},{Z}Observables.txt")
n_samples = len(actual_results)
print(n_samples,"samples")

# Initial guess setup
##################################################
energy_guess = 60.0
initial_guess_array = func.initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson[0],num_basis_meson[1],num_basis_meson[2],num_basis_meson[3],num_basis_meson[4],[energy_guess]*nstates_n,[energy_guess]*nstates_p)

# Nonlinear solve
##############################################################################
start_time = time.time()
errBA = 0
errRch = 0
errWk = 0
for i in range(n_samples):
    params = param_set[i,:]
    #params = [4.91160767e+02, 1.05813435e+02, 1.82548004e+02, 1.67125801e+02, 1.10939990e+02, 3.36134447e+00, -1.20853287e-03, 2.80633218e-02, 6.69791611e-03]
    params_array = np.array(params, dtype=np.double)

    solution = root(c_wrapper2, x0=initial_guess_array, args=(params_array,num_basis_states_wf,num_basis_states_meson,), jac=None, method='hybr',options={'col_deriv': 1, 'xtol': 1e-8})
    BA_mev = (BA_function(solution.x,params_array,num_basis_states_wf,num_basis_states_meson)*13.269584506383948 - 0.75*41.0*A**(-1.0/3.0))/A - 939
    #Rcharge = Rch(solution.x)
    #FchFwk = Wkskin(solution.x)
    #print(f"Rch = {Rcharge}" )
    #print(f"Fch - Fwk = {FchFwk}")

    # compute the average err of each observable
    errBA = errBA + abs(actual_results[i][0] - BA_mev)
    print("Binding energy = ", BA_mev, abs(actual_results[i][0] - BA_mev)) #18 0.006
    #errRch = errRch + abs(actual_results[i%50][1] - Rcharge)
    #errWk = errWk + abs(actual_results[i%50][2] - FchFwk)

end_time = time.time()
print("SVD took:{:.4f} seconds".format(end_time - start_time))
print("{:.4f}s/run".format((end_time - start_time)/n_samples))

errBA = errBA/n_samples
errRch = errRch/n_samples
errWk = errWk/n_samples
print(errBA, errRch, errWk)

########################################################
####################################################################################

finalerr = errBA
num_basis_states_base_f, num_basis_states_base_g, num_basis_states_base_c, num_basis_states_base_d, num_basis_meson = func.import_basis_numbers(A,Z)
real_en_n = np.loadtxt(f"{A},{Z}/{A},{Z}energies_n.txt")
real_en_p = np.loadtxt(f"{A},{Z}/{A},{Z}energies_p.txt")

n_labels, state_file_n = func.load_spectrum( dir + "/neutron_spectrum.txt")
p_labels, state_file_p = func.load_spectrum(dir + "/proton_spectrum.txt")

file_pattern = dir + "/neutron/f_wave/val_{}.txt"
f_files = [file_pattern.format(n_labels[i]) for i in range(nstates_n)]
f_fields = [func.load_data(f_file) for f_file in f_files]

file_pattern = dir + "/proton/c_wave/val_{}.txt"
c_files = [file_pattern.format(p_labels[i]) for i in range(nstates_p)]
c_fields = [func.load_data(c_file) for c_file in c_files]


r_vec = func.load_data(dir + "/rvec.txt")[1:]
n_steps = 2

exp = np.loadtxt("exp_data.txt")
thresh = -0.001*exp[nucleus][2]
flag = False
while(finalerr>thresh):

    # initialize errors
    errBA = 0.0
    err_n = [0.0]*nstates_n
    err_p = [0.0]*nstates_p
    wf_error_n = [0.0]*nstates_n
    wf_error_p = [0.0]*nstates_p

    # set the number of basis states and set guess
    num_basis_states_wf = np.array(num_basis_states_f+num_basis_states_g+num_basis_states_c+num_basis_states_d, dtype=np.int32)
    initial_guess_array = func.initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson[0],num_basis_meson[1],num_basis_meson[2],num_basis_meson[3],num_basis_meson[4],[energy_guess]*nstates_n,[energy_guess]*nstates_p)
    
    # compute the RBM for nsamples
    start_time = time.time()
    for j in range(n_samples):
        params = param_set[j,:]
        params_array = np.array(params, dtype=np.double)
        solution = root(c_wrapper2, x0=initial_guess_array, args=(params_array,num_basis_states_wf,num_basis_states_meson,), jac=None, method='hybr',options={'col_deriv': 1, 'xtol': 1e-8})
        BA_mev = (BA_function(solution.x,params_array,num_basis_states_wf,num_basis_states_meson)*13.269584506383948 - 0.75*41.0*A**(-1.0/3.0))/A - 939
        
        # get the error in the energies
        for k in range(nstates_n):
            err_n[k] = err_n[k] + abs(solution.x[k-nstates_n-nstates_p]-real_en_n[j,k])
        for k in range(nstates_p):
            err_p[k] = err_p[k] + abs(solution.x[k-nstates_p]-real_en_p[j,k])
        errBA = errBA + abs(actual_results[j][0] - BA_mev)
    end_time = time.time()
    errBA = errBA/n_samples
    err_n = [element/n_samples for element in err_n]
    err_p = [element/n_samples for element in err_p]
    greedy_basis = num_basis_states_f+num_basis_states_c
    err = err_n + err_p
    func.greedy(err,greedy_basis,n_steps)
    num_basis_states_f = greedy_basis[:nstates_n]
    num_basis_states_g = greedy_basis[:nstates_n]
    num_basis_states_c = greedy_basis[nstates_n:]
    num_basis_states_d = greedy_basis[nstates_n:]
    print(errBA,"{:.4f} s".format(end_time - start_time),np.sum(num_basis_states_f)+np.sum(num_basis_states_c))
    basis_fg = '  '.join(map(str, num_basis_states_f))
    basis_cd = '  '.join(map(str, num_basis_states_c))
    print(basis_fg)
    print(basis_cd)

    # set the number of basis states and set guess
    num_basis_states_wf = np.array(num_basis_states_f+num_basis_states_g+num_basis_states_c+num_basis_states_d, dtype=np.int32)
    initial_guess_array = func.initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson[0],num_basis_meson[1],num_basis_meson[2],num_basis_meson[3],num_basis_meson[4],[energy_guess]*nstates_n,[energy_guess]*nstates_p)
    
    errBA = 0.0
    start_time = time.time()
    for j in range(n_samples):
        params = param_set[j,:]
        params_array = np.array(params, dtype=np.double)
        solution = root(c_wrapper2, x0=initial_guess_array, args=(params_array,num_basis_states_wf,num_basis_states_meson,), jac=None, method='hybr',options={'col_deriv': 1, 'xtol': 1e-8})
        BA_mev = (BA_function(solution.x,params_array,num_basis_states_wf,num_basis_states_meson)*13.269584506383948 - 0.75*41.0*A**(-1.0/3.0))/A - 939
        
        # Reconstruct the wave functions
        f_basis, c_basis = func.get_wf_basis_states(A,Z,nstates_n,nstates_p,num_basis_states_f,num_basis_states_c)
        f_coeff = np.array(func.pad([[solution.x[int(np.sum(num_basis_states_f[:j])) + i] for i in range(num_basis_states_f[j])] for j in range(nstates_n)]))
        c_coeff = np.array(func.pad([[solution.x[sum(num_basis_states_f) + sum(num_basis_states_g) + int(np.sum(num_basis_states_c[:j])) + i] for i in range(num_basis_states_c[j])] for j in range(nstates_p)]))
        f_fields_approx, c_fields_approx = func.compute_fields(f_coeff, c_coeff, nstates_n, nstates_p, f_basis, c_basis)

        # get the error in wave functions
        # this is where i left off. need to take difference of wf and integrate
        wf_n = [(f_fields_approx[:,i] - f_fields[i][:,j])**2 for i in range(nstates_n)]
        wf_p = [(c_fields_approx[:,i] - c_fields[i][:,j])**2 for i in range(nstates_p)]
        L2_n = [simps(wf_n[i],x=r_vec,axis=0) for i in range(nstates_n)]
        L2_p = [simps(wf_p[i],x=r_vec,axis=0) for i in range(nstates_p)]
        wf_error_n = [wf_error_n[i] + L2_n[i] for i in range(nstates_n)]
        wf_error_p = [wf_error_p[i] + L2_p[i] for i in range(nstates_p)]
        errBA = errBA + abs(actual_results[j][0] - BA_mev)
    end_time = time.time()

    errBA = errBA/n_samples
    wf_error_n = [element/n_samples for element in wf_error_n]
    wf_error_p = [element/n_samples for element in wf_error_p]
    greedy_basis = num_basis_states_f+num_basis_states_c
    wf_error = wf_error_n + wf_error_p
    flag = func.greedy(wf_error,greedy_basis,n_steps)
    num_basis_states_f = greedy_basis[:nstates_n]
    num_basis_states_g = greedy_basis[:nstates_n]
    num_basis_states_c = greedy_basis[nstates_n:]
    num_basis_states_d = greedy_basis[nstates_n:]
    print(errBA,"{:.4f} s".format(end_time - start_time),np.sum(num_basis_states_f)+np.sum(num_basis_states_c))
    basis_fg = '  '.join(map(str, num_basis_states_f))
    basis_cd = '  '.join(map(str, num_basis_states_c))
    print(basis_fg)
    print(basis_cd)

    finalerr = errBA
    if (flag == True):
        break