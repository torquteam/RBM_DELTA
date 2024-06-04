from scipy.optimize import root
import numpy as np
import time
import ctypes
import sys
sys.path.append('/Users/marcsalinas/Desktop/RBM')
import functions as func

# Specify the nucleus
##################################################################
nucleus = 7
##################################################################

# Specify the number of proton and neutron states
nstates_n_list = [3,6,7,10,11,11,14,16,16,22]
nstates_p_list = [3,6,6,7,10,11,11,11,13,16]
nstates_n = nstates_n_list[nucleus]
nstates_p = nstates_p_list[nucleus]

# Specify the number of protons and neutrons (for file specification purposes)
A_list = [16,40,48,68,90,100,116,132,144,208]
Z_list = [8,20,20,28,40,50,50,50,62,82]
A = A_list[nucleus]
Z = Z_list[nucleus]

mNuc_mev = 939

# Load C functions shared library
lib = ctypes.CDLL(f'./{A},{Z}/c_functions.so')

# Define the argument and return types of the functions
lib.c_function.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                                          np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                                          np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.c_function.restype = None

lib.compute_jacobian.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                             np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
                                             np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.compute_jacobian.restype = None

lib.BA_function.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.BA_function.restype = ctypes.c_double

lib.Rch.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.Rch.restype = ctypes.c_double

lib.Wkskin.argtypes = [np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
lib.Wkskin.restype = ctypes.c_double

# Define a wrapper function that calls the C functions
def c_function_wrapper(x,params):
    y = np.empty_like(x, dtype=np.double)
    lib.c_function(x, y, params)
    return y

def compute_jacobian_wrapper(x,params):
    jac = np.empty((len(x), len(x)), dtype=np.double)
    lib.compute_jacobian(x, jac.reshape(-1),params)
    return jac.T

def BA_function(x,params):
    BA = lib.BA_function(x,params)
    return BA

def Rch(x):
    res = lib.Rch(x)
    return res

def Wkskin(x):
    res = lib.Wkskin(x)
    return res

# import Data
##################################################
dir = f"{A},{Z}/{A},{Z},Data"
num_basis_states_f, num_basis_states_g, num_basis_states_c, num_basis_states_d, num_basis_meson = func.import_basis_numbers(A,Z)
param_set = func.load_data("validation_DINO_RBM.txt")
actual_results = func.load_data(dir + f"/{A},{Z}Observables.txt")

# Initial guess setup
##################################################
energy_guess = 40.0
en_n = [energy_guess]*nstates_n
en_p = [energy_guess]*nstates_p
initial_guess_array = func.initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson[0],num_basis_meson[1],num_basis_meson[2],num_basis_meson[3],num_basis_meson[4],en_n,en_p)

# Nonlinear solve
##############################################################################
start_time = time.time()
errBA = 0
errRch = 0
errWk = 0
nruns = len(param_set)
for i in range(1):
    params = param_set[i,:]
    #params = [4.89445423e+02,  1.02393497e+02,  1.75357860e+02,  5.56211986e+02,7.32658996e+02,  3.84365875e+00, -6.73118413e-03,  2.13691817e-02,1.83094957e-03]
    params = [4.92341077e+02, 1.07804520e+02, 1.86997023e+02, 1.48354499e+02, 8.98367209e+01, 3.05434252e+00, 1.98682552e-03, 3.20014208e-02, 6.57430595e-03]
    params_array = np.array(params, dtype=np.double)

    solution = root(c_function_wrapper, x0=initial_guess_array, args=(params_array,), jac=compute_jacobian_wrapper, method='hybr',options={'col_deriv': 1, 'xtol': 1e-8})
    BA_mev = (BA_function(solution.x,params_array)*13.269584506383948 - 0.75*41.0*A**(-1.0/3.0))/A - 939
    Rcharge = Rch(solution.x)
    FchFwk = Wkskin(solution.x)
    print("Binding energy = ", BA_mev)
    print(f"Rch = {Rcharge}" )
    print(FchFwk)
    #print(solution.x)

    # compute the average err of each observable
    errBA = errBA + abs(actual_results[i][0] - BA_mev)
    errRch = errRch + abs(actual_results[i][1] - Rcharge)
    #errWK = errWk + abs(actual_results[i%50][2] - FchFwk)
    
end_time = time.time()
print("RBM took:{:.4f} seconds".format(end_time - start_time))
print("{:.4f}s/run".format((end_time - start_time)/nruns))

errBA = errBA/nruns
errRch = errRch/nruns
print(errBA, errRch)