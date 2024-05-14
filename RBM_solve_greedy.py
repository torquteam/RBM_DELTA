from scipy.optimize import root
import numpy as np
import time
import ctypes
import sys
sys.path.append('/home/msals97/Desktop/RBM/RBM')
import functions as func

# Specify the nucleus
##################################################################
nucleus = 2
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
lib = ctypes.CDLL(f'./{A},{Z}/c_functions_greedy.so')

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
    n_total = nstates_wf*5 + n_meson*7 + nstates_n + nstates_p
    x = np.zeros(n_total, dtype=np.double)
    count = 0
    for i in range(nstates_wf):
        for j in range(0,num_basis_states_wf[i]):   
            x[5*i+j] = w[count+j]
        count = count + num_basis_states_wf[i]
    for i in range(n_meson):
        for j in range(0,num_basis_states_meson[i]):
            x[5*nstates_wf+7*i+j] = w[count+j]
        count = count + num_basis_states_meson[i]
    for i in range(nstates_n+nstates_p):
        x[5*nstates_wf+7*n_meson+i] = w[count+i]
    return c_function_wrapper(x,params,num_basis_states_wf,num_basis_states_meson)

def compute_jacobian_wrapper(x,params):
    jac = np.empty((len(x), len(x)), dtype=np.double)
    lib.compute_jacobian(x, jac.reshape(-1),params)
    return jac.T

def BA_function(x,params,num_basis_states_wf,num_basis_states_meson):
    nstates_wf = nstates_n*2 + nstates_p*2
    n_meson = 5
    n_total = nstates_wf*5 + n_meson*7 + nstates_n + nstates_p
    w = np.zeros(n_total, dtype=np.double)
    count = 0
    for i in range(nstates_wf):
        for j in range(0,num_basis_states_wf[i]):   
            w[5*i+j] = x[count+j]
        count = count + num_basis_states_wf[i]
    for i in range(n_meson):
        for j in range(0,num_basis_states_meson[i]):
            w[5*nstates_wf+7*i+j] = x[count+j]
        count = count + num_basis_states_meson[i]
    for i in range(nstates_n+nstates_p):
        w[5*nstates_wf+7*n_meson+i] = x[count+i]
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
print(np.sum(num_basis_states_wf)+np.sum(num_basis_meson))

param_set = func.load_data("param_sets_DINO.txt")
actual_results = func.load_data(dir + f"/{A},{Z}Observables.txt")
n_samples = len(actual_results)
print(n_samples)

# Initial guess setup
##################################################
initial_guess_array = func.initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson[0],num_basis_meson[1],num_basis_meson[2],num_basis_meson[3],num_basis_meson[4],[70]*nstates_n,[70]*nstates_p)

# Nonlinear solve
##############################################################################
start_time = time.time()
errBA = 0
errRch = 0
errWk = 0
for i in range(n_samples):
    params = param_set[i,:]
    params_array = np.array(params, dtype=np.double)

    solution = root(c_wrapper2, x0=initial_guess_array, args=(params_array,num_basis_states_wf,num_basis_states_meson,), jac=None, method='hybr',options={'col_deriv': 1, 'xtol': 1e-8})
    BA_mev = (BA_function(solution.x,params,num_basis_states_wf,num_basis_states_meson)*13.269584506383948 - 0.75*41.0*A**(-1.0/3.0))/A - 939
    #print(solution.x)
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

'''
import matplotlib.pyplot as plt
rvec = np.loadtxt("208,82/208,82,Data/rvec.txt")
actual = np.loadtxt("208,82/208,82,Data/neutron/g_wave/state1d3;2.txt")
#wave = np.loadtxt("208,82/High_fidelity_sol/Ap.txt")
meson = np.loadtxt("208,82/High_fidelity_sol/meson_fields.txt")/13.269584506383948
plt.plot(rvec,actual,ls='dashed') #
#plt.plot(rvec,wave[:,1:]) #
#plt.plot(rvec,meson[:,1])
plt.show()
'''


def greedy(err, basis, N):
    flag = False  # Initialize flag
    count = 0
    while(count<N):
        max_index = np.argmax(err)
        if basis[max_index] == 5:
            err[max_index] = -100  # Exclude the current maximum from further consideration
        else:
            basis[max_index] += 1
            count += 1
            err[max_index] = -100  # Exclude the current maximum from further consideration
    
    if all(val == 5 for val in basis):
        flag = True
    return flag

def import_greedy_config(file_path):
    try:
        # Open the file in read mode
        with open(file_path, "r") as file:
            # Read the contents of the file
            data = file.read()

        # Split the data into lines
        lines = data.split('\n')

        # Initialize an empty list to hold the list of lists
        result = []

        # Iterate over each line in the input data
        for line in lines:
            # Split the line into individual numbers
            numbers = line.split()
            # Convert each number to an integer and append to the list
            result.append([int(num) for num in numbers])

        return result

    except FileNotFoundError:
        print("File not found.")
        return None

###############################################################
real_en_n = np.loadtxt(f"{A},{Z}/{A},{Z}energies_n.txt")
real_en_p = np.loadtxt(f"{A},{Z}/{A},{Z}energies_p.txt")
greedy_add = import_greedy_config("/home/msals97/Desktop/RBM/RBM/greedy_configuration.txt")
improv_metric = np.zeros(len(greedy_add))   # keeps track of average error reduction per/basis
finalerr = errBA
num_basis_states_base_f, num_basis_states_base_g, num_basis_states_base_c, num_basis_states_base_d, num_basis_meson = func.import_basis_numbers(A,Z)

while(finalerr>0.01):
    BAerr_array = [0.0]*len(greedy_add)
    test_improv = 0.0
    for s in range(len(greedy_add)):
        num_basis_states_f = num_basis_states_base_f
        num_basis_states_g = num_basis_states_base_g
        num_basis_states_c = num_basis_states_base_c
        num_basis_states_d = num_basis_states_base_d
        for i in range(len(greedy_add[s])):
            err_n = [0.0]*nstates_n
            err_p = [0.0]*nstates_p
            errBA = 0.0
            num_basis_states_wf = np.array(num_basis_states_f+num_basis_states_g+num_basis_states_c+num_basis_states_d, dtype=np.int32)
            initial_guess_array = func.initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson[0],num_basis_meson[1],num_basis_meson[2],num_basis_meson[3],num_basis_meson[4],[20]*nstates_n,[20]*nstates_p)
            for j in range(n_samples):
                params = param_set[j,:]
                params_array = np.array(params, dtype=np.double)
                solution = root(c_wrapper2, x0=initial_guess_array, args=(params_array,num_basis_states_wf,num_basis_states_meson,), jac=None, method='hybr',options={'col_deriv': 1, 'xtol': 1e-8})
                BA_mev = (BA_function(solution.x,params_array,num_basis_states_wf,num_basis_states_meson)*13.269584506383948 - 0.75*41.0*A**(-1.0/3.0))/A - 939
                for k in range(nstates_n):
                    err_n[k] = err_n[k] + abs(solution.x[k-nstates_n-nstates_p]-real_en_n[j,k])
                for k in range(nstates_p):
                    err_p[k] = err_p[k] + abs(solution.x[k-nstates_p]-real_en_p[j,k])
                errBA = errBA + abs(actual_results[j][0] - BA_mev)
            errBA = errBA/n_samples

            print(num_basis_states_f)
            print(num_basis_states_c)
            
            err_n = [element/n_samples for element in err_n]
            err_p = [element/n_samples for element in err_p]
            greedy_basis = num_basis_states_f+num_basis_states_c
            err = err_n + err_p
            greedy(err,greedy_basis,greedy_add[s][i])
            num_basis_states_f = greedy_basis[:nstates_n]
            num_basis_states_g = greedy_basis[:nstates_n]
            num_basis_states_c = greedy_basis[nstates_n:]
            num_basis_states_d = greedy_basis[nstates_n:]
        
        BAerr_array[s] = errBA
        improv_metric[s] = (finalerr-errBA)
        print(BAerr_array[s],improv_metric[s])
        if (improv_metric[s] > test_improv):
            temp_basis_f = num_basis_states_f
            temp_basis_c = num_basis_states_c
            test_improv = improv_metric[s]
    
    max_index = np.argmax(improv_metric) # get best configuration improvement
    finalerr = BAerr_array[max_index]    # update the new best error
    num_basis_states_base_f = temp_basis_f
    num_basis_states_base_g = temp_basis_f
    num_basis_states_base_c = temp_basis_c
    num_basis_states_base_d = temp_basis_c
    print(improv_metric,max_index,finalerr)