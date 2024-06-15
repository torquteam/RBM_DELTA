import numpy as np
import math
import sympy as sp
from scipy.linalg import svd
from scipy.integrate import simps
from scipy.optimize import root
import bulk2params as trans

import time
import sys
sys.path.append('/home/msals97/Desktop/RBM/RBM')

# Define functions to be used in this RBM scripts
################################################################
################################################################

# Functions used for RBM script
################################################################
################################################################

# Load the data from a text file
def load_data(file_path):
    data = np.loadtxt(file_path)
    return data

# loads the spectrum and returns list of orbital labels eg. (1s1/2) and a list of numerical values (n,j,l,alpha,fill frac)
def load_spectrum(file_path):
    data = np.loadtxt(file_path, dtype=str)
    values = data[:, :-1].astype(float)  # Convert numeric part to float64
    labels = data[:, -1]  # Last column remains as strings
    return labels, values

# Prints a list to a file
def write_list_to_file(my_list, file_path):
    with open(file_path, 'w') as file:
        for item in my_list:
            file.write(str(item) + '\n')

# Perform Proper Orthogonal Decomposition (POD) for each component and returns basis vectors
def perform_pod(data, n_basis):
    U, s, VT = svd(data, full_matrices=True)
    U_red = U[:, :n_basis]
    return U_red

# input is a list of basis vectors [x0, x1, x2 ...]
# used to take derivative of something like a0*x0 + a1*x1 + ... with respect to r (where x_i are vectors)
def der(array, r_vec):
    result_array = np.zeros_like(array, dtype=float)
    for i in range(array.shape[1]):
        result_array[:, i] = np.gradient(array[:, i], r_vec, edge_order=2)
    return result_array

# input is a list of basis vectors [x0, x1, x2 ...]
# used to take second derivative of something like a0*x0 + a1*x1 + ... with respect to r (where x_i are vectors)
def der2(array, r_vec):
    result_array = np.zeros_like(array, dtype=float)
    for i in range(array.shape[1]):
        first_derivative = np.gradient(array[:, i], r_vec,edge_order=2)
        second_derivative = np.gradient(first_derivative, r_vec, edge_order=2)
        result_array[:, i] = second_derivative
    return result_array

# generates coefficients for wave functions (function refers to f,g,c,d where {{f,g},{c,d}} are the upper and lower components of the neutron and proton wave functions respectively)
# n is the number of basis coefficients and alpha is the quantum number
def coeff_generate_wf(function, alpha, n):
    return [sp.symbols(f'coeff_a{i}_{function}_{alpha}') for i in range(0, n)]

# generate coefficients for the meson fields (field refers to s,v,b,a for meson and coulomb fields)
# n is the number of basis coefficients 
def coeff_generate_meson(field, n):
    return [sp.symbols(f'coeff_b{i}_{field}') for i in range(0, n)]

# generate energy variables for proton
def generate_energies_n(n_states):
    return [sp.symbols(f'en_n_{i}') for i in range(0, n_states)]

# generate energy variables for neutron
def generate_energies_p(n_states):
    return [sp.symbols(f'en_p_{i}') for i in range(0, n_states)]

# when multiplying basis expansions the coefficients are saved in an array 
# (multiplying coefficients and basis states seperately and recombining later speeds up computation)
# most likely bypasses the need for a full cython implementation since sympy is very slow
def coeff_multiply(x_symbols, y_symbols):
    result = [x * y for x in x_symbols for y in y_symbols]
    return result

# when multiplying basis expansions the new basis states are saved in an array 
def basis_multiply(x, y):
    n_columns_x = x.shape[1]
    n_columns_y = y.shape[1]
    result = np.column_stack([
        x[:, i] * y[:, j] for i in range(n_columns_x) for j in range(n_columns_y)
    ])
    return result

# Remaps the coefficients to the corresponding arrays
# ex (a0*x0 + a1*x1)*(b0*x0 + b1*x1) --> [a0*b0, a0*b1, a1*b0, a1*b1] + [x0^2, x0*x1, x1*x0, x1^2] --> a0*b0*x0^2 + a0*b1*x0*x1 + a1*b0*x1*x0 + a1*b1*x1^2
def remap_coeff_to_arrays(array, symbol_list):
    if array.shape[1] != len(symbol_list):
        raise ValueError("Number of symbols must match the number of columns in the array.")
    result = np.zeros_like(array, dtype=object)
    for j in range(array.shape[1]):
        result[:, j] = array[:, j] * symbol_list[j]
    return result

# When doing galerkin projections the integrals are done and then coefficients are remapped to the resulting integrals 
# (speeds up computation since the integral of a sympy expressions is very slow)
def remap_coeff_to_integrals(list_of_integrals, coeff_list):
    result = 0
    if len(list_of_integrals) != len(coeff_list):
        raise ValueError("The lists are not of equal size.")
    
    for l, k in zip(list_of_integrals, coeff_list):
        result = result + l*k
    return result

# when doing galerkin projections this computes the projection of a wave function with a basis state of a wave function 
# ex <x_f_i | f_a(r)>  where f_a(r) = a_f_0*x_0 + a_f_1*x_1 + ...
def wf_project(bound_state, single_basis_state, basis_proj,coeff_set, wf_array, r_vec):
    num_res = simps(basis_proj[bound_state][:,single_basis_state][:,np.newaxis]*wf_array[bound_state],x=r_vec,axis=0)
    result = remap_coeff_to_integrals(num_res,coeff_set[bound_state])
    return result

# when doing galerkin projections this computes the projection of a meson field*wave function with a basis state of a wave function 
# ex <x_f_i | f_a(r)*v(r)>  where v(r) = b_v_0*y_0 + b_v_1*y_1 + ... and f_a(r) = a_f_0*x_0 + a_f_1*x_1 + ...
def meson_wf_project(bound_state, single_basis_state, basis_proj,coeff_set_wf, coeff_set_meson, wf_array, meson_array, r_vec):
    meson_wf = basis_multiply(meson_array,wf_array[bound_state])
    num_res = simps(basis_proj[bound_state][:,single_basis_state][:,np.newaxis]*meson_wf,x=r_vec,axis=0)
    coeff_set = coeff_multiply(coeff_set_meson,coeff_set_wf[bound_state])
    result = remap_coeff_to_integrals(num_res,coeff_set)
    return result

# when doing galerkin projections this computes the projection of a meson field with a basis state of a meson field
# ex <y_i | v(r)>  where v(r) = b_v_0*y_0 + b_v_1*y_1 + ...
def meson_project(single_basis_state, basis_proj, coeff_set_meson, meson_array, r_vec):
    num_res = simps(basis_proj[:,single_basis_state][:,np.newaxis]*meson_array,x=r_vec,axis=0)
    result = remap_coeff_to_integrals(num_res,coeff_set_meson)
    return result

# Used to evaluate bidning energy integrals
def nonlinear_BA(coeff_set_meson, meson_array, r_vec):
    num_res = simps(meson_array*r_vec[:,np.newaxis]**2,x=r_vec,axis=0)
    result = remap_coeff_to_integrals(num_res,coeff_set_meson)
    return result

# used for doing things like f(r)/r
def divide_array_wf(list_of_arrays, common_array):
    result_list = [array / common_array[:,np.newaxis] for array in list_of_arrays]
    return result_list

# used for multiplying a whole set of wave functions by a basis state
def multiply_array_wf(list_of_arrays, common_array):
    result_list = [array * common_array[:,np.newaxis] for array in list_of_arrays]
    return result_list

# get the scalar density and perform a galerkin projection onto a specified basis state
def scalardens_proj(basis_proj, basis_f, basis_g, coeff_f, coeff_g, nstates,state_info,r_vec):
    res = 0
    coeff_list_f = [coeff_multiply(coeff_f[i],coeff_f[i]) for i in range(nstates)]
    coeff_list_g = [coeff_multiply(coeff_g[i],coeff_g[i]) for i in range(nstates)]
    f2r2 = [basis_multiply(basis_f[i],basis_f[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #f^2/r^2
    g2r2 = [basis_multiply(basis_g[i],basis_g[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #g^2/r^2
    proj_array_f = multiply_array_wf(f2r2,basis_proj) #y_i*f^2/r^2
    proj_array_g = multiply_array_wf(g2r2,basis_proj) #y_i*g^2/r^2

    for i in range(nstates):
        proj_f = state_info[i][2]*(2.0*state_info[i][0]+1)/(4.0*math.pi)*simps(proj_array_f[i],x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(f^2/r^2 * y_i)
        proj_g = state_info[i][2]*(2.0*state_info[i][0]+1)/(4.0*math.pi)*simps(proj_array_g[i],x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(g^2/r^2 * y_i)
        res = res + remap_coeff_to_integrals(proj_f,coeff_list_f[i]) - remap_coeff_to_integrals(proj_g,coeff_list_g[i]) # subtract or add for scalar/vec
    return res

# get the scalar density times the meson field to compute BA
def scalardens_BA(meson_basis, basis_f, basis_g, coeff_f, coeff_g, nstates,state_info,r_vec):
    res = 0
    coeff_list_f = [coeff_multiply(coeff_f[i],coeff_f[i]) for i in range(nstates)]
    coeff_list_g = [coeff_multiply(coeff_g[i],coeff_g[i]) for i in range(nstates)]
    f2r2 = [basis_multiply(basis_f[i],basis_f[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #f^2/r^2
    g2r2 = [basis_multiply(basis_g[i],basis_g[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #g^2/r^2
    meson_dens_array_f = multiply_array_wf(f2r2,meson_basis) #phi_i*f^2/r^2
    meson_dens_array_g = multiply_array_wf(g2r2,meson_basis) #phi_i*g^2/r^2

    for i in range(nstates):
        proj_f = state_info[i][2]*(2.0*state_info[i][0]+1)/(4.0*math.pi)*simps(meson_dens_array_f[i]*r_vec[:, np.newaxis]**2,x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(f^2/r^2 * y_i)
        proj_g = state_info[i][2]*(2.0*state_info[i][0]+1)/(4.0*math.pi)*simps(meson_dens_array_g[i]*r_vec[:, np.newaxis]**2,x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(g^2/r^2 * y_i)
        res = res + remap_coeff_to_integrals(proj_f,coeff_list_f[i]) - remap_coeff_to_integrals(proj_g,coeff_list_g[i]) # subtract or add for scalar/vec
    return res

# get the vector density times the meson field to compute BA
def vectordens_BA(meson_basis, basis_f, basis_g, coeff_f, coeff_g, nstates,state_info,r_vec):
    res = 0
    coeff_list_f = [coeff_multiply(coeff_f[i],coeff_f[i]) for i in range(nstates)]
    coeff_list_g = [coeff_multiply(coeff_g[i],coeff_g[i]) for i in range(nstates)]
    f2r2 = [basis_multiply(basis_f[i],basis_f[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #f^2/r^2
    g2r2 = [basis_multiply(basis_g[i],basis_g[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #g^2/r^2
    meson_dens_array_f = multiply_array_wf(f2r2,meson_basis) #phi_i*f^2/r^2
    meson_dens_array_g = multiply_array_wf(g2r2,meson_basis) #phi_i*g^2/r^2

    for i in range(nstates):
        proj_f = state_info[i][2]*(2.0*state_info[i][0]+1)/(4.0*math.pi)*simps(meson_dens_array_f[i]*r_vec[:, np.newaxis]**2,x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(f^2/r^2 * y_i)
        proj_g = state_info[i][2]*(2.0*state_info[i][0]+1)/(4.0*math.pi)*simps(meson_dens_array_g[i]*r_vec[:, np.newaxis]**2,x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(g^2/r^2 * y_i)
        res = res + remap_coeff_to_integrals(proj_f,coeff_list_f[i]) + remap_coeff_to_integrals(proj_g,coeff_list_g[i]) # subtract or add for scalar/vec
    return res

# get the vector density and perform a galerkin projection onto a specified basis state
def vectordens_proj(basis_proj, basis_f, basis_g, coeff_f, coeff_g, nstates,state_info, r_vec):
    res = 0
    coeff_list_f = [coeff_multiply(coeff_f[i],coeff_f[i]) for i in range(nstates)]
    coeff_list_g = [coeff_multiply(coeff_g[i],coeff_g[i]) for i in range(nstates)]
    f2r2 = [basis_multiply(basis_f[i],basis_f[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #f^2/r^2
    g2r2 = [basis_multiply(basis_g[i],basis_g[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #g^2/r^2
    proj_array_f = multiply_array_wf(f2r2,basis_proj) #y_i*f^2/r^2
    proj_array_g = multiply_array_wf(g2r2,basis_proj) #y_i*g^2/r^2

    for i in range(nstates):
        proj_f = state_info[i][2]*(2.0*state_info[i][0]+1)/(4*math.pi)*simps(proj_array_f[i],x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(f^2/r^2 * y_i)
        proj_g = state_info[i][2]*(2.0*state_info[i][0]+1)/(4*math.pi)*simps(proj_array_g[i],x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(g^2/r^2 * y_i)
        res = res + remap_coeff_to_integrals(proj_f,coeff_list_f[i]) + remap_coeff_to_integrals(proj_g,coeff_list_g[i]) # subtract or add for scalar/vec
    return res

# replace sympy symbols with vector unknowns for coefficients
def replace_coeff_with_vector(expressions_list, symbol_list):
    x = [sp.symbols(f'x[{i}]') for i in range(len(symbol_list))]

    # Replace symbols with corresponding elements in the vector x
    expressions_with_vector = [expr.subs({symbol_list[i]: x[i] for i in range(len(symbol_list))}) for expr in expressions_list]
    return expressions_with_vector

# replace sympy symbols with vector unknowns for parameters
def replace_params_with_vector(expressions_list, symbol_list):
    x = [sp.symbols(f'params[{i}]') for i in range(len(symbol_list))]

    # Replace symbols with corresponding elements in the vector x
    expressions_with_vector = [expr.subs({symbol_list[i]: x[i] for i in range(len(symbol_list))}) for expr in expressions_list]
    return expressions_with_vector

# replace sympy symbols in jacobian
def replace_coeff_with_vector_jac(expressions_matrix, symbol_list):
    x = [sp.symbols(f'x[{i}]') for i in range(len(symbol_list))]

    # Replace symbols with corresponding elements in the vector x
    matrix_with_vector = [
        [expr.subs({symbol_list[i]: x[i] for i in range(len(symbol_list))}) for expr in row]
        for row in expressions_matrix
    ]
    return matrix_with_vector

# replace sympy symbols for parameters in jacobian
def replace_params_with_vector_jac(expressions_matrix, symbol_list):
    x = [sp.symbols(f'params[{i}]') for i in range(len(symbol_list))]

    # Replace symbols with corresponding elements in the vector x
    matrix_with_vector = [
        [expr.subs({symbol_list[i]: x[i] for i in range(len(symbol_list))}) for expr in row]
        for row in expressions_matrix
    ]
    return matrix_with_vector

# prints galerkin equations to file
def print_equations_to_file(array, filename):
    with open(filename, 'a') as file:
        for i, value in enumerate(array):
            equation = f"    y[{i}] = {array[i]};\n"
            file.write(equation)

# splits long expressions into 2. some long expressions cause recursion errors in python
def split_expr(expression):
    # Extract operands and operators
    s_length = len(expression)
    s_mid = s_length//2

    char = s_mid
    while char < s_length:
        if expression[char] == '+' or expression[char] == '-':
            sub_a = expression[:(char-1)]
            sub_b = expression[char:]
            return sub_a, sub_b
        else:
            char += 1

# replace x**y exponentiation by the c standard pow(x,y) in the jacobian
def c_exponent_jac(jacobian):
    for row in jacobian:
        for i in range(len(row)):
            expr = row[i]
            index = expr.find("**")
            while(index >= 0):
                char = index-1
                while char >= 0:
                    if expr[char] == '+' or expr[char] == '-' or expr[char] == '*':
                        term = expr[char+1:index+3]
                        exp = expr[index+2]
                        base = expr[char+1:index]
                        c_term = f"pow({base},{exp})"
                        expr = expr.replace(term,c_term)
                        break
                    else:
                        char = char - 1
                index = expr.find("**")
                row[i] = expr

# replace x**y exponentiation by the c standard pow(x,y) in the galerkin equations
def c_exponent_func(arr):
    for i in range(len(arr)):
        expr = arr[i]
        index = expr.find("**")
        while(index >= 0):
            char = index-1
            while char >= 0:
                if expr[char] == '+' or expr[char] == '-' or expr[char] == '*':
                    term = expr[char+1:index+3]
                    exp = expr[index+2]
                    base = expr[char+1:index]
                    c_term = f"pow({base},{exp})"
                    expr = expr.replace(term,c_term)
                    break
                else:
                    char = char - 1
            index = expr.find("**")
            arr[i] = expr

# used for uneven length arrays such as {{0,1},{0,1,2},{0}} to make the consistently same size
def pad(arr):
    # Find the maximum length among all inner arrays
    max_length = max(len(row) for row in arr)
    
    # Pad each inner array with zeros
    padded_arr = np.array([np.pad(row, (0, max_length - len(row)), mode='constant') for row in arr])
    return padded_arr

# These functions are used to compute weak skin
################################################################

# compute vector and tensor form factors at a given momentum transfer
def Fv_Ft(q, vdens, tdens, rvec, r0_fm):
    q = q*r0_fm
    Fvec = 4.0*math.pi*simps(vdens*np.sin(q*rvec)*rvec/q,x=rvec)
    Ftens = 4.0*math.pi*simps(tdens*(np.sin(q*rvec)/q**2 - np.cos(q*rvec)*rvec/q),x=rvec)
    return Fvec, Ftens

# get the scalar density times the meson field to compute BA
def Fv(basis_f, basis_g, coeff_f, coeff_g, nstates,state_info,r_vec,q):
    res = 0
    q = q*1.25
    coeff_list_f = [coeff_multiply(coeff_f[i],coeff_f[i]) for i in range(nstates)]
    coeff_list_g = [coeff_multiply(coeff_g[i],coeff_g[i]) for i in range(nstates)]
    f2r2 = [basis_multiply(basis_f[i],basis_f[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #f^2/r^2
    g2r2 = [basis_multiply(basis_g[i],basis_g[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #g^2/r^2

    for i in range(nstates):
        f2 = state_info[i][2]*(2.0*state_info[i][0]+1)*simps(f2r2[i]*np.sin(q*r_vec[:,np.newaxis])*r_vec[:,np.newaxis]/q,x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(f^2/r^2 * y_i)
        g2 = state_info[i][2]*(2.0*state_info[i][0]+1)*simps(g2r2[i]*np.sin(q*r_vec[:,np.newaxis])*r_vec[:,np.newaxis]/q,x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(g^2/r^2 * y_i)
        res = res + remap_coeff_to_integrals(f2,coeff_list_f[i]) + remap_coeff_to_integrals(g2,coeff_list_g[i]) # subtract or add for scalar/vec
    return res

# get the scalar density times the meson field to compute BA
def Ft(basis_f, basis_g, coeff_f, coeff_g, nstates,state_info,r_vec,q):
    res = 0
    q = q*1.25
    coeff_list_fg = [coeff_multiply(coeff_f[i],coeff_g[i]) for i in range(nstates)]
    fgr2 = [basis_multiply(basis_f[i],basis_g[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #f^2/r^2

    for i in range(nstates):
        fg = 2.0*state_info[i][2]*(2.0*state_info[i][0]+1)*simps(fgr2[i]*(np.sin(q*r_vec[:,np.newaxis])/q**2 - np.cos(q*r_vec[:,np.newaxis])*r_vec[:,np.newaxis]/q),x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(f^2/r^2 * y_i)
        res = res + remap_coeff_to_integrals(fg,coeff_list_fg[i])
    return res

def EM_Sachs_FF_pn(q):
    hbar_gevfm = .19732698
    q2 = (q*hbar_gevfm)**2
    mup = 2.79284356
    mun = -1.91304272

    a_GE_p = [0.239163298067,-1.10985857441,1.44438081306,0.479569465603,-2.28689474187,1.12663298498,1.250619843540,
                    -3.63102047159,4.08221702379,0.504097346499,-5.08512046051,3.96774254395,-0.981529071103]
    a_GM_p = [0.264142994136,-1.09530612212,1.21855378178,0.661136493537,-1.40567892503,-1.35641843888,1.447029155340,
                    4.23566973590,-5.33404565341,-2.916300520960,8.70740306757,-5.70699994375,1.280814375890]
    a_GE_n = [0.048919981379,-0.064525053912,-0.240825897382,0.392108744873,0.300445258602,-0.661888687179,-0.175639769687,
                    0.624691724461,-0.077684299367,-0.236003975259,0.090401973470,0.000000000000,0.000000000000]
    a_GM_n = [0.257758326959,-1.079540642058,1.182183812195,0.711015085833,-1.348080936796,-1.662444025208,2.624354426029,
                    1.751234494568,-4.922300878888,3.197892727312,-0.712072389946,0.000000000000,0.000000000000]
    
    t0 = -0.7
    tcut = 0.0779191396
    z = (np.sqrt(tcut+q2)-np.sqrt(tcut-t0))/(np.sqrt(tcut+q2)+np.sqrt(tcut-t0))

    GE_p = 0
    GM_p = 0
    GE_n = 0
    GM_n = 0

    for i in range(13):
        GE_p = GE_p + a_GE_p[i]*z**i
        GM_p = GM_p + a_GM_p[i]*z**i
        GE_n = GE_n + a_GE_n[i]*z**i
        GM_n = GM_n + a_GM_n[i]*z**i

    GM_p = mup*GM_p
    GM_n = mun*GM_n

    return GE_p, GE_n, GM_p, GM_n

def WKEM_Sachs_FF_pn(GE_p, GE_n, GM_p, GM_n, Z):
    qwp = 0.0713
    if (Z == 20):
        qwn = -0.9795
    else:
        qwn = -0.9821
    
    WGE_p = qwp*GE_p+qwn*GE_n
    WGE_n = qwp*GE_n+qwn*GE_p
    WGM_p = qwp*GM_p+qwn*GM_n
    WGM_n = qwp*GM_n+qwn*GM_p

    return WGE_p, WGE_n, WGM_p, WGM_n

def FchFwk(q,vdensn,vdensp,tdensn,tdensp,rvec,A,Z):
    r0_fm = 1.25
    hbar_gevfm = .19732698
    M_gev = 0.939
    q2 = (q*hbar_gevfm)**2 
    tau = 0.25*q2/M_gev**2

    qwp = 0.0713
    if (Z == 20):
        qwn = -0.9795
    else:
        qwn = -0.9821
    Qwk = Z*qwp + (A-Z)*qwn

    GEp, GEn, GMp, GMn = EM_Sachs_FF_pn(q)
    WGEp, WGEn, WGMp, WGMn = WKEM_Sachs_FF_pn(GEp,GEn,GMp,GMn,Z)
    Fvec_p, Ftens_p = Fv_Ft(q,vdensp,tdensp,rvec,r0_fm)
    Fvec_n, Ftens_n = Fv_Ft(q,vdensn,tdensn,rvec,r0_fm)

    Fch_p = GEp*Fvec_p + (GMp - GEp)/(1+tau)*(tau*Fvec_p + 0.5*q*hbar_gevfm/M_gev*Ftens_p)
    Fch_n = GEn*Fvec_n + (GMn - GEn)/(1+tau)*(tau*Fvec_n + 0.5*q*hbar_gevfm/M_gev*Ftens_n)
    Fch = (Fch_p + Fch_n)/Z

    Fwk_p = WGEp*Fvec_p + (WGMp - WGEp)/(1+tau)*(tau*Fvec_p + 0.5*q*hbar_gevfm/M_gev*Ftens_p)
    Fwk_n = WGEn*Fvec_n + (WGMn - WGEn)/(1+tau)*(tau*Fvec_n + 0.5*q*hbar_gevfm/M_gev*Ftens_n)
    Fwk = (Fwk_p + Fwk_n)/Qwk

    return Fch-Fwk

################################################################
################################################################

# Define functions used in solving the galerkin equations RBM solve script
################################################################
################################################################
def initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_states_s,num_basis_states_v,num_basis_states_b,num_basis_states_l,num_basis_states_a,n_energies,p_energies):
    initial_guesses = []
    for i in range(nstates_n):
        initial_guesses = initial_guesses + [1.0] + [0.0]*(num_basis_states_f[i]-1)

    for i in range(nstates_n):
        initial_guesses = initial_guesses + [1.0] + [0.0]*(num_basis_states_g[i]-1)

    for i in range(nstates_p):
        initial_guesses = initial_guesses + [1.0] + [0.0]*(num_basis_states_c[i]-1)

    for i in range(nstates_p):
        initial_guesses = initial_guesses + [1.0] + [0.0]*(num_basis_states_d[i]-1)

    initial_guesses = initial_guesses + [1.0] + [0.0]*(num_basis_states_s-1) + [1.0] + [0.0]*(num_basis_states_v-1) + [1.0] + [0.0]*(num_basis_states_b-1) + [1.0] + [0.0]*(num_basis_states_l-1) + [1.0] + [0.0]*(num_basis_states_a-1)
    initial_guesses = initial_guesses + n_energies + p_energies

    initial_guess_array = np.array(initial_guesses, dtype=np.double)
    return initial_guess_array

def get_BA(nstates_n, nstates_p,state_info_n,state_info_p,solution,s_field_approx,v_field_approx,b_field_approx,l_field_approx,a_field_approx,sdensn,sdensp,vdensn,vdensp,params,r_vec,A):
    r0_fm = 1.25
    hbar_mevfm = 197.32698
    enscale_mev = hbar_mevfm**2/(2*939*r0_fm**2)
    fm_to_inverse_mev = 1/197.32698
    conv_r0_en = r0_fm*fm_to_inverse_mev*enscale_mev
    en_neutrons = 0.0
    en_protons = 0.0
    for i in range(nstates_n):
        en_neutrons = en_neutrons + state_info_n[i,2]*(2.0*state_info_n[i,0]+1.0)*solution.x[i-nstates_n-nstates_p]
    for i in range(nstates_p):
        en_protons = en_protons + state_info_p[i,2]*(2.0*state_info_p[i,0]+1.0)*solution.x[i-nstates_p]

    meson_integrand = s_field_approx*(sdensn+sdensp) - v_field_approx*(vdensn+vdensp) - 0.5*b_field_approx*(vdensp-vdensn) + 0.5*l_field_approx*(sdensp-sdensn) - 1/6*params[5]/enscale_mev*s_field_approx**3*conv_r0_en**3 -1/12*params[6]*s_field_approx**4*conv_r0_en**3 \
                    + 1/12*params[7]*v_field_approx**4*conv_r0_en**3 + 2.0*params[8]*v_field_approx**2*b_field_approx**2*conv_r0_en**3 
    coulomb_integrand = -a_field_approx*vdensp
    integral = simps((meson_integrand+coulomb_integrand)*r_vec**2,x=r_vec)
    BA_unitless = en_neutrons + en_protons + 2*math.pi*integral
    #print(en_neutrons*enscale_mev/A, en_protons*enscale_mev/A, 2*math.pi*integral*enscale_mev/A)
    BA_mev = (BA_unitless*enscale_mev - 0.75*41.0*A**(-1.0/3.0))/A - 939
    return BA_mev

def get_radii(A,Z,r_vec,vdensn,vdensp):
    r0_fm = 1.25
    rp = 0.84
    Rn2 = 4*math.pi/(A-Z)*simps(r_vec**4*vdensn,x=r_vec)*r0_fm**2
    Rp2 = 4*math.pi/Z*simps(r_vec**4*vdensp,x=r_vec)*r0_fm**2
    Rn = math.sqrt(Rn2)
    Rp = math.sqrt(Rp2)
    Rch = math.sqrt(Rp2 + rp**2)
    return Rn, Rp, Rch

def get_Rp2(Z, basis_f, basis_g, coeff_f, coeff_g, nstates,state_info,r_vec):
    rp = 0.84
    Rp2 = 0
    coeff_list_f = [coeff_multiply(coeff_f[i],coeff_f[i]) for i in range(nstates)]
    coeff_list_g = [coeff_multiply(coeff_g[i],coeff_g[i]) for i in range(nstates)]
    f2r2 = [basis_multiply(basis_f[i],basis_f[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #f^2/r^2
    g2r2 = [basis_multiply(basis_g[i],basis_g[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #g^2/r^2

    for i in range(nstates):
        proj_f = state_info[i][2]*(2.0*state_info[i][0]+1)*simps(f2r2[i]*r_vec[:, np.newaxis]**4,x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(f^2/r^2 * y_i)
        proj_g = state_info[i][2]*(2.0*state_info[i][0]+1)*simps(g2r2[i]*r_vec[:, np.newaxis]**4,x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(g^2/r^2 * y_i)
        Rp2 = Rp2 + remap_coeff_to_integrals(proj_f,coeff_list_f[i]) + remap_coeff_to_integrals(proj_g,coeff_list_g[i]) # subtract or add for scalar/vec
    Rp2 = Rp2/Z*1.25**2
    return Rp2

# get the scalar density times the meson field to compute BA
def vectordens_BA(meson_basis, basis_f, basis_g, coeff_f, coeff_g, nstates,state_info,r_vec):
    res = 0
    coeff_list_f = [coeff_multiply(coeff_f[i],coeff_f[i]) for i in range(nstates)]
    coeff_list_g = [coeff_multiply(coeff_g[i],coeff_g[i]) for i in range(nstates)]
    f2r2 = [basis_multiply(basis_f[i],basis_f[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #f^2/r^2
    g2r2 = [basis_multiply(basis_g[i],basis_g[i])/r_vec[:, np.newaxis]**2 for i in range(nstates)] #g^2/r^2
    meson_dens_array_f = multiply_array_wf(f2r2,meson_basis) #phi_i*f^2/r^2
    meson_dens_array_g = multiply_array_wf(g2r2,meson_basis) #phi_i*g^2/r^2

    for i in range(nstates):
        proj_f = state_info[i][2]*(2.0*state_info[i][0]+1)/(4.0*math.pi)*simps(meson_dens_array_f[i]*r_vec[:, np.newaxis]**2,x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(f^2/r^2 * y_i)
        proj_g = state_info[i][2]*(2.0*state_info[i][0]+1)/(4.0*math.pi)*simps(meson_dens_array_g[i]*r_vec[:, np.newaxis]**2,x=r_vec,axis=0) # sum over states and integrate (2j+1)/(4*pi)*int(g^2/r^2 * y_i)
        res = res + remap_coeff_to_integrals(proj_f,coeff_list_f[i]) + remap_coeff_to_integrals(proj_g,coeff_list_g[i]) # subtract or add for scalar/vec
    return res

def import_basis_numbers(A,Z):
    dir = f"{A},{Z}/{A},{Z},Data"
    # Initialize arrays to store the values
    arrays = []

    # Open the file
    with open(dir+ '/basis_numbers.txt', 'r') as file:
        # Read the first four lines
        for _ in range(4):
            line = file.readline().strip()  # Read the line and remove leading/trailing whitespace
            array = list(map(int, line.split()))  # Split the line into numbers and convert them to integers
            arrays.append(array)  # Append the array to the list of arrays
        
        # Read the last four lines
        last_line = file.read().strip()  # Read the remaining lines and remove leading/trailing whitespace
        meson_array = list(map(int, last_line.split()))  # Split the line into numbers and convert them to integers

    # Separate the last four lines into separate arrays
    f, g, c, d = arrays
    return f,g,c,d,meson_array

def get_basis(A, Z, nstates_n, nstates_p):
    # Specify conversion factors and electric charge (unitless)
    r0_fm = 1.25
    mNuc_mev = 939
    hbar_mevfm = 197.32698
    enscale_mev = hbar_mevfm**2/(2*mNuc_mev*r0_fm**2)
    fm_to_inverse_mev = 1/197.32698

    dir = f"{A},{Z}/{A},{Z},Data"
    # Specify the meson field files
    sigma_file = dir + "/meson_fields/sigma.txt"
    omega_file = dir + "/meson_fields/omega.txt"
    rho_file = dir + "/meson_fields/rho.txt"
    delta_file = dir + "/meson_fields/delta.txt"
    coulomb_file = dir + "/meson_fields/coulomb.txt"

    # Import meson fields
    sigma_fields = load_data(sigma_file)
    omega_fields = load_data(omega_file)
    rho_fields = load_data(rho_file)
    delta_fields = load_data(delta_file)
    coulomb_fields = load_data(coulomb_file)

    # import state information (j, alpha, fill_frac, filetag)
    n_labels, state_file_n = load_spectrum( dir + "/neutron_spectrum.txt")
    p_labels, state_file_p = load_spectrum(dir + "/proton_spectrum.txt")

    # Specify the wave function files for the f_wave neutrons and import
    file_pattern = dir + "/neutron/f_wave/state{}.txt"
    f_files = [file_pattern.format(n_labels[i]) for i in range(nstates_n)]
    f_fields = [load_data(f_file) for f_file in f_files]

    # Specify the wave function files for the g_wave neutrons and import
    file_pattern = dir + "/neutron/g_wave/state{}.txt"
    g_files = [file_pattern.format(n_labels[i]) for i in range(nstates_n)]
    g_fields = [load_data(g_file) for g_file in g_files]

    # Specify the wave function files for the f_wave protons and import 
    # (I use c and d instead of f and g to differentiate from neutrons and protons)
    file_pattern = dir + "/proton/c_wave/state{}.txt"
    c_files = [file_pattern.format(p_labels[i]) for i in range(nstates_p)]
    c_fields = [load_data(c_file) for c_file in c_files]

    # Specify the wave function files for the g_wave protons and import 
    # (I use c and d instead of f and g to differentiate from neutrons and protons)
    file_pattern = dir + "/proton/d_wave/state{}.txt"
    d_files = [file_pattern.format(p_labels[i]) for i in range(nstates_p)]
    d_fields = [load_data(d_file) for d_file in d_files]

    # Import basis numbers
    num_basis_states_f, num_basis_states_g, num_basis_states_c, num_basis_states_d, num_basis_meson = import_basis_numbers(A,Z)
    num_basis_states_s = num_basis_meson[0]
    num_basis_states_v = num_basis_meson[1]
    num_basis_states_b = num_basis_meson[2]
    num_basis_states_l = num_basis_meson[3]
    num_basis_states_a = num_basis_meson[4]

    S_basis = perform_pod(sigma_fields, num_basis_states_s)
    V_basis = perform_pod(omega_fields, num_basis_states_v)
    B_basis = perform_pod(rho_fields, num_basis_states_b)
    L_basis = perform_pod(delta_fields, num_basis_states_l)
    A_basis = perform_pod(coulomb_fields, num_basis_states_a)
    f_basis = [perform_pod(f_fields[i], np.max(num_basis_states_f)) for i in range(nstates_n)]
    g_basis = [perform_pod(g_fields[i], np.max(num_basis_states_g)) for i in range(nstates_n)]
    c_basis = [perform_pod(c_fields[i], np.max(num_basis_states_c)) for i in range(nstates_p)]
    d_basis = [perform_pod(d_fields[i], np.max(num_basis_states_d)) for i in range(nstates_p)]

    # rescale the basis states in accordance to the mean value of the sampled fields
    S_basis = S_basis[1:,:]*np.mean(sigma_fields[10,:])/S_basis[10,0]
    V_basis = V_basis[1:,:]*np.mean(omega_fields[10,:])/V_basis[10,0]
    B_basis = B_basis[1:,:]*np.mean(rho_fields[10,:])/B_basis[10,0]
    L_basis = L_basis[1:,:]*np.mean(delta_fields[10,:])/L_basis[10,0]
    A_basis = A_basis[1:,:]*np.mean(coulomb_fields[10,:])/A_basis[10,0]

    f_basis = np.array([f_basis[i][1:,:]*np.mean(f_fields[i][10,:])/f_basis[i][10,0] for i in range(nstates_n)])
    g_basis = np.array([g_basis[i][1:,:]*np.mean(g_fields[i][10,:])/g_basis[i][10,0] for i in range(nstates_n)])
    c_basis = np.array([c_basis[i][1:,:]*np.mean(c_fields[i][10,:])/c_basis[i][10,0] for i in range(nstates_p)])
    d_basis = np.array([d_basis[i][1:,:]*np.mean(d_fields[i][10,:])/d_basis[i][10,0] for i in range(nstates_p)])
    return f_basis, g_basis, c_basis, d_basis, S_basis, V_basis, B_basis, L_basis, A_basis


def get_wf_basis_states(A, Z, nstates_n, nstates_p, num_basis_states_f, num_basis_states_c):
    dir = f"{A},{Z}/{A},{Z},Data"

    # import state information (j, alpha, fill_frac, filetag)
    n_labels, state_file_n = load_spectrum( dir + "/neutron_spectrum.txt")
    p_labels, state_file_p = load_spectrum(dir + "/proton_spectrum.txt")

    # Specify the wave function files for the f_wave neutrons and import
    file_pattern = dir + "/neutron/f_wave/state{}.txt"
    f_files = [file_pattern.format(n_labels[i]) for i in range(nstates_n)]
    f_fields = [load_data(f_file) for f_file in f_files]

    # Specify the wave function files for the f_wave protons and import 
    # (I use c and d instead of f and g to differentiate from neutrons and protons)
    file_pattern = dir + "/proton/c_wave/state{}.txt"
    c_files = [file_pattern.format(p_labels[i]) for i in range(nstates_p)]
    c_fields = [load_data(c_file) for c_file in c_files]

    f_basis = [perform_pod(f_fields[i][1:,:], np.max(num_basis_states_f)) for i in range(nstates_n)]
    c_basis = [perform_pod(c_fields[i][1:,:], np.max(num_basis_states_c)) for i in range(nstates_p)]

    f_basis = np.array([f_basis[i]*np.mean(f_fields[i][10,:])/f_basis[i][10,0] for i in range(nstates_n)])
    c_basis = np.array([c_basis[i]*np.mean(c_fields[i][10,:])/c_basis[i][10,0] for i in range(nstates_p)])
    return f_basis, c_basis
    
def hartree_RBM(A,nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson,params,c_function_wrapper,BA_function_wrapper,Rch_function_wrapper,Wkskin_wrapper,n_energies,p_energies,jac=None):
    if (A==132):
        initial_guess_array = initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson[0],num_basis_meson[1],num_basis_meson[2],num_basis_meson[3],num_basis_meson[4],[60.0]*nstates_n,[60.0]*nstates_p)
    else:
        initial_guess_array = initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson[0],num_basis_meson[1],num_basis_meson[2],num_basis_meson[3],num_basis_meson[4],n_energies,p_energies)
    params_array = np.array(params, dtype=np.double)
    solution = root(c_function_wrapper, x0=initial_guess_array, args=(params_array,), jac=jac, method='hybr',options={'col_deriv': 1, 'xtol': 1e-8})

    BA_mev = (BA_function_wrapper(solution.x,params)*13.269584506383948 - 0.75*41.0*A**(-1.0/3.0))/A - 939
    Rcharge = Rch_function_wrapper(solution.x)
    FchFwk = Wkskin_wrapper(solution.x)

    en_n = [solution.x[i-nstates_n-nstates_p] for i in range(nstates_n)]
    en_p = [solution.x[i-nstates_p] for i in range(nstates_p)]

    if (abs(BA_mev) > 9.0 or abs(BA_mev) < 7.0):
        initial_guess_array = initial_guess(nstates_n,nstates_p,num_basis_states_f,num_basis_states_g,num_basis_states_c,num_basis_states_d,num_basis_meson[0],num_basis_meson[1],num_basis_meson[2],num_basis_meson[3],num_basis_meson[4],[60.0]*nstates_n,[60.0]*nstates_p)
        solution = root(c_function_wrapper, x0=initial_guess_array, args=(params_array,), jac=jac, method='hybr',options={'col_deriv': 1, 'xtol': 1e-8})
        BA_mev = (BA_function_wrapper(solution.x,params)*13.269584506383948 - 0.75*41.0*A**(-1.0/3.0))/A - 939
        Rcharge = Rch_function_wrapper(solution.x)
        FchFwk = Wkskin_wrapper(solution.x)
        en_n = n_energies
        en_p = p_energies
    
    return BA_mev, Rcharge, FchFwk, en_n, en_p

################################################################
################################################################

# Define functions used in Bayesian script
################################################################
################################################################
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

# define the prior distribution
def compute_prior(bulks):
    prior_data = [-16.0, 0.15, 0.6, 230.0, 34.0, 80.0, 100.0, 0.03, 500.0]
    prior_unct = [1.0  , 0.04, 0.1, 10.0 , 4.0 , 40.0, 500.0, 0.03, 50.0 ]
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
        lkl = lkl*np.exp(-0.5*(exp_data[4]-FchFwk_th)**2/exp_data[5]**2)
        #lkl = lkl*1.0
    return lkl

# define the metropolis hastings algorithm
def metropolis(post0, postp, bulks_0, bulks_p, acc_counts, index, n_params):
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
def param_change(n_params, bulks_0, bulks_p, stds, mw, mp, md, index):
    for k in range(n_params):
        bulks_p[k] = bulks_0[k] # copy old values
    bulks_p[index] = np.random.normal(bulks_0[index],stds[index])      # change one param
    params, flag = trans.get_parameters(bulks_p[0],bulks_p[1],bulks_p[2],bulks_p[3],bulks_p[4],bulks_p[5],bulks_p[6],bulks_p[7],bulks_p[8],mw,mp,md)
    return params, flag

######################################################################
######################################################################

# functions used in getting ideal number of basis
######################################################################

def greedy(err, basis, N, max_basis):
    flag = False  # Initialize flag
    count = 0
    while(count<N):
        max_index = np.argmax(err)
        if basis[max_index] == max_basis:
            err[max_index] = -100  # Exclude the current maximum from further consideration
        else:
            basis[max_index] += 1
            count += 1
            err[max_index] = -100  # Exclude the current maximum from further consideration
    
    if all(val == max_basis for val in basis):
        flag = True
    return flag

def compute_fields(f_coeff,c_coeff,nstates_n,nstates_p,f_basis,c_basis):
    f_fields_approx = np.transpose([np.dot(f_basis[i], f_coeff[i]) for i in range(nstates_n)])
    c_fields_approx = np.transpose([np.dot(c_basis[i], c_coeff[i]) for i in range(nstates_p)])
    return f_fields_approx, c_fields_approx