import numpy as np
import sympy as sp
import time
import math
from itertools import chain
from scipy.integrate import simpson as simps
import os
import sys

sys.path.append('/home/msals97/Desktop/RBM/RBM')
import functions as func
import multiprocessing

def replace_coeff_with_vector_parallel(expr, symbols):
    return func.replace_coeff_with_vector(expr, symbols)

if __name__ == "__main__":
    current_directory = os.getcwd()
    print("Current Working directory:", current_directory)

    # User Input
    ##################################################################
    #specify the nucleus (8,6 broken)
    nucleus = 2

    # Jacobian computation
    jac = True
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

    mNuc_mev = 939.0 
    mOmega = 782.5
    mRho = 763.0
    mDelta = 980.0

    # Define parameters of model
    mSigma, gs2, gw2, gp2, gd2, kappa, lambda0, zeta, lambdav = sp.symbols('mSigma gs2 gw2 gp2 gd2 kappa lambda0 zeta lambdav')
    param_list = [mSigma,gs2,gw2,gp2,gd2,kappa,lambda0,zeta,lambdav]
    ##################################################################

    # File specifications
    ##################################################################
    start_time = time.time()

    # Set the directory
    dir = f"{A},{Z}/{A},{Z},Data"

    # Specify common grid
    r_vec = func.load_data(dir + "/rvec.txt")[1:]

    # Import basis numbers
    
    num_basis_states_f, num_basis_states_g, num_basis_states_c, num_basis_states_d, num_basis_meson = func.import_basis_numbers(A,Z)
    num_basis_states_s = num_basis_meson[0]
    num_basis_states_v = num_basis_meson[1]
    num_basis_states_b = num_basis_meson[2]
    num_basis_states_l = num_basis_meson[3]
    num_basis_states_a = num_basis_meson[4]
    
    # For greedy optimization
    # Specify the number of basis states and masses
    """
    num_basis_states_f = [6]*nstates_n
    num_basis_states_g = [6]*nstates_n
    num_basis_states_c = [6]*nstates_p
    num_basis_states_d = [6]*nstates_p
    num_basis_states_s = 7
    num_basis_states_v = 7
    num_basis_states_b = 7
    num_basis_states_l = 7
    num_basis_states_a = 7
    """

    # Import state information (j, alpha, fill_frac)
    n_labels, state_file_n = func.load_spectrum( dir + "/neutron_spectrum.txt")
    p_labels, state_file_p = func.load_spectrum(dir + "/proton_spectrum.txt")
    state_info_n = state_file_n[:,[0,1,2]]
    state_info_p = state_file_p[:,[0,1,2]]

    # Specify the meson field files
    sigma_file = dir + "/meson_fields/sigma.txt"
    omega_file = dir + "/meson_fields/omega.txt"
    rho_file = dir + "/meson_fields/rho.txt"
    delta_file = dir + "/meson_fields/delta.txt"
    coulomb_file = dir + "/meson_fields/coulomb.txt"

    # Import meson fields
    sigma_fields = func.load_data(sigma_file)
    omega_fields = func.load_data(omega_file)
    rho_fields = func.load_data(rho_file)
    delta_fields = func.load_data(delta_file)
    coulomb_fields = func.load_data(coulomb_file)

    # Specify the wave function files for the f_wave neutrons and import
    file_pattern = dir + "/neutron/f_wave/state{}.txt"
    f_files = [file_pattern.format(n_labels[i]) for i in range(nstates_n)]
    f_fields = [func.load_data(f_file) for f_file in f_files]

    # Specify the wave function files for the g_wave neutrons and import
    file_pattern = dir + "/neutron/g_wave/state{}.txt"
    g_files = [file_pattern.format(n_labels[i]) for i in range(nstates_n)]
    g_fields = [func.load_data(g_file) for g_file in g_files]

    # Specify the wave function files for the f_wave protons and import 
    # (I use c and d instead of f and g to differentiate from neutrons and protons)
    file_pattern = dir + "/proton/c_wave/state{}.txt"
    c_files = [file_pattern.format(p_labels[i]) for i in range(nstates_p)]
    c_fields = [func.load_data(c_file) for c_file in c_files]

    # Specify the wave function files for the g_wave protons and import 
    # (I use c and d instead of f and g to differentiate from neutrons and protons)
    file_pattern = dir + "/proton/d_wave/state{}.txt"
    d_files = [file_pattern.format(p_labels[i]) for i in range(nstates_p)]
    d_fields = [func.load_data(d_file) for d_file in d_files]

    # Conversion factors and electric charge (unitless)
    r0_fm = 1.25
    hbar_mevfm = 197.32698
    enscale_mev = hbar_mevfm**2/(2*mNuc_mev*r0_fm**2)
    fm_to_inverse_mev = 1/197.32698
    conv_r0_en = r0_fm*fm_to_inverse_mev*enscale_mev
    e2_unitless = 1.4399627/(enscale_mev*r0_fm)

    end_time_section1 = time.time()
    print("Variable Declarations and data import took {:.4f} seconds".format(end_time_section1 - start_time))
    #################################################################

    # Proper Orthogonal Decomposition
    #################################################################

    # perform SVD and construct basis
    S_basis = func.perform_pod(sigma_fields[1:,:], num_basis_states_s)
    V_basis = func.perform_pod(omega_fields[1:,:], num_basis_states_v)
    B_basis = func.perform_pod(rho_fields[1:,:], num_basis_states_b)
    L_basis = func.perform_pod(delta_fields[1:,:], num_basis_states_l)
    A_basis = func.perform_pod(coulomb_fields[1:,:], num_basis_states_a)
    f_basis = [func.perform_pod(f_fields[i][1:,:], num_basis_states_f[i]) for i in range(nstates_n)]
    g_basis = [func.perform_pod(g_fields[i][1:,:], num_basis_states_g[i]) for i in range(nstates_n)]
    c_basis = [func.perform_pod(c_fields[i][1:,:], num_basis_states_c[i]) for i in range(nstates_p)]
    d_basis = [func.perform_pod(d_fields[i][1:,:], num_basis_states_d[i]) for i in range(nstates_p)]

    # rescale the basis states in accordance to the mean value of the sampled fields
    S_basis = S_basis*np.mean(sigma_fields[10,:])/S_basis[10,0]
    V_basis = V_basis*np.mean(omega_fields[10,:])/V_basis[10,0]
    B_basis = B_basis*np.mean(rho_fields[10,:])/B_basis[10,0]
    L_basis = L_basis*np.mean(delta_fields[10,:])/L_basis[10,0]
    A_basis = A_basis*np.mean(coulomb_fields[10,:])/A_basis[10,0]

    f_basis = [f_basis[i]*np.mean(f_fields[i][10,:])/f_basis[i][10,0] for i in range(nstates_n)]
    g_basis = [g_basis[i]*np.mean(g_fields[i][10,:])/g_basis[i][10,0] for i in range(nstates_n)]
    c_basis = [c_basis[i]*np.mean(c_fields[i][10,:])/c_basis[i][10,0] for i in range(nstates_p)]
    d_basis = [d_basis[i]*np.mean(d_fields[i][10,:])/d_basis[i][10,0] for i in range(nstates_p)]

    end_time_section2 = time.time()
    print("SVD took {:.4f} seconds".format(end_time_section2 - end_time_section1))
    ###############################################################################

    # Generate Reduced Basis Coefficients
    ###############################################################################
    coeff_set_f = [func.coeff_generate_wf('f',i,num_basis_states_f[i]) for i in range(nstates_n)]
    coeff_set_g = [func.coeff_generate_wf('g',i,num_basis_states_g[i]) for i in range(nstates_n)]
    coeff_set_c = [func.coeff_generate_wf('c',i,num_basis_states_c[i]) for i in range(nstates_p)]
    coeff_set_d = [func.coeff_generate_wf('d',i,num_basis_states_d[i]) for i in range(nstates_p)]
    coeff_set_s = func.coeff_generate_meson('s',num_basis_states_s)
    coeff_set_v = func.coeff_generate_meson('v',num_basis_states_v)
    coeff_set_b = func.coeff_generate_meson('b',num_basis_states_b)
    coeff_set_l = func.coeff_generate_meson('l',num_basis_states_l)
    coeff_set_a = func.coeff_generate_meson('a',num_basis_states_a)

    # Define unknown energies
    en_set_n = func.generate_energies_n(nstates_n)
    en_set_p = func.generate_energies_p(nstates_p)

    end_time_section3 = time.time()
    print("Generating coefficients took {:.4f} seconds".format(end_time_section3 - end_time_section2))
    #################################################################################

    # Generate derivatives of the basis states
    ################################################################################
    dgdr_basis = [func.der(g_basis[i], r_vec) for i in range(nstates_n)]
    dfdr_basis = [func.der(f_basis[i], r_vec) for i in range(nstates_n)]
    dddr_basis = [func.der(d_basis[i], r_vec) for i in range(nstates_p)]
    dcdr_basis = [func.der(c_basis[i], r_vec) for i in range(nstates_p)]
    dsdr2_basis = func.der2(S_basis, r_vec)
    dvdr2_basis = func.der2(V_basis, r_vec)
    dbdr2_basis = func.der2(B_basis, r_vec)
    dldr2_basis = func.der2(L_basis, r_vec)
    dadr2_basis = func.der2(A_basis, r_vec)
    dsdr_basis = func.der(S_basis, r_vec)
    dvdr_basis = func.der(V_basis, r_vec)
    dbdr_basis = func.der(B_basis, r_vec)
    dldr_basis = func.der(L_basis, r_vec)
    dadr_basis = func.der(A_basis, r_vec)

    end_time_section4 = time.time()
    print("Derivatives took {:.4f} seconds".format(end_time_section4 - end_time_section3))
    ###################################################################################

    # Galerkin Projection for the Wave functions
    ####################################################################################

    # Define galerkin pojections of the wave functions
    galerk_f, galerk_g, galerk_c, galerk_d = ([] for i in range(4))

    # Galerkin prjection for nucleon fields
    # Information on the field equations go here but need to be fed in carefully since manipulation of the coeff and array objects are handled seperately and recombined
    for i in range(nstates_n):
        for j in range(num_basis_states_g[i]):
            galerk_g_j = func.wf_project(i,j,g_basis,coeff_set_g,dgdr_basis,r_vec) + state_info_n[i][1]*func.wf_project(i,j,g_basis,coeff_set_g,func.divide_array_wf(g_basis,r_vec),r_vec) \
                    + conv_r0_en*en_set_n[i]*func.wf_project(i,j,g_basis,coeff_set_f,f_basis,r_vec) - conv_r0_en*func.meson_wf_project(i,j,g_basis,coeff_set_f,coeff_set_v,f_basis,V_basis,r_vec) \
                    + conv_r0_en*0.5*func.meson_wf_project(i,j,g_basis,coeff_set_f,coeff_set_b,f_basis,B_basis,r_vec) - conv_r0_en*mNuc_mev/enscale_mev*func.wf_project(i,j,g_basis,coeff_set_f,f_basis,r_vec) \
                    + conv_r0_en*func.meson_wf_project(i,j,g_basis,coeff_set_f,coeff_set_s,f_basis,S_basis,r_vec) - conv_r0_en*0.5*func.meson_wf_project(i,j,g_basis,coeff_set_f,coeff_set_l,f_basis,L_basis,r_vec)
            galerk_g.append(galerk_g_j)

    for i in range(nstates_n):
        for j in range(num_basis_states_f[i]):
            galerk_f_j = func.wf_project(i,j,f_basis,coeff_set_f,dfdr_basis,r_vec) - state_info_n[i][1]*func.wf_project(i,j,f_basis,coeff_set_f,func.divide_array_wf(f_basis,r_vec),r_vec) \
                    - conv_r0_en*en_set_n[i]*func.wf_project(i,j,f_basis,coeff_set_g,g_basis,r_vec) + conv_r0_en*func.meson_wf_project(i,j,f_basis,coeff_set_g,coeff_set_v,g_basis,V_basis,r_vec) \
                    - conv_r0_en*0.5*func.meson_wf_project(i,j,f_basis,coeff_set_g,coeff_set_b,g_basis,B_basis,r_vec) - conv_r0_en*mNuc_mev/enscale_mev*func.wf_project(i,j,f_basis,coeff_set_g,g_basis,r_vec) \
                    + conv_r0_en*func.meson_wf_project(i,j,f_basis,coeff_set_g,coeff_set_s,g_basis,S_basis,r_vec) - conv_r0_en*0.5*func.meson_wf_project(i,j,f_basis,coeff_set_g,coeff_set_l,g_basis,L_basis,r_vec)
            galerk_f.append(galerk_f_j)

    for i in range(nstates_p):
        for j in range(num_basis_states_d[i]):
            galerk_d_j = func.wf_project(i,j,d_basis,coeff_set_d,dddr_basis,r_vec) + state_info_p[i][1]*func.wf_project(i,j,d_basis,coeff_set_d,func.divide_array_wf(d_basis,r_vec),r_vec) \
                    + conv_r0_en*en_set_p[i]*func.wf_project(i,j,d_basis,coeff_set_c,c_basis,r_vec) - conv_r0_en*func.meson_wf_project(i,j,d_basis,coeff_set_c,coeff_set_v,c_basis,V_basis,r_vec) \
                    - conv_r0_en*0.5*func.meson_wf_project(i,j,d_basis,coeff_set_c,coeff_set_b,c_basis,B_basis,r_vec) - conv_r0_en*mNuc_mev/enscale_mev*func.wf_project(i,j,d_basis,coeff_set_c,c_basis,r_vec) \
                    + conv_r0_en*func.meson_wf_project(i,j,d_basis,coeff_set_c,coeff_set_s,c_basis,S_basis,r_vec) - conv_r0_en*func.meson_wf_project(i,j,d_basis,coeff_set_c,coeff_set_a,c_basis,A_basis,r_vec) \
                    + conv_r0_en*0.5*func.meson_wf_project(i,j,d_basis,coeff_set_c,coeff_set_l,c_basis,L_basis,r_vec)
            galerk_d.append(galerk_d_j)

    for i in range(nstates_p):
        for j in range(num_basis_states_c[i]):
            galerk_c_j = func.wf_project(i,j,c_basis,coeff_set_c,dcdr_basis,r_vec) - state_info_p[i][1]*func.wf_project(i,j,c_basis,coeff_set_c,func.divide_array_wf(c_basis,r_vec),r_vec) \
                    - conv_r0_en*en_set_p[i]*func.wf_project(i,j,c_basis,coeff_set_d,d_basis,r_vec) + conv_r0_en*func.meson_wf_project(i,j,c_basis,coeff_set_d,coeff_set_v,d_basis,V_basis,r_vec) \
                    + conv_r0_en*0.5*func.meson_wf_project(i,j,c_basis,coeff_set_d,coeff_set_b,d_basis,B_basis,r_vec) - conv_r0_en*mNuc_mev/enscale_mev*func.wf_project(i,j,c_basis,coeff_set_d,d_basis,r_vec) \
                    + conv_r0_en*func.meson_wf_project(i,j,c_basis,coeff_set_d,coeff_set_s,d_basis,S_basis,r_vec) + conv_r0_en*func.meson_wf_project(i,j,c_basis,coeff_set_d,coeff_set_a,d_basis,A_basis,r_vec) \
                    + conv_r0_en*0.5*func.meson_wf_project(i,j,c_basis,coeff_set_d,coeff_set_l,d_basis,L_basis,r_vec)
            galerk_c.append(galerk_c_j)

    end_time_section5 = time.time()
    print("Wave Function Galerkin projection took {:.4f} seconds".format(end_time_section5 - end_time_section4))
    ######################################################################################
            
    # Galerkin projection for meson fields
    #######################################################################################

    # Information on the field equations go here but need to be fed in carefully since manipulation of the coeff and array objects are handled seperately and recombined
    S2_basis = func.basis_multiply(S_basis,S_basis)          
    S2_coeff = func.coeff_multiply(coeff_set_s,coeff_set_s)
    S3_basis = func.basis_multiply(func.basis_multiply(S_basis,S_basis),S_basis)
    S3_coeff = func.coeff_multiply(func.coeff_multiply(coeff_set_s,coeff_set_s),coeff_set_s)
    V3_basis = func.basis_multiply(func.basis_multiply(V_basis,V_basis),V_basis)
    V3_coeff = func.coeff_multiply(func.coeff_multiply(coeff_set_v,coeff_set_v),coeff_set_v)
    V2_basis = func.basis_multiply(V_basis,V_basis)          
    B2_basis = func.basis_multiply(B_basis,B_basis)          
    VB2_basis = func.basis_multiply(V_basis,B2_basis)
    VB2_coeff = func.coeff_multiply(coeff_set_v,func.coeff_multiply(coeff_set_b,coeff_set_b))
    BV2_basis = func.basis_multiply(B_basis,V2_basis)
    BV2_coeff = func.coeff_multiply(coeff_set_b,func.coeff_multiply(coeff_set_v,coeff_set_v))

    # declare terms for meson fields
    # I break up the meson field terms because the equation length goes over pythons recursion limit
    sdensn_s, sdensp_s, dsdr2_s, dsdr_s, S_s, S2_s, S3_s = ([] for i in range(7))
    vdensn_v, vdensp_v, dvdr2_v, dvdr_v, V_v, V3_v, VB2_v = ([] for i in range(7))
    vdensn_b, vdensp_b, dbdr2_b, dbdr_b, B_b, BV2_b = ([] for i in range(6))
    sdensn_l, sdensp_l, dldr2_l, dldr_l, L_l = ([] for i in range(5))
    vdensp_a, dadr2_a, dadr_a = ([] for _ in range(3))
    galerk_s, galerk_v, galerk_b, galerk_l, galerk_a = ([] for i in range(5))

    # generate the terms for each of the basis states and store them in a list
    for j in range(num_basis_states_s):
        # define terms for the sigma field
        sdensn_s_j = func.scalardens_proj(S_basis[:,j],f_basis,g_basis,coeff_set_f,coeff_set_g,nstates_n,state_info_n,r_vec)/conv_r0_en**3
        sdensp_s_j = func.scalardens_proj(S_basis[:,j],c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec)/conv_r0_en**3
        dsdr2_s_j = func.meson_project(j,S_basis,coeff_set_s,dsdr2_basis,r_vec)/conv_r0_en**2
        dsdr_s_j = func.meson_project(j,S_basis,coeff_set_s,dsdr_basis/r_vec[:,np.newaxis],r_vec)
        S_s_j = func.meson_project(j,S_basis,coeff_set_s,S_basis,r_vec)
        S2_s_j = func.meson_project(j,S_basis,S2_coeff,S2_basis,r_vec)
        S3_s_j = func.meson_project(j,S_basis,S3_coeff,S3_basis,r_vec)
        galerk_s_j = (mSigma/enscale_mev)**2*S_s_j - dsdr2_s_j - 2.0/conv_r0_en**2*dsdr_s_j - gs2*(sdensn_s_j + sdensp_s_j - 0.5*kappa/enscale_mev*S2_s_j - 1.0/6.0*lambda0*S3_s_j )
        galerk_s.append(galerk_s_j)
        
        # compile each basis projection to list
        sdensn_s.append(sdensn_s_j)
        sdensp_s.append(sdensp_s_j)
        dsdr2_s.append(dsdr2_s_j)
        dsdr_s.append(dsdr_s_j)
        S_s.append(S_s_j)
        S2_s.append(S2_s_j)
        S3_s.append(S3_s_j)

    for j in range(num_basis_states_v):
        # Define the terms for the omega field
        vdensn_v_j = func.vectordens_proj(V_basis[:,j],f_basis,g_basis,coeff_set_f,coeff_set_g,nstates_n,state_info_n,r_vec)/conv_r0_en**3
        vdensp_v_j = func.vectordens_proj(V_basis[:,j],c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec)/conv_r0_en**3 
        V_v_j = func.meson_project(j,V_basis,coeff_set_v,V_basis,r_vec)
        dvdr2_v_j = func.meson_project(j,V_basis,coeff_set_v,dvdr2_basis,r_vec)/conv_r0_en**2
        dvdr_v_j = func.meson_project(j,V_basis,coeff_set_v,dvdr_basis/r_vec[:,np.newaxis],r_vec)
        V3_v_j = func.meson_project(j,V_basis,V3_coeff,V3_basis,r_vec)
        VB2_v_j = func.meson_project(j,V_basis,VB2_coeff,VB2_basis,r_vec)
        galerk_v_j = (mOmega/enscale_mev)**2*V_v_j - dvdr2_v_j - 2.0/conv_r0_en**2*dvdr_v_j - gw2*(vdensn_v_j + vdensp_v_j - 1.0/6.0*zeta*V3_v_j - 2.0*lambdav*VB2_v_j)
        galerk_v.append(galerk_v_j)
        
        # compile each basis projection to list
        vdensn_v.append(vdensn_v_j)
        vdensp_v.append(vdensp_v_j)
        dvdr2_v.append(dvdr2_v_j)
        dvdr_v.append(dvdr_v_j)
        V_v.append(V_v_j)
        V3_v.append(V3_v_j)
        VB2_v.append(VB2_v_j)

    for j in range(num_basis_states_b):
        # Define the terms for the rho field
        vdensp_b_j = func.vectordens_proj(B_basis[:,j],c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec)/conv_r0_en**3
        vdensn_b_j = func.vectordens_proj(B_basis[:,j],f_basis,g_basis,coeff_set_f,coeff_set_g,nstates_n,state_info_n,r_vec)/conv_r0_en**3 
        B_b_j = func.meson_project(j,B_basis,coeff_set_b,B_basis,r_vec)
        dbdr2_b_j = func.meson_project(j,B_basis,coeff_set_b,dbdr2_basis,r_vec)/conv_r0_en**2
        dbdr_b_j = func.meson_project(j,B_basis,coeff_set_b,dbdr_basis/r_vec[:,np.newaxis],r_vec)
        BV2_b_j = func.meson_project(j,B_basis,BV2_coeff,BV2_basis,r_vec)
        galerk_b_j = (mRho/enscale_mev)**2*B_b_j - dbdr2_b_j - 2/conv_r0_en**2*dbdr_b_j - gp2*(0.5*vdensp_b_j - 0.5*vdensn_b_j - 2.0*lambdav*BV2_b_j)
        galerk_b.append(galerk_b_j)

        # compile each basis projection to list
        vdensn_b.append(vdensn_b_j)
        vdensp_b.append(vdensp_b_j)
        dbdr2_b.append(dbdr2_b_j)
        dbdr_b.append(dbdr_b_j)
        B_b.append(B_b_j)
        BV2_b.append(BV2_b_j)

    for j in range(num_basis_states_l):
        # Define the terms for the delta field
        sdensp_l_j = func.scalardens_proj(L_basis[:,j],c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec)/conv_r0_en**3
        sdensn_l_j = func.scalardens_proj(L_basis[:,j],f_basis,g_basis,coeff_set_f,coeff_set_g,nstates_n,state_info_n,r_vec)/conv_r0_en**3 
        L_l_j = func.meson_project(j,L_basis,coeff_set_l,L_basis,r_vec)
        dldr2_l_j = func.meson_project(j,L_basis,coeff_set_l,dldr2_basis,r_vec)/conv_r0_en**2
        dldr_l_j = func.meson_project(j,L_basis,coeff_set_l,dldr_basis/r_vec[:,np.newaxis],r_vec)
        galerk_l_j = (mDelta/enscale_mev)**2*L_l_j - dldr2_l_j - 2/conv_r0_en**2*dldr_l_j - gd2*(0.5*sdensp_l_j - 0.5*sdensn_l_j)
        galerk_l.append(galerk_l_j)

        # compile each basis projection to list
        sdensn_l.append(sdensn_l_j)
        sdensp_l.append(sdensp_l_j)
        dldr2_l.append(dldr2_l_j)
        dldr_l.append(dldr_l_j)
        L_l.append(L_l_j)

    for j in range(num_basis_states_a):
        # Define the terms for the coulomb field
        dadr2_a_j = func.meson_project(j,A_basis,coeff_set_a,dadr2_basis,r_vec)
        dadr_a_j = func.meson_project(j,A_basis,coeff_set_a,dadr_basis/r_vec[:,np.newaxis],r_vec)
        vdensp_a_j = func.vectordens_proj(A_basis[:,j],c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec)
        galerk_a_j = dadr2_a_j + 2*dadr_a_j + 4.0*math.pi*e2_unitless*vdensp_a_j
        galerk_a.append(galerk_a_j)

        # compile each basis projection to list
        vdensp_a.append(vdensp_a_j)
        dadr2_a.append(dadr2_a_j)
        dadr_a.append(dadr_a_j)

    end_time_section6 = time.time()
    print("Meson Field Galerkin Projection took {:.4f} seconds".format(end_time_section6 - end_time_section5))
    ##############################################################################################################

    # Energy equations 
    #####################################################################################
    equations_en_n = [] 
    equations_en_p = []

    # create equations for neutron energies using normalization condition of the wave function
    coeff_list_f2 = [func.coeff_multiply(coeff_set_f[i],coeff_set_f[i]) for i in range(nstates_n)]
    coeff_list_g2 = [func.coeff_multiply(coeff_set_g[i],coeff_set_g[i]) for i in range(nstates_n)]
    f2 = [func.basis_multiply(f_basis[i],f_basis[i]) for i in range(nstates_n)]
    g2 = [func.basis_multiply(g_basis[i],g_basis[i]) for i in range(nstates_n)]
    for i in range(nstates_n):
        int_f2 = simps(f2[i],x=r_vec,axis=0)
        int_g2 = simps(g2[i],x=r_vec,axis=0)
        f2_exp = func.remap_coeff_to_integrals(int_f2,coeff_list_f2[i])
        g2_exp = func.remap_coeff_to_integrals(int_g2,coeff_list_g2[i])
        norm_en_n = f2_exp + g2_exp
        equation = norm_en_n-1.0
        equations_en_n.append(equation)

    coeff_list_c2 = [func.coeff_multiply(coeff_set_c[i],coeff_set_c[i]) for i in range(nstates_p)]
    coeff_list_d2 = [func.coeff_multiply(coeff_set_d[i],coeff_set_d[i]) for i in range(nstates_p)]
    c2 = [func.basis_multiply(c_basis[i],c_basis[i]) for i in range(nstates_p)]
    d2 = [func.basis_multiply(d_basis[i],d_basis[i]) for i in range(nstates_p)]
    for i in range(nstates_p):
        int_c2 = simps(c2[i],x=r_vec,axis=0)
        int_d2 = simps(d2[i],x=r_vec,axis=0)
        c2_exp = func.remap_coeff_to_integrals(int_c2,coeff_list_c2[i])
        d2_exp = func.remap_coeff_to_integrals(int_d2,coeff_list_d2[i])
        norm_en_p = c2_exp + d2_exp
        equation = norm_en_p-1.0
        equations_en_p.append(equation)

    end_time_section7 = time.time()
    print("Energy Equations took {:.4f} seconds".format(end_time_section7 - end_time_section6))
    ##########################################################################################

    # Compile Equations and coefficients
    ##########################################################################################
    # Compile all the equations and symbols into one list
    equations_f, equations_g, equations_c, equations_d, equations_s, equations_v, equations_b, equations_l, equations_a = ([] for _ in range(9))

    # Store the equations
    for i, expression in enumerate(galerk_g):
        equations_g.append(expression)

    for i, expression in enumerate(galerk_f):
        equations_f.append(expression)

    for i, expression in enumerate(galerk_c):
        equations_c.append(expression)

    for i, expression in enumerate(galerk_d):
        equations_d.append(expression)

    # Create equations for the meson fields (for jacobian)
    for i, expression in enumerate(galerk_s):
        equations_s.append(expression)

    for i, expression in enumerate(galerk_v):
        equations_v.append(expression)

    for i, expression in enumerate(galerk_b):
        equations_b.append(expression)

    for i, expression in enumerate(galerk_l):
        equations_l.append(expression)

    for i, expression in enumerate(galerk_a):
        equations_a.append(expression)

    # Concatenate wf symbols lists of lists into one list
    all_wf_symbols = list(chain.from_iterable(coeff_set_f + coeff_set_g + coeff_set_c + coeff_set_d))

    # Concatenate all lists of symbols into a single list
    all_symbols = all_wf_symbols + coeff_set_s + coeff_set_v + coeff_set_b + coeff_set_l + coeff_set_a + en_set_n + en_set_p
    #print(all_symbols)
    end_time_section8 = time.time()
    print("Equation compile took {:.4f} seconds".format(end_time_section8 - end_time_section7))
    #########################################################################################


    # Replace coefficients with vector elements
    ######################################################################################
    def replace_coeff_with_vector_parallel(expr, symbols):
        return func.replace_coeff_with_vector(expr, symbols)

    # List of functions and variables to parallelize (SIGMA)
    functions_and_variables = [(sdensn_s,all_symbols),(sdensp_s,all_symbols),(dsdr2_s,all_symbols),(dsdr_s,all_symbols),(S_s,all_symbols),(S2_s,all_symbols),(S3_s,all_symbols)]
    pool = multiprocessing.Pool()
    results = pool.starmap(replace_coeff_with_vector_parallel, functions_and_variables)
    pool.close()
    pool.join()
    sdensn_s, sdensp_s, dsdr2_s, dsdr_s, S_s, S2_s, S3_s = results

    # List of functions and variables to parallelize (OMEGA)
    functions_and_variables = [(vdensn_v,all_symbols),(vdensp_v,all_symbols),(dvdr2_v,all_symbols),(dvdr_v,all_symbols),(V_v,all_symbols),(VB2_v,all_symbols),(V3_v,all_symbols)]
    pool = multiprocessing.Pool()
    results = pool.starmap(replace_coeff_with_vector_parallel, functions_and_variables)
    pool.close()
    pool.join()
    vdensn_v, vdensp_v, dvdr2_v, dvdr_v, V_v, VB2_v, V3_v = results

    # List of functions and variables to parallelize (RHO)
    functions_and_variables = [(vdensn_b,all_symbols),(vdensp_b,all_symbols),(dbdr2_b,all_symbols),(dbdr_b,all_symbols),(B_b,all_symbols),(BV2_b,all_symbols)]
    pool = multiprocessing.Pool()
    results = pool.starmap(replace_coeff_with_vector_parallel, functions_and_variables)
    pool.close()
    pool.join()
    vdensn_b, vdensp_b, dbdr2_b, dbdr_b, B_b, BV2_b = results

    # List of functions and variables to parallelize (DELTA)
    functions_and_variables = [(sdensn_l,all_symbols),(sdensp_l,all_symbols),(dldr2_l,all_symbols),(dldr_l,all_symbols),(L_l,all_symbols)]
    pool = multiprocessing.Pool()
    results = pool.starmap(replace_coeff_with_vector_parallel, functions_and_variables)
    pool.close()
    pool.join()
    sdensn_l, sdensp_l, dldr2_l, dldr_l, L_l = results

    # List of functions and variables to parallelize (COULOMB)
    functions_and_variables = [(vdensp_a,all_symbols),(dadr2_a,all_symbols),(dadr_a,all_symbols)]
    pool = multiprocessing.Pool()
    results = pool.starmap(replace_coeff_with_vector_parallel, functions_and_variables)
    pool.close()
    pool.join()
    vdensp_a, dadr2_a, dadr_a =  results

    # convert to c form
    sdensn_s = [str(expr) for expr in sdensn_s]
    func.c_exponent_func(sdensn_s)
    sdensn_s = np.array(sdensn_s)

    sdensp_s = [str(expr) for expr in sdensp_s]
    func.c_exponent_func(sdensp_s)
    sdensp_s = np.array(sdensp_s)

    vdensp_v = [str(expr) for expr in vdensp_v]
    func.c_exponent_func(vdensp_v)
    vdensp_v = np.array(vdensp_v)

    vdensn_v = [str(expr) for expr in vdensn_v]
    func.c_exponent_func(vdensn_v)
    vdensn_v = np.array(vdensn_v)

    vdensp_b = [str(expr) for expr in vdensp_b]
    func.c_exponent_func(vdensp_b)
    vdensp_b = np.array(vdensp_b)

    vdensn_b = [str(expr) for expr in vdensn_b]
    func.c_exponent_func(vdensn_b)
    vdensn_b = np.array(vdensn_b)

    sdensp_l = [str(expr) for expr in sdensp_l]
    func.c_exponent_func(sdensp_l)
    sdensp_l = np.array(sdensp_l)

    sdensn_l = [str(expr) for expr in sdensn_l]
    func.c_exponent_func(sdensn_l)
    sdensn_l = np.array(sdensn_l)

    vdensp_a = [str(expr) for expr in vdensp_a]
    func.c_exponent_func(vdensp_a)
    vdensp_a = np.array(vdensp_a)

    S2_s = [str(expr) for expr in S2_s]
    func.c_exponent_func(S2_s)
    S2_s = np.array(S2_s)

    S3_s = [str(expr) for expr in S3_s]
    func.c_exponent_func(S3_s)
    S3_s = np.array(S3_s)

    V3_v = [str(expr) for expr in V3_v]
    func.c_exponent_func(V3_v)
    V3_v = np.array(V3_v)

    VB2_v = [str(expr) for expr in VB2_v]
    func.c_exponent_func(VB2_v)
    VB2_v = np.array(VB2_v)

    BV2_b = [str(expr) for expr in BV2_b]
    func.c_exponent_func(BV2_b)
    BV2_b = np.array(BV2_b)

    # replace coefficients and model parameters with vector elements
    wf_equations = equations_f + equations_g + equations_c + equations_d
    en_equations = equations_en_n + equations_en_p

    functions_and_variables = [(wf_equations,all_symbols),(en_equations,all_symbols)]
    pool = multiprocessing.Pool()
    results = pool.starmap(replace_coeff_with_vector_parallel, functions_and_variables)
    pool.close()
    pool.join()
    new_expressions, new_en_expr = results

    final_expressions_wf = func.replace_params_with_vector(new_expressions,param_list)
    final_en_expr = func.replace_params_with_vector(new_en_expr,param_list)

    end_time_section9 = time.time()
    print("Equation coefficent and parameter replacement took {:.4f} seconds".format(end_time_section9 - end_time_section8))
    ######################################################################

    # Compute the Jacobian
    #####################################################################
    if (jac == True):
        meson_equations = equations_s + equations_v + equations_b + equations_l + equations_a
        all_expressions = wf_equations + meson_equations + en_equations
        jacobian_matrix = sp.Matrix([[sp.diff(expr, symbol) for symbol in all_symbols] for expr in all_expressions])

        # replace jacobian coefficients with elements of vector x
        new_jac = func.replace_coeff_with_vector_jac(jacobian_matrix.tolist(),all_symbols)
        final_jac = func.replace_params_with_vector_jac(new_jac,param_list)

        # convert to c form
        str_jac = [[str(expr) for expr in row] for row in final_jac]
        func.c_exponent_jac(str_jac)
        c_jac = np.array(str_jac).flatten()

    end_time_section10 = time.time()
    print("Jacobian took {:.4f} seconds".format(end_time_section10 - end_time_section9))

    # Write equations to file
    ######################################################################
    # Reset the file
    with open(f"{A},{Z}/equations.txt", 'w') as file:
        file.write('')

    # Store meson terms
    with open(f"{A},{Z}/equations.txt", 'a') as file:
        # Store all scalar terms in the file
        for j in range(num_basis_states_s):
            file.write(f"    sdensn_s[{j}] = {sdensn_s[j]};\n")
            file.write(f"    sdensp_s[{j}] = {sdensp_s[j]};\n")
            file.write(f"    dsdr2_s[{j}] = {dsdr2_s[j]};\n")
            file.write(f"    dsdr_s[{j}] = {dsdr_s[j]};\n")
            file.write(f"    S_s[{j}] = {S_s[j]};\n")
            file.write(f"    S2_s[{j}] = {S2_s[j]};\n")
            file.write(f"    S3_s[{j}] = {S3_s[j]};\n")
        
        # Store all omega terms in the file
        for j in range(num_basis_states_v):
            file.write(f"    vdensn_v[{j}] = {vdensn_v[j]};\n")
            file.write(f"    vdensp_v[{j}] = {vdensp_v[j]};\n")
            file.write(f"    dvdr2_v[{j}] = {dvdr2_v[j]};\n")
            file.write(f"    dvdr_v[{j}] = {dvdr_v[j]};\n")
            file.write(f"    V_v[{j}] = {V_v[j]};\n")
            file.write(f"    V3_v[{j}] = {V3_v[j]};\n")
            a,b = func.split_expr(f"{VB2_v[j]}")
            file.write(f"    a = {a};\n")
            file.write(f"    b = {b};\n")
            file.write(f"    VB2_v[{j}] = a+b;\n")
        
        # Store all rho terms in the file
        for j in range(num_basis_states_b):
            file.write(f"    vdensn_b[{j}] = {vdensn_b[j]};\n")
            file.write(f"    vdensp_b[{j}] = {vdensp_b[j]};\n")
            file.write(f"    dbdr2_b[{j}] = {dbdr2_b[j]};\n")
            file.write(f"    dbdr_b[{j}] = {dbdr_b[j]};\n")
            file.write(f"    B_b[{j}] = {B_b[j]};\n")
            a,b = func.split_expr(f"{BV2_b[j]}")
            file.write(f"    a = {a};\n")
            file.write(f"    b = {b};\n")
            file.write(f"    BV2_b[{j}] = a+b;\n")
        
        # Store all delta terms in the file
        for j in range(num_basis_states_b):
            file.write(f"    sdensn_l[{j}] = {sdensn_l[j]};\n")
            file.write(f"    sdensp_l[{j}] = {sdensp_l[j]};\n")
            file.write(f"    dldr2_l[{j}] = {dldr2_l[j]};\n")
            file.write(f"    dldr_l[{j}] = {dldr_l[j]};\n")
            file.write(f"    L_l[{j}] = {L_l[j]};\n")
        
        # Store all  coulomb terms in the file
        for j in range(num_basis_states_a):
            file.write(f"    vdensp_a[{j}] = {vdensp_a[j]};\n")
            file.write(f"    dadr2_a[{j}] = {dadr2_a[j]};\n")
            file.write(f"    dadr_a[{j}] = {dadr_a[j]};\n")

    # convert to c
    final_expressions_wf = [str(expr) for expr in final_expressions_wf]
    func.c_exponent_func(final_expressions_wf)
    final_expressions_wf = np.array(final_expressions_wf)

    # Store wf equations
    func.print_equations_to_file(final_expressions_wf,f"{A},{Z}/equations.txt")

    # Store meson equations
    #  0   1    2    3    4   5   6    7     8
    # ms  gs2  gw2  gp2  gd2  k  lam  zeta  lambV
    n_wf_equations = sum(num_basis_states_f) + sum(num_basis_states_g) + sum(num_basis_states_c) + sum(num_basis_states_d)
    with open(f"{A},{Z}/equations.txt", 'a') as file:
        for j in range(num_basis_states_s):
            file.write(f"    y[{n_wf_equations + j}] = pow(params[0]/enscale_mev,2.0)*S_s[{j}] - dsdr2_s[{j}] - 2.0*pow(1.0/conv_r0_en,2.0)*dsdr_s[{j}] - params[1]*(sdensn_s[{j}] + sdensp_s[{j}] - 0.5*params[5]/enscale_mev*S2_s[{j}] - 1.0/6.0*params[6]*S3_s[{j}] ); \n")
        
        for j in range(num_basis_states_v):
            file.write(f"    y[{n_wf_equations + num_basis_states_s + j}] = pow(782.5/enscale_mev,2.0)*V_v[{j}] - dvdr2_v[{j}] - 2.0*pow(1.0/conv_r0_en,2.0)*dvdr_v[{j}] - params[2]*(vdensn_v[{j}] + vdensp_v[{j}] - 1.0/6.0*params[7]*V3_v[{j}] - 2.0*params[8]*VB2_v[{j}] ); \n")
        
        for j in range(num_basis_states_b):
            file.write(f"    y[{n_wf_equations + num_basis_states_s + num_basis_states_v + j}] = pow(763.0/enscale_mev,2.0)*B_b[{j}] - dbdr2_b[{j}] - 2.0*pow(1.0/conv_r0_en,2.0)*dbdr_b[{j}] - params[3]*(0.5*vdensp_b[{j}] - 0.5*vdensn_b[{j}] - 2.0*params[8]*BV2_b[{j}] ); \n")
        
        for j in range(num_basis_states_l):
            file.write(f"    y[{n_wf_equations + num_basis_states_s + num_basis_states_v + num_basis_states_b + j}] = pow(980.0/enscale_mev,2.0)*L_l[{j}] - dldr2_l[{j}] - 2.0*pow(1.0/conv_r0_en,2.0)*dldr_l[{j}] - params[4]*(0.5*sdensp_l[{j}] - 0.5*sdensn_l[{j}] ); \n")
        
        for j in range(num_basis_states_a):
            file.write(f"    y[{n_wf_equations + num_basis_states_s + num_basis_states_v + num_basis_states_b + num_basis_states_l + j}] = dadr2_a[{j}] + 2.0*dadr_a[{j}] + 1.09092217*vdensp_a[{j}]; \n")

    # convert to c
    final_en_expr = [str(expr) for expr in final_en_expr]
    func.c_exponent_func(final_en_expr)
    final_en_expr = np.array(final_en_expr)

    with open(f"{A},{Z}/equations.txt", 'a') as file:
        for j in range(nstates_n+nstates_p):
            file.write(f"    y[{n_wf_equations + num_basis_states_s + num_basis_states_v + num_basis_states_b + num_basis_states_l + num_basis_states_a + j}] = {final_en_expr[j]}; \n")

    if (jac == True):
        with open(f"{A},{Z}/jacobian.txt", 'w') as file:
            for i in range(len(c_jac)):
                file.write(f"    jac[{i}] = {c_jac[i]}; \n")

    # Get BA Equations
    V4_basis = func.basis_multiply(func.basis_multiply(func.basis_multiply(V_basis,V_basis),V_basis),V_basis)
    V4_coeff = func.coeff_multiply(func.coeff_multiply(func.coeff_multiply(coeff_set_v,coeff_set_v),coeff_set_v),coeff_set_v)
    S4_basis = func.basis_multiply(func.basis_multiply(func.basis_multiply(S_basis,S_basis),S_basis),S_basis)
    S4_coeff = func.coeff_multiply(func.coeff_multiply(func.coeff_multiply(coeff_set_s,coeff_set_s),coeff_set_s),coeff_set_s)
    V2_coeff = func.coeff_multiply(coeff_set_v,coeff_set_v)
    B2_coeff = func.coeff_multiply(coeff_set_b,coeff_set_b)
    V2B2_coeff = func.coeff_multiply(V2_coeff,B2_coeff)
    V2B2_basis = func.basis_multiply(V2_basis,B2_basis)

    en_neutrons = 0
    en_protons = 0
    for i in range(nstates_n):
        en_neutrons = en_neutrons + state_info_n[i,2]*(2.0*state_info_n[i,0]+1.0)*en_set_n[i]
    for i in range(nstates_p):
        en_protons = en_protons + state_info_p[i,2]*(2.0*state_info_p[i,0]+1.0)*en_set_p[i]

    meson_free = 0
    for i in range(num_basis_states_s):
        S_sdensn = func.scalardens_BA(S_basis[:,i],f_basis,g_basis,coeff_set_f,coeff_set_g,nstates_n,state_info_n,r_vec)
        S_sdensp = func.scalardens_BA(S_basis[:,i],c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec)
        meson_free = meson_free + coeff_set_s[i]*(S_sdensn+S_sdensp)
    for i in range(num_basis_states_v):
        V_vdensn = func.vectordens_BA(V_basis[:,i],f_basis,g_basis,coeff_set_f,coeff_set_g,nstates_n,state_info_n,r_vec)
        V_vdensp = func.vectordens_BA(V_basis[:,i],c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec)
        meson_free = meson_free - coeff_set_v[i]*(V_vdensn+V_vdensp)
    for i in range(num_basis_states_b):
        B_vdensn = func.vectordens_BA(B_basis[:,i],f_basis,g_basis,coeff_set_f,coeff_set_g,nstates_n,state_info_n,r_vec)
        B_vdensp = func.vectordens_BA(B_basis[:,i],c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec)
        meson_free = meson_free - coeff_set_b[i]*0.5*(B_vdensp-B_vdensn)
    for i in range(num_basis_states_l):
        L_sdensn = func.scalardens_BA(L_basis[:,i],f_basis,g_basis,coeff_set_f,coeff_set_g,nstates_n,state_info_n,r_vec)
        L_sdensp = func.scalardens_BA(L_basis[:,i],c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec)
        meson_free = meson_free + coeff_set_l[i]*0.5*(L_sdensp-L_sdensn)
    for i in range(num_basis_states_a):
        A_vdensp = func.vectordens_BA(A_basis[:,i],c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec)
        meson_free = meson_free - coeff_set_a[i]*A_vdensp

    meson_interact = - 1.0/6.0*kappa/enscale_mev*func.nonlinear_BA(S3_coeff,S3_basis,r_vec)*conv_r0_en**3 -1.0/12.0*lambda0*func.nonlinear_BA(S4_coeff,S4_basis,r_vec)*conv_r0_en**3 \
                    + 1.0/12.0*zeta*func.nonlinear_BA(V4_coeff,V4_basis,r_vec)*conv_r0_en**3 + 2.0*lambdav*func.nonlinear_BA(V2B2_coeff,V2B2_basis,r_vec)*conv_r0_en**3

    meson_free = meson_free*2.0*math.pi
    meson_interact = meson_interact*2.0*math.pi
    #BA_unitless = en_neutrons + en_protons + 2.0*math.pi*meson_integrand

    x = [sp.symbols(f'x[{i}]') for i in range(len(all_symbols))]
    sub_dict = {all_symbols[i]: x[i] for i in range(len(all_symbols))}
    meson_free = meson_free.subs(sub_dict)
    meson_interact = meson_interact.subs(sub_dict)
    en_BA = en_neutrons + en_protons
    en_BA = en_BA.subs(sub_dict)

    meson_free_str = [str(meson_free)]
    func.c_exponent_func(meson_free_str)
    meson_free_c = np.array(meson_free_str)

    x = [sp.symbols(f'params[{i}]') for i in range(len(param_list))]
    sub_dict = {param_list[i]: x[i] for i in range(len(param_list))}
    meson_interact = meson_interact.subs(sub_dict)

    meson_interact_str = [str(meson_interact)]
    func.c_exponent_func(meson_interact_str)
    meson_interact_c = np.array(meson_interact_str)

    en_BA_str = [str(en_BA)]
    func.c_exponent_func(en_BA_str)
    en_BA_c = np.array(en_BA_str)
    
    with open(f"{A},{Z}/BA.txt", 'w') as file:
        file.write(f"    meson_free = {meson_free_c[0]};\n")
        file.write(f"    meson_interact = {meson_interact_c[0]};\n")
        file.write(f"    en_BA = {en_BA_c[0]};\n")
        file.write(f"    BA = en_BA + meson_free + meson_interact;\n")
    
    x = [sp.symbols(f'x[{i}]') for i in range(len(all_symbols))]
    sub_dict = {all_symbols[i]: x[i] for i in range(len(all_symbols))}

    # get Rch
    Rp2 = func.get_Rp2(Z,c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec)
    Rp2 = Rp2.subs(sub_dict)

    Rp2_str = [str(Rp2)]
    func.c_exponent_func(Rp2_str)
    Rp2_c = np.array(Rp2_str)

    if (nucleus == 2 or nucleus == 9):
        # get form factor at q
        if (nucleus ==2):
            q = 0.8733
        else:
            q = 0.3977

        Fvn = func.Fv(f_basis,g_basis,coeff_set_f,coeff_set_g,nstates_n,state_info_n,r_vec,q)
        Fvp = func.Fv(c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec,q)
        Ftn = func.Ft(f_basis,g_basis,coeff_set_f,coeff_set_g,nstates_n,state_info_n,r_vec,q)
        Ftp = func.Ft(c_basis,d_basis,coeff_set_c,coeff_set_d,nstates_p,state_info_p,r_vec,q)

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

        GEp, GEn, GMp, GMn = func.EM_Sachs_FF_pn(q)
        WGEp, WGEn, WGMp, WGMn = func.WKEM_Sachs_FF_pn(GEp,GEn,GMp,GMn,Z)

        Fch_p = GEp*Fvp + (GMp - GEp)/(1+tau)*(tau*Fvp + 0.5*q*hbar_gevfm/M_gev*Ftp)
        Fch_n = GEn*Fvn + (GMn - GEn)/(1+tau)*(tau*Fvn + 0.5*q*hbar_gevfm/M_gev*Ftn)
        Fch = (Fch_p + Fch_n)/Z

        Fwk_p = WGEp*Fvp + (WGMp - WGEp)/(1+tau)*(tau*Fvp + 0.5*q*hbar_gevfm/M_gev*Ftp)
        Fwk_n = WGEn*Fvn + (WGMn - WGEn)/(1+tau)*(tau*Fvn + 0.5*q*hbar_gevfm/M_gev*Ftn)
        Fwk = (Fwk_p + Fwk_n)/Qwk

        Wkskin = Fch-Fwk
        x = [sp.symbols(f'x[{i}]') for i in range(len(all_symbols))]
        sub_dict = {all_symbols[i]: x[i] for i in range(len(all_symbols))}
        Wkskin = Wkskin.subs(sub_dict)

        Wkskin_str = [str(Wkskin)]
        func.c_exponent_func(Wkskin_str)
        Wkskin_c = np.array(Wkskin_str)

    if (nucleus == 2 or nucleus == 9):
        with open(f"{A},{Z}/BA.txt", 'a') as file:
            file.write(f"    FchFwk = {Wkskin_c[0]};\n")
            file.write(f"    Rp2 = {Rp2_c[0]};\n")
            file.write(f"    Rch = sqrt(Rp2 + pow(0.84,2.0));\n")
    else:
        with open(f"{A},{Z}/BA.txt", 'a') as file:
            file.write(f"    Rp2 = {Rp2_c[0]};\n")
            file.write(f"    Rch = sqrt(Rp2 + pow(0.84,2.0));\n")
        

    end_time_section11 = time.time()
    print("Printing took {:.4f} seconds".format(end_time_section11 - end_time_section10))
    ################################################################################



