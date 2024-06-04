import numpy as np
import math
from scipy import optimize

# evaluates the scalar density
def scalardens(k, mstar):
    en = np.sqrt(k**2+mstar**2)
    integral = mstar/2.0*( k*en - mstar**2*np.log( abs((k+en)/mstar) ) )
    return integral/math.pi**2

# evaluates the energy density integral
def endensity_integral(k, mstar):
    en = np.sqrt(k**2+mstar**2)
    integral = k*en**3/4.0 - mstar**2*k*en/8.0 - mstar**4/8.0*np.log(k+en) + mstar**4/8.0*np.log(abs(mstar))
    return integral/math.pi**2

# integral of k^4/rp^3 dk (comes from derivative of the scalar dens)
def common_integral(k, mstar):
    en = np.sqrt(k**2+mstar**2)
    integral = (0.5*k**3 + 3.0/2.0*k*mstar**2)/en + 3.0*mstar**2/2.0*np.log(abs(mstar/(k+en)))
    return integral

# solve for gpomp2
def get_gpomp2(kf, J, L, gss, gww, gsoms2, gwomw2, kappa, lambda0, zeta, lambda_s=0.0, gdomd2=0.0):
    mNuc = 939.0
    p0 = 2.0/(3.0*math.pi**2)*kf**3
    mstar = mNuc - gss
    en = np.sqrt(kf**2 + mstar**2)
    integral = common_integral(kf,mstar)
    alpha_s = 1.0 + gsoms2*(2.0/math.pi**2*integral + kappa*gss + 0.5*lambda0*gss**2)
    dgssdp = gsoms2*(mstar/en)*pow(alpha_s,-1.0)
    dmstardp = -dgssdp
    dendp = 1.0/en*(0.5*pow(math.pi,2.0)/kf + mstar*dmstardp)
    dgdddt = -pow(kf,3.0)/(3.0*pow(math.pi,2.0))*gdomd2*(mstar/en)*pow(1.0 + gdomd2*0.5/pow(math.pi,2.0)*integral + 2.0*lambda_s*gdomd2*pow(gss,2.0),-1.0)
    alpha_d = alpha_d = 1.0 + gdomd2*1.0/(2.0*pow(math.pi,2.0))*integral + 2.0*lambda_s*gdomd2*pow(gss,2.0)
    dIdp = 0.5*pow(kf*math.pi,2.0)/pow(en,3.0) - 3.0*dmstardp*mstar*(-1.0/3.0*pow(kf/en,3.0) - kf/en + np.log(abs((kf+en)/mstar)))
    dgdddpdt =  -0.5*gdomd2*(mstar/en)*pow(alpha_d,-1) - pow(kf,3.0)/(3.0*pow(math.pi,2.0))*gdomd2*(dmstardp/en - mstar/pow(en,2.0)*dendp)*pow(alpha_d,-1.0) + pow(kf,3.0)/(3.0*pow(math.pi,2.0))*pow(gdomd2,2.0)*(mstar/en)*(1.0/(2.0*pow(math.pi,2.0))*dIdp + 4.0*lambda_s*gss*dgssdp)*pow(alpha_d,-2.0)
    phi = pow(math.pi,2.0)/(6.0*kf*en) - pow(kf,2.0)/(6.0*pow(en,2.0))*dendp + 0.25/en*(dmstardp - mstar/en*dendp)*dgdddt + 0.25*mstar/en*dgdddpdt
    chi = 12.0*pow(math.pi,2.0)/pow(kf,3.0)*J - 2.0*pow(math.pi,2.0)/kf*1.0/en + pow(mstar/en,2.0)*gdomd2*pow(alpha_d,-1.0)
    gpomp2 = gpomp2 = 2.0*gwomw2*pow(1.0+0.5*zeta*gwomw2*pow(gww,2.0),-1.0)*pow(chi,2.0)/(2.0*gwomw2*pow(1.0+0.5*zeta*gwomw2*pow(gww,2.0),-1.0)*chi - (4.0*phi - 4.0*L/(3.0*p0) + 0.5*chi)*3.0*pow(math.pi,2.0)/pow(kf,3.0)*gww)
    return gpomp2

# solve for lambda_v
def get_lambda_v(kf, J, gss, gww, gpomp2, lambda_s=0.0, gdomd2=0.0):
    mNuc = 939.0
    mstar = mNuc - gss
    en = np.sqrt(kf**2 + mstar**2)

    integral = common_integral(kf,mstar)
    alpha_d = 1.0 + gdomd2*1.0/(2.0*math.pi**2)*integral + 2.0*lambda_s*gdomd2*gss**2
    lambda_v = 0.5*(pow(12.0*pow(math.pi,2.0)/pow(kf,3.0)*J - 2.0*pow(math.pi,2.0)/(kf*en) + pow(mstar/en,2.0)*gdomd2*pow(alpha_d,-1.0),-1.0)*gpomp2 - 1.0)/(pow(gww,2.0)*gpomp2)
    return lambda_v

def get_Ksym(kf,gss,gww,gsoms2,gwomw2,gpomp2,gdomd2,kappa,lambda0,zeta,lambda_v,lambda_s=0.0):
    mNuc = 939.0

    mstar = mNuc - gss
    en = math.sqrt(pow(kf,2) + pow(mstar,2))
    integral = common_integral(kf,mstar)
    alpha_s = 1.0 + gsoms2*(2.0/pow(math.pi,2.0)*integral + kappa*gss + 0.5*lambda0*pow(gss,2.0))
    dgssdp = gsoms2*(mstar/en)*pow(alpha_s,-1.0)
    dmstardp = -dgssdp
    dgwwdp = gwomw2*pow(1.0 + 0.5*zeta*gwomw2*pow(gww,2.0),-1.0)
    dgdddt = -pow(kf,3.0)/(3.0*pow(math.pi,2.0))*gdomd2*(mstar/en)*pow(1.0 + gdomd2*0.5/pow(math.pi,2.0)*integral + 2.0*lambda_s*gdomd2*pow(gss,2.0),-1.0)
    dendp = 1.0/en*(0.5*pow(math.pi,2.0)/kf + mstar*dmstardp)
    dIdp = 0.5*pow(kf*math.pi,2.0)/pow(en,3.0) - 3.0*dmstardp*mstar*(-1.0/3.0*pow(kf/en,3.0) - kf/en + math.log(abs((kf+en)/mstar)))

    d2gssdp2 = gsoms2*1.0/en*(dmstardp - mstar/en*dendp)*pow(alpha_s,-1.0) - pow(gsoms2,2.0)*mstar/en*(2.0/pow(math.pi,2.0)*dIdp + kappa*dgssdp + lambda0*gss*dgssdp)*pow(alpha_s,-2.0)
    d2mstardp2 = -d2gssdp2
    d2endp2 = -1.0/pow(en,2.0)*dendp*(pow(math.pi,2.0)*0.5/kf + mstar*dmstardp) + 1.0/en*(-0.25*pow(math.pi/kf,4.0) + pow(dmstardp,2.0) + mstar*d2mstardp2)
    d2gwwdp2 = -pow(gwomw2,2.0)*zeta*gww*dgwwdp*pow(1.0 + 0.5*zeta*gwomw2*pow(gww,2.0),-2.0)

    var = -1.0/3.0*pow(kf/en,3.0) - kf/en + math.log(abs((kf+en)/mstar)); 
    varprime = -0.5*pow(math.pi,2.0)/pow(en,3.0) - 0.5*pow(math.pi/kf,2.0)/en + pow(kf,3.0)*dendp/pow(en,4.0) + kf*dendp/pow(en,2.0) \
                -(kf+en)*dmstardp/(mstar*kf+mstar*en) + mstar*(0.5*pow(math.pi/kf,2.0) + dendp)/(mstar*kf+mstar*en)
    d2Idp2 = pow(math.pi,4.0)/(2.0*kf*pow(en,3.0)) - 3.0*pow(kf*math.pi,2.0)*0.5/pow(en,4.0)*dendp - 3.0*d2mstardp2*mstar*var - 3.0*pow(dmstardp,2.0)*var \
                -3.0*dmstardp*mstar*varprime

    alpha_d = 1.0 + gdomd2*1.0/(2.0*pow(math.pi,2.0))*integral + 2.0*lambda_s*gdomd2*pow(gss,2.0)
    dalpha_ddp = gdomd2*(1.0/(2.0*pow(math.pi,2.0))*dIdp + 4.0*lambda_s*gss*dgssdp)
    d2alpha_ddp2 = gdomd2*(1.0/(2.0*pow(math.pi,2.0))*d2Idp2 + 4.0*lambda_s*pow(dgssdp,2.0) + 4.0*lambda_s*gss*d2gssdp2)

    d3gdddp2dt = -pow(kf,3.0)/(3.0*pow(math.pi,2.0))*gdomd2*1.0/en*(d2mstardp2 - 2.0*dmstardp/en*dendp + 2.0*mstar/pow(en,2.0)*pow(dendp,2.0) - mstar/en*d2endp2)*pow(alpha_d,-1.0) \
                -pow(kf,3.0)/(3.0*pow(math.pi,2.0))*gdomd2*mstar/en*(2.0*pow(alpha_d,-1.0)*pow(dalpha_ddp,2.0) - d2alpha_ddp2)*pow(alpha_d,-2.0) \
                + 2.0*pow(kf,3.0)/(3.0*pow(math.pi,2.0))*gdomd2/en*(dmstardp - mstar/en*dendp)*dalpha_ddp*pow(alpha_d,-2.0) + gdomd2*mstar/en*dalpha_ddp*pow(alpha_d,-2.0) \
                - gdomd2/en*(dmstardp - mstar/en*dendp)*pow(alpha_d,-1.0)

    alpha_p = 1.0 + 2.0*lambda_v*gpomp2*pow(gww,2.0)
    dalpha_pdp = 4.0*lambda_v*gpomp2*gww*dgwwdp
    d2alpha_pdp2 = 4.0*lambda_v*gpomp2*(pow(dgwwdp,2.0) + gww*d2gwwdp2)
    d3gppdp2dt = -pow(kf,3.0)/(3.0*pow(math.pi,2.0))*gpomp2*(2.0*pow(alpha_p,-1.0)*pow(dalpha_pdp,2.0) - d2alpha_pdp2)*pow(alpha_p,-2.0) + gpomp2*dalpha_pdp*pow(alpha_p,-2.0)

    dgdddpdt =  - 0.5*gdomd2*(mstar/en)*pow(alpha_d,-1) - pow(kf,3.0)/(3.0*pow(math.pi,2.0))*gdomd2*(dmstardp/en - mstar/pow(en,2.0)*dendp)*pow(alpha_d,-1.0) \
                + pow(kf,3.0)/(3.0*pow(math.pi,2.0))*pow(gdomd2,2.0)*(mstar/en)*(1.0/(2.0*pow(math.pi,2.0))*dIdp + 4.0*lambda_s*gss*dgssdp)*pow(alpha_d,-2.0)

    d2Sdp2 = 1.0/(6.0*en)*(-0.5*pow(math.pi/kf,4.0) - 2.0*pow(math.pi,2.0)/(kf*en)*dendp + 2.0*pow(kf/en,2.0)*pow(dendp,2.0) - pow(kf,2.0)/en*d2endp2) \
                    + 0.25/en*( d2mstardp2*dgdddt + 2.0/pow(en,2.0)*pow(dendp,2.0)*mstar*dgdddt -mstar/en*d2endp2*dgdddt + mstar*d3gdddp2dt \
                    - 2.0*mstar/en*dendp*dgdddpdt + 2.0*dmstardp*dgdddpdt - 2.0/en*dendp*dmstardp*dgdddt  ) - 0.25*d3gppdp2dt

    dens = 2.0/(3.0*pow(math.pi,2.0))*pow(kf,3.0)
    Ksym = 9.0*pow(dens,2.0)*d2Sdp2

    return Ksym

def Ksym_bisect(gdomd2,kf,J,L,gss,gww,gsoms2,gwomw2,kappa,lambda0,zeta,Ksym,lambda_s):
    gpomp2 = get_gpomp2(kf,J,L,gss,gww,gsoms2,gwomw2,kappa,lambda0,zeta,gdomd2=gdomd2,lambda_s=lambda_s)
    lambda_v = get_lambda_v(kf,J,gss,gww,gpomp2,gdomd2=gdomd2,lambda_s=lambda_s)
    y = get_Ksym(kf,gss,gww,gsoms2,gwomw2,gpomp2,gdomd2,kappa,lambda0,zeta,lambda_v,lambda_s=lambda_s) - Ksym
    return y

def get_gdomd2(kf,J,L,gss,gww,gsoms2,gwomw2,kappa,lambda0,zeta,Ksym,lambda_s=0.0):
    # compute gd2 and the solution on a grid
    x = 1e-8
    gd2_array = np.empty((1000,2))
    for i in range(1000):
        gpomp2 = get_gpomp2(kf,J,L,gss,gww,gsoms2,gwomw2,kappa,lambda0,zeta,gdomd2=x,lambda_s=lambda_s)
        lambda_v = get_lambda_v(kf,J,gss,gww,gpomp2,gdomd2=x,lambda_s=lambda_s)
        y = get_Ksym(kf,gss,gww,gsoms2,gwomw2,gpomp2,x,kappa,lambda0,zeta,lambda_v,lambda_s=lambda_s) - Ksym
        gd2_array[i,0] = x
        gd2_array[i,1] = y
        x = x + 5e-6

    # get a bounds on gdomd2
    i = 0
    sgn = 1.0
    while (sgn > 0):
        sgn = gd2_array[999-i][1]*gd2_array[999-i-1][1]
        min = gd2_array[999-i-1][0]
        max = gd2_array[999-i][0]
        i = i+1
        if (i == 999):
            print("no zero exists gd2 for J,L,Ksym: ",J,L,Ksym)
            return -1.0

    # use bounds in bisection
    sol = optimize.bisect(Ksym_bisect,min,max,args=(kf,J,L,gss,gww,gsoms2,gwomw2,kappa,lambda0,zeta,Ksym,lambda_s))
    return sol

# Solves for the model parameters for a given set of bulk properties
def get_parameters(BA, p0, mstar, K, J, L, Ksym, zeta, ms, mw, mp, md):
    mNuc = 939.0
    mstar = mstar*mNuc
    p0 = p0*197.32698**3    # convert density from 1/fm3 to MeV^3
    kf = pow(3.0/2.0*math.pi**2*p0,1.0/3.0)     # get fermi momentum

    gss = mNuc - mstar      # given mstar get gss
    en = np.sqrt(kf**2.0 + mstar**2.0)
    gww = mNuc + BA - en      # get gww at saturation

    gwomw2 = gww/(p0-1/6*zeta*gww**3.0)     # get (gw/mw)^2

    # alphas, betas, gammas
    a1 = gss
    a2 = 0.5*gss**2.0
    a3 = 1/6*gss**3.0
    b1 = 1.0/24.0*gss**4.0
    g1 = 1.0

    # scalar densities
    sdensp = scalardens(kf,mstar)
    sdensn = scalardens(kf,mstar)
    sdens = sdensp + sdensn

    # algebraically convenient expressions continued
    c1 = sdens
    c2 = p0*(mNuc+BA) - endensity_integral(kf,mstar) - endensity_integral(kf,mstar) - 0.5*gww**2/gwomw2 - 1.0/8.0*zeta*gww**4
    integral = common_integral(kf,mstar)
    tau = math.pi**2/(2.0*kf)*en/mstar**2 + gwomw2*(en/mstar)**2.0*(1.0+0.5*zeta*gwomw2*gww**2)**(-1.0) - math.pi**2/(6.0*kf**3)*(en/mstar)**2.0*K
    c3 = pow(tau,-1.0) - 2.0/math.pi**2*integral

    # algebraic expressions for the sigma coupling constants
    gsoms2 = -(a2*a2*a2 - 2.0*a1*a2*a3 + a1*a1*b1 + a3*a3*g1 - a2*b1*g1)/(a2*a3*c1 - a1*b1*c1 - a2*a2*c2 + a1*a3*c2 - a3*a3*c3 + a2*b1*c3)
    kappa = -(-a2*a2*c1 + a1*a2*c2 + a2*a3*c3 - a1*b1*c3 + b1*c1*g1 - a3*c2*g1)/(a2*a2*a2 - 2.0*a1*a2*a3 + a1*a1*b1 + a3*a3*g1 - a2*b1*g1)
    lambda0 = -(a1*a2*c1 - a1*a1*c2 - a2*a2*c3 + a1*a3*c3 - a3*c1*g1 + a2*c2*g1)/(a2*a2*a2 - 2.0*a1*a2*a3 + a1*a1*b1 + a3*a3*g1 - a2*b1*g1)
    
    # get delta coupling
    gdomd2 = get_gdomd2(kf,J,L,gss,gww,gsoms2,gwomw2,kappa,lambda0,zeta,Ksym)

    # isovector couplings
    gpomp2 = get_gpomp2(kf,J,L,gss,gww,gsoms2,gwomw2,kappa,lambda0,zeta,gdomd2=gdomd2)
    lambda_v = get_lambda_v(kf,J,gss,gww,gpomp2,gdomd2=gdomd2)
    
    # store the coupling constants in a list
    fin_couplings = np.zeros(9)
    fin_couplings[0] = ms
    fin_couplings[1] = gsoms2*ms**2
    fin_couplings[2] = gwomw2*mw**2
    fin_couplings[3] = gpomp2*mp**2
    fin_couplings[4] = gdomd2*md**2
    fin_couplings[5] = kappa
    fin_couplings[6] = lambda0
    fin_couplings[7] = zeta
    fin_couplings[8] = lambda_v

    # flag the couplings if they become imaginary
    if (gsoms2<0 or gwomw2<0 or gpomp2<0 or gdomd2<0):
        flag = True
    else:
        flag = False

    return fin_couplings, flag