import math
import numpy as np
import matplotlib.pyplot as plt

hx_hbar = 1.05457e-27
hx_mele = 9.10938e-28
hx_boltz = 1.38065e-16
hx_qele = 4.80320e-10
gamma_E = 0.57721566490153286060651209008240243

# Functions for qLB
def calc_fermi_energy(nion):
    fermi_energy = hx_hbar ** 2.0 * (3.0 * math.pi ** 2.0 * nion) ** (2./3.) / 2.0 / hx_mele
    return fermi_energy
  

def calc_theta(tele, nion):
    fermi_energy = calc_fermi_energy(nion)
    theta = (tele * hx_boltz) / fermi_energy
    return theta
    
def calc_mu(theta):
    A = -3./2. * math.log((theta), math.e)
    B = math.log((4./3./math.sqrt(math.pi)), math.e)
    C = 0.25054 * theta ** (-1.858) + 0.072 * theta ** (-1.858/2.0)
    D = 1 + 0.25054 * theta ** (-0.858)
    mu_ichimaru = (A + B + C/D)
    return -mu_ichimaru
 
def calc_U_one(tele, nion):
    theta = calc_theta(tele, nion)
    mu = calc_mu(theta)
    a = [-0.0617725,-0.183813,-0.052559,0.0183355,0.0113972,0.00199856,0.00012039]
    mup = [1.0, mu, mu ** 2, mu ** 3, mu ** 4, mu **5 , mu ** 6]
    val = 0

    if mu < -5:
      out1 = 0.949714 * math.exp(mu) * math.log((math.fabs(mu)), math.e)
      
    elif  1 >= mu >= -5:
        for i in range(len(mup)):
            val = mup[i] * a[i] + val
        out1 = val

    elif mu > 1:
      out1 = -gamma_E * math.tanh(0.4753 * (mu + 0.04989))

    return out1


def calc_U_two(tele, nion):
    theta = calc_theta(tele, nion)
    mu = calc_mu(theta)
    val = 0
  
    mup = [1.0, mu, mu ** 2, mu ** 3, mu ** 4, mu **5 , mu ** 6]
    a = [-0.118312,0.0823933,0.0971156,0.013315,-0.00760402,-0.00246233,-0.000203598]

    if mu < -4:
        out2 = 1.16511 * math.exp(mu) / mu

    elif 1 >= mu >= -4:
        for i in range(len(mup)):
            val = mup[i] * a[i] + val
        out2 = val
    
    elif mu > 1: 
        out2 = gamma_E * math.tanh(0.4914 * (mu - 0.772571))
  
    return out2

def calc_beta_eff_approx(tele, nion):
  fermi_energy = calc_fermi_energy(nion)
  p = 2
  beta_eff_approx = 1. / ((tele * hx_boltz) ** p + (2./3. * fermi_energy) ** p) ** (1./p)

  return beta_eff_approx

def calc_f_tilda(x, tele, nion):
    theta = calc_theta(tele, nion)
    out1 = calc_U_one(tele, nion)
    out2 = calc_U_two(tele, nion)
    mu = calc_mu(theta)

    mu_eff = mu - x
    BB = 1 + math.exp(mu_eff)
    A = out1
    B = math.exp(mu_eff) / BB
    C = math.exp(mu_eff) / BB * math.log(x, math.e)
    D = math.exp(2.0 * mu_eff) / BB ** 2 * x * math.log(x, math.e)
    E = (2.0 * math.exp(2.0 * mu_eff) / BB ** 2 - out2) * x
    F = (math.exp(2.0 * mu_eff) * ( -1.0 + math.exp(mu_eff)) / 4.0 / BB ** 3) * x ** 2
    G = (math.exp(2.0 * mu_eff) * (1.0 - 4.0 * math.exp(mu_eff) + math.exp(2.0 * mu_eff)) / 36.0 / BB ** 4) * x ** 3
        
    f_tilda = math.exp(x) * (A - B - C - D + E + F -G)
    return f_tilda

def calc_dTi_dt_Scullard(nion, tele, tion, zbar, abar):
    fermi_energy = calc_fermi_energy(nion)
    theta = calc_theta(tele, nion)
    mu = calc_mu(theta)
    beta_eff_approx = calc_beta_eff_approx(tele, nion)
        


    mion = abar * 1.6e-24
    beta_e = 1.0/(tele * hx_boltz)
    beta_i = 1.0/(tion * hx_boltz)
    gamma = beta_e / beta_i
    lambda_Q = math.sqrt(hx_hbar ** 2 * beta_i / 8.0 / hx_mele)
    lambda_D = math.sqrt(1.0 / 4.0 / math.pi / hx_qele ** 2 / zbar ** 2 / nion / beta_i)
    eta = lambda_Q / lambda_D
    beta_eff = beta_eff_approx
    gamma_eff = beta_eff / beta_i
    lambda_val = zbar / (gamma_eff * eta ** 2 * (hx_mele / mion + gamma))

    f_tilda = calc_f_tilda(1.0/lambda_val, tele, nion) 

    A = -4./3. * hx_qele ** 4 * zbar ** 2 * hx_mele ** 2 * math.exp(-mu) / math.pi / hx_hbar ** 3 / beta_e / mion
    B = tion / tele - 1
    C = f_tilda

    dT = A * B * C
    return dT


def logLambda_qLB(nion, tele, tion, zbar, abar):
    theta = calc_theta(tele, nion)
    mu = calc_mu(theta)
    dT = calc_dTi_dt_Scullard(nion, tele, tion, zbar, abar)
    
    mion = abar * 1.6e-24
    A = 8./3. * zbar ** 2 * hx_qele ** 4 * hx_mele ** 2 / math.pi / hx_hbar ** 3 / mion
    B = 1.0 / (1.0 + math.exp(mu))
    C = hx_boltz * (tele - tion)


    ll =  dT / A / B / C
    return ll

# Functions for Born

def loglambda_born(tele, nele):
    ll_floor = 1.0
    gamma = 0.57721566490153286060651209008240243
    debye = ((hx_boltz * tele)/(4.0*math.pi*nele*hx_qele**2)) ** (1.0/2.0)
    thermal_len = hx_hbar / (2.0*hx_mele*hx_boltz*tele) ** (1.0/2.0)
    lambda_d = debye / thermal_len
    alpha = math.exp(1.0) ** (1.0/(4.0*lambda_d ** 2.0))
    beta = (1.0+(1.0/(4.0*lambda_d**2.0)))
    x = 1.0/(4.0*lambda_d ** 2.0)

    n = 100

    multi = 1

    for i in range(1,n,1):
        if i == 0:
            multi = 1
        else:
            multi = i * multi 
        val = (((-1.0) ** (i))*(x**i)) / (i * multi)  
    

    y = -1.0 * gamma - math.log(x, math.e) - val
    
    ll = max((1.0/2.0) * (alpha * y * beta - 1.0),ll_floor)
    
    return ll

# Functions for BPS:
def loglambda_BPS(nion, tele, tion, zbar, abar):
    theta = calc_theta(tele, nion)
    mu = calc_mu(theta)
    ll_born = loglambda_born(tele, zbar * nion)
    A = (-mu/(tele * hx_boltz))
    if A > 1000:
        A = - A

    ll = ll_born + math.exp(A) * -(1.0 - 1.0/(2.0 ** (3.0/2.0)))*ll_born + math.log((2.0)/2 + 1.0/2.0 ** (5.0/2.0), math.e)
    return ll

# GMS Function

def logLambda_GMS(tele, nele, nion, zbar):

    debye_len = ((tele * hx_boltz)/(4.0 * math.pi * nele * hx_qele ** 2)) ** (1./2)
    ion_rad   = (3.0/(4.0 * math.pi * nion)) ** (1./3)
    lan_len   = (zbar * hx_qele ** 2) / (tele * hx_boltz)
    db_len    = hx_hbar / 2.0 / hx_mele / ((tele * hx_boltz) /hx_mele) ** (1./2)

    ll = (0.5) * math.log((1.0 + (debye_len ** 2 + ion_rad ** 2) / (db_len ** 2 + lan_len ** 2)), math.e)

    return ll

# Spizter Function

def loglambda_Spitzer(tele, nele, zbar):
    ll_floor = 1.0
    
    bmax = math.sqrt(hx_boltz * tele / (4* math.pi * hx_qele**2 * nele))
    
    bmin_classic = zbar * hx_qele**2 / (3*hx_boltz*tele)
    bmin_quantum = hx_hbar / (2*math.sqrt(3*hx_boltz*tele*hx_mele))
    bmin = max(bmin_classic, bmin_quantum)

    ll = max(math.log((bmax/bmin), math.e), ll_floor)
    return ll

# LeeMore Function 

def loglambda_LeeMore(tele, nele, tion, zbar):
    ll_floor = 1.0

    nion = nele/zbar
    R0 = (4*math.pi*nion/3)**(-1./3)

    Tf = hx_hbar**2 / (2*hx_mele) * (3 * math.pi **2 * nele)**(2./3) / hx_boltz; 
    debye_len = 1/math.sqrt((4* math.pi * hx_qele**2 * nele)/(hx_boltz*math.sqrt(tele**2 + Tf**2)) + 4*math.pi*nion *zbar**2 *hx_qele**2 /(hx_boltz*tion) )
    bmax = max(debye_len,R0)
    bmin_classic = zbar * hx_qele**2 / (3*hx_boltz*tele)

    bmin_quantum = hx_hbar / (2*math.sqrt(3*hx_boltz*tele*hx_mele))

    bmin = max(bmin_classic, bmin_quantum)

    ll = max(0.5*math.log((1+(bmax/bmin)**2), math.e), ll_floor)
    return ll




hall = []
navo = 6.023e23


tion = 300 
tele = 250 * 11600
B = 30 * 10000
# nele = [1.00e18,1.53e18,2.33e18,3.56e18,5.43e18,
nele =  [8.29e19,1.26e19,1.93e19,2.95e19,4.50e19,	
        6.87e20,1.05e20,1.60e20,2.44e20,3.73e20]	
        # 5.69e21,8.69e21,1.33e21,2.02e21,3.09e21]
        # 4.71e22,7.20e22,1.10e22,1.68e22,2.56e22,	
        # 3.91e23,5.96e23,9.10e23,1.39e23,2.12e23,
        # 3.24e24,4.94e24,7.54e24,1.15e24,1.76e24,	
        # 2.68e25,4.09e25,6.25e25,9.54e25,1.46e25,	
        # 2.22e26,3.39e26,5.18e26,7.91e26,1.21e26]
zbar = 1.0
abar = 2.515

def cyclotron_frequency(B):
    cyc_freq = hx_qele * B / hx_mele
    return cyc_freq

def eq_time(nele, zbar, tele, logLambda):
    A = (4/3) * (2 * math.pi / hx_mele) ** (1/2)
    num = nele * zbar * hx_qele ** 4 * logLambda
    denom = (hx_boltz * tele) ** (3/2)
    return A * (num / denom)

ll_GMS = []
ll_qLB = []
ll_LM = []
ll_born = []
ll_BPS = []
ll_LS = []
n = 0


for i in range(len(nele)):
    ll_GMS.append(logLambda_GMS(tele, nele[i], zbar*nele[i], zbar))
    ll_qLB.append(logLambda_qLB(nele[i] * zbar, tele, tion, zbar, abar))
    ll_LM.append(loglambda_LeeMore(tele, nele[i], tion, zbar))
    ll_born.append(loglambda_born(tele, nele[i]))
    ll_BPS.append(loglambda_BPS(nele[i] * zbar, tele, tion, zbar, abar))
    ll_LS.append(loglambda_Spitzer(tele, nele[i], zbar))

   

hall_GMS = []
hall_qLB = []
hall_LM = []
hall_born = []
hall_BPS = []
hall_LS = []
omega = cyclotron_frequency(B)
ie_time_GMS = []
ie_time_qLB = []
ie_time_LM = []
ie_time_born = []
ie_time_BPS = []
ie_time_LS = []

for i in range(len(nele)):
    ie_GMS = eq_time(nele[i] * zbar, zbar, tele, ll_GMS[i])
    ie_qLB = eq_time(nele[i] * zbar, zbar, tele, ll_qLB[i])
    ie_LM = eq_time(nele[i] * zbar, zbar, tele, ll_LM[i])
    ie_born = eq_time(nele[i] * zbar, zbar, tele, ll_born[i])
    ie_BPS = eq_time(nele[i] * zbar, zbar, tele, ll_BPS[i])
    ie_LS = eq_time(nele[i] * zbar, zbar, tele, ll_LS[i]) 
    hall_GMS.append(omega * 1/ie_GMS)
    hall_qLB.append(omega * 1/ie_qLB)
    hall_LM.append(omega * 1/ie_LM)
    hall_born.append(omega * 1/ie_born)
    hall_BPS.append(omega * 1/ie_BPS)
    hall_LS.append(omega * 1/ie_LS)
    ie_time_born.append(ie_born)
    ie_time_BPS.append(ie_BPS)
    ie_time_GMS.append(ie_GMS)
    ie_time_LM.append(ie_LM)
    ie_time_LS.append(ie_LS)
    ie_time_qLB.append(ie_qLB)

  

    
    
    
# for i in range(len(nele)):
    # print("GMS", f'{ll_GMS[i]:.2f}', "|", "qLB", f'{ll_qLB[i]:.2f}', "|", "Born", f'{ll_born[i]:.2f}',  "|", "BPS", f'{ll_BPS[i]:.2f}', "|", "LS", f'{ll_LS[i]:.2f}', "|", "LM", f'{ll_LM[i]:.2f}')


# plt.plot(ll_GMS, hall_GMS, label = "GMS", c = (1, 0.129, 0.42), linestyle = "dashdot")
# plt.plot(ll_qLB, hall_qLB, label = "qLB", c= (0.522, 0.118, 0.851), linestyle="dotted")
# plt.plot(ll_LM, hall_LM, label = "LM", c = (0, 0.812, 1))
# plt.plot(ll_born, hall_born, label = "Born", c=(0, 0.812, 0.149), linestyle="dashed")
# plt.plot(ll_BPS, hall_BPS, label = "BPS", c=(0.169, 0.29, 0.271))
# plt.plot(ll_LS, hall_LS, label = "LS", c=(1, 0.8, 0.341))
# # plt.axvline(x=1.25e26, color=(0.549, 0.051, 0.259), linestyle='--', linewidth=2)
# plt.yscale("log")
# #lt.xscale("log")
# plt.xlabel("Log Lambda")
# plt.ylabel("Hall Parameter")
# plt.legend()
# plt.show()

# plt.plot(nele, ll_GMS, label = "GMS", c = (1, 0.129, 0.42), linestyle = "dashdot")
# plt.plot(nele, ll_qLB, label = "qLB", c= (0.522, 0.118, 0.851), linestyle="dotted")
# plt.plot(nele, ll_LM, label = "LM", c = (0, 0.812, 1))
# plt.plot(nele, ll_born, label = "Born", c=(0, 0.812, 0.149), linestyle="dashed")
# plt.plot(nele, ll_BPS, label = "BPS", c=(0.169, 0.29, 0.271))
# plt.plot(nele, ll_LS, label = "LS", c=(1, 0.8, 0.341))
# plt.xscale("log")
# plt.xlabel("Electron Density 1/cc")
# plt.ylabel("Log Lambda")
# plt.legend()
# plt.show()

plt.plot(nele, hall_GMS, label = "GMS", c = (1, 0.129, 0.42), linestyle = "dashdot")
plt.plot(nele, hall_qLB, label = "qLB", c= (0.522, 0.118, 0.851), linestyle="dotted")
plt.plot(nele, hall_LM, label = "LM", c = (0, 0.812, 1))
plt.plot(nele, hall_born, label = "Born", c=(0, 0.812, 0.149), linestyle="dashed")
plt.plot(nele, hall_BPS, label = "BPS", c=(0.169, 0.29, 0.271))
plt.plot(nele, hall_LM, label = "LS", c=(1, 0.8, 0.341)) 
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Electron Density 1/cc")
plt.ylabel("Hall Parameter")
plt.legend()
plt.show()

# plt.plot(nele, ie_time_GMS, label = "GMS", c = (1, 0.129, 0.42), linestyle = "dashdot")
# plt.plot(nele, ie_time_qLB, label = "qLB", c= (0.522, 0.118, 0.851), linestyle="dotted")
# plt.plot(nele, ie_time_LM, label = "LM", c = (0, 0.812, 1))
# plt.plot(nele, ie_time_born, label = "Born", c=(0, 0.812, 0.149), linestyle="dashed")
# plt.plot(nele, ie_time_BPS, label = "BPS", c=(0.169, 0.29, 0.271))
# plt.plot(nele, ie_time_LS, label = "LS", c=(1, 0.8, 0.341))
# plt.xscale("log")
# plt.yscale("log")
# # plt.axvline(x=1.25e26, color=(0.549, 0.051, 0.259), linestyle='--', linewidth=2)
# plt.xlabel("Electron Density 1/cc")
# plt.ylabel("Equilibration Time")
# plt.legend()
# plt.savefig("ie_time.png", dpi = 800)
# #LM is messed up and equilibration time is increasing when should decrease
