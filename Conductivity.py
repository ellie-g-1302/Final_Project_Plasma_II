import math
import numpy as np
import matplotlib.pyplot as plt

class Conductivity:

    def __init__(self, tele, tion, nion, nele, Z, A):
        self.Ti = tion
        self.Te = tele
        self.ne = nele
        self.ni = nion
        self.Z = Z
        self.A = A
    ## Constants 
    hx_hbar = 1.05457e-27
    hx_mele = 9.10938e-28
    hx_boltz = 1.38065e-16
    hx_qele = 4.80320e-10
    gamma_E = 0.57721566490153286060651209008240243

    # Functions for qLB
    def calc_fermi_energy(self):
        fermi_energy = self.hx_hbar ** 2.0 * (3.0 * np.pi ** 2.0 * self.ni) ** (2./3.) / 2.0 / self.hx_mele
        return fermi_energy

    def calc_theta(self):
        fermi_energy = Conductivity.calc_fermi_energy(self)
        theta = (self.Te * self.hx_boltz) / fermi_energy
        return theta
    
    def calc_mu(self):
        theta = Conductivity.calc_theta(self)
        A = -3./2. * np.log((theta))
        B = np.log((4./3./np.sqrt(np.pi)))
        C = 0.25054 * theta ** (-1.858) + 0.072 * theta ** (-1.858/2.0)
        D = 1 + 0.25054 * theta ** (-0.858)
        mu_ichimaru = (A + B + C/D)
        return -mu_ichimaru
    
    def calc_U_one(self):
        theta = Conductivity.calc_theta(self)
        mu = Conductivity.calc_mu(self)
        a = [-0.0617725,-0.183813,-0.052559,0.0183355,0.0113972,0.00199856,0.00012039]
        mup = [1.0, mu, mu ** 2, mu ** 3, mu ** 4, mu **5 , mu ** 6]
        val = 0

        if mu < -5:
          out1 = 0.949714 * np.exp(mu) * np.log((np.fabs(mu)))
        
        elif  1 >= mu >= -5:
            for i in range(len(mup)):
                val = mup[i] * a[i] + val
            out1 = val

        elif mu > 1:
          out1 = -self.gamma_E * np.tanh(0.4753 * (mu + 0.04989))

        return out1


    def calc_U_two(self):
        theta = Conductivity.calc_theta(self)
        mu = Conductivity.calc_mu(self)
        val = 0
    
        mup = [1.0, mu, mu ** 2, mu ** 3, mu ** 4, mu **5 , mu ** 6]
        a = [-0.118312,0.0823933,0.0971156,0.013315,-0.00760402,-0.00246233,-0.000203598]

        if mu < -4:
            out2 = 1.16511 * np.exp(mu) / mu

        elif 1 >= mu >= -4:
            for i in range(len(mup)):
                val = mup[i] * a[i] + val
            out2 = val
        
        elif mu > 1: 
            out2 = self.gamma_E * np.tanh(0.4914 * (mu - 0.772571))
    
        return out2

    def calc_beta_eff_approx(self):
      fermi_energy = Conductivity.calc_fermi_energy(self)
      p = 2
      beta_eff_approx = 1. / ((self.Te * self.hx_boltz) ** p + (2./3. * fermi_energy) ** p) ** (1./p)

      return beta_eff_approx

    def calc_f_tilda(self, x):
        theta = Conductivity.calc_theta(self)
        out1 = Conductivity.calc_U_one(self)
        out2 = Conductivity.calc_U_two(self)
        mu = Conductivity.calc_mu(self)

        mu_eff = mu - x
        BB = 1 + np.exp(mu_eff)
        A = out1
        B = np.exp(mu_eff) / BB
        C = np.exp(mu_eff) / BB * np.log(x)
        D = np.exp(2.0 * mu_eff) / BB ** 2 * x * np.log(x)
        E = (2.0 * np.exp(2.0 * mu_eff) / BB ** 2 - out2) * x
        F = (np.exp(2.0 * mu_eff) * ( -1.0 + np.exp(mu_eff)) / 4.0 / BB ** 3) * x ** 2
        G = (np.exp(2.0 * mu_eff) * (1.0 - 4.0 * np.exp(mu_eff) + np.exp(2.0 * mu_eff)) / 36.0 / BB ** 4) * x ** 3
            
        f_tilda = np.exp(x) * (A - B - C - D + E + F -G)
        return f_tilda

    def calc_dTi_dt_Scullard(self):
        fermi_energy = Conductivity.calc_fermi_energy(self)
        theta = Conductivity.calc_theta(self)
        mu = Conductivity.calc_mu(self)
        beta_eff_approx = Conductivity.calc_beta_eff_approx(self)
            


        mion = self.A * 1.6e-24
        beta_e = 1.0/(self.Te * self.hx_boltz)
        beta_i = 1.0/(self.Ti * self.hx_boltz)
        gamma = beta_e / beta_i
        lambda_Q = np.sqrt(self.hx_hbar ** 2 * beta_i / 8.0 / self.hx_mele)
        lambda_D = np.sqrt(1.0 / 4.0 / np.pi / self.hx_qele ** 2 / self.Z ** 2 / self.ni / beta_i)
        eta = lambda_Q / lambda_D
        beta_eff = beta_eff_approx
        gamma_eff = beta_eff / beta_i
        lambda_val = self.Z / (gamma_eff * eta ** 2 * (self.hx_mele / mion + gamma))

        f_tilda = Conductivity.calc_f_tilda(self, 1.0/lambda_val) 

        A = -4./3. * self.hx_qele ** 4 * self.Z ** 2 * self.hx_mele ** 2 * np.exp(-mu) / np.pi / self.hx_hbar ** 3 / beta_e / mion
        B = self.Ti / self.Te - 1
        C = f_tilda

        dT = A * B * C
        return dT


    def logLambda_qLB(self):
        theta = Conductivity.calc_theta(self)
        mu = Conductivity.calc_mu(self)
        dT = Conductivity.calc_dTi_dt_Scullard(self)
        
        mion = self.A * 1.6e-24
        A = 8./3. * self.Z ** 2 * self.hx_qele ** 4 * self.hx_mele ** 2 / np.pi / self.hx_hbar ** 3 / mion
        B = 1.0 / (1.0 + np.exp(mu))
        C = self.hx_boltz * (self.Te - self.Ti)

        if dT == 0 or A == 0 or B == 0 or C == 0:
            ll = 0
        else: 
            ll =  dT / A / B / C
        return ll

    # Functions for Born

    def loglambda_born(self):
        ll_floor = 1.0
        gamma = 0.57721566490153286060651209008240243
        debye = ((self.hx_boltz * self.Te)/(4.0*np.pi*self.ne*self.hx_qele**2)) ** (1.0/2.0)
        thermal_len = self.hx_hbar / (2.0*self.hx_mele*self.hx_boltz*self.Te) ** (1.0/2.0)
        lambda_d = debye / thermal_len
        alpha = np.exp(1.0) ** (1.0/(4.0*lambda_d ** 2.0))
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
        

        y = -1.0 * gamma - np.log(x)- val
        
        ll = max((1.0/2.0) * (alpha * y * beta - 1.0),ll_floor)
        
        return ll

    # Functions for BPS:
    def loglambda_BPS(self):
        theta = Conductivity.calc_theta(self)
        mu = Conductivity.calc_mu(self)
        ll_born = Conductivity.loglambda_born(self)
        A = (-mu/(self.Te * self.hx_boltz))
        if A > 1000:
            A = - A

        ll = ll_born + np.exp(A) * -(1.0 - 1.0/(2.0 ** (3.0/2.0)))*ll_born + np.log((2.0)/2 + 1.0/2.0 ** (5.0/2.0))
        return ll

    # GMS Function

    def logLambda_GMS(self):

        debye_len = ((self.Te * self.hx_boltz)/(4.0 * np.pi * self.ne * self.hx_qele ** 2)) ** (1./2)
        ion_rad   = (3.0/(4.0 * np.pi * self.ni)) ** (1./3)
        lan_len   = (self.Z * self.hx_qele ** 2) / (self.Te * self.hx_boltz)
        db_len    = self.hx_hbar / 2.0 / self.hx_mele / ((self.Te* self.hx_boltz) /self.hx_mele) ** (1./2)

        ll = (0.5) * np.log((1.0 + (debye_len ** 2 + ion_rad ** 2) / (db_len ** 2 + lan_len ** 2)))

        return ll

    # Spizter Function

    def loglambda_Spitzer(self):
        ll_floor = 1
        
        bmax = np.sqrt(self.hx_boltz * self.Te / (4* np.pi * self.hx_qele**2 * self.ne))
        
        bmin_classic = self.Z * self.hx_qele**2 / (3*self.hx_boltz*self.Te)
        bmin_quantum = self.hx_hbar / (2*np.sqrt(3*self.hx_boltz*self.Te*self.hx_mele))
        bmin = max(bmin_classic, bmin_quantum)

        ll = max(np.log((bmax/bmin)), ll_floor)
        return ll

# # LeeMore Function 
    def loglambda_LeeMore(self):

        nion = self.ne/self.Z
        R0 = (4*np.pi*nion/3)**(-1./3)

        Tf = self.hx_hbar**2 / (2*self.hx_mele) * (3 * np.pi **2 * self.ne)**(2./3) / self.hx_boltz; 
        debye_len = 1/np.sqrt((4* np.pi * self.hx_qele**2 * self.ne)/(self.hx_boltz*np.sqrt(self.Te**2 + Tf**2)) + 4*np.pi*nion *self.Z**2 *self.hx_qele**2 /(self.hx_boltz*self.Ti) )
        bmax = max(debye_len,R0)

        bmin_classic = self.Z * self.hx_qele**2 / (3*self.hx_boltz*self.Te)

        bmin_quantum = self.hx_hbar / (2*np.sqrt(3*self.hx_boltz*self.Te*self.hx_mele))

        bmin = max(bmin_classic, bmin_quantum)

        ll = 0.5*np.log((1+(bmax/bmin)**2))
        return ll
    
    def calcLogLambda(self, key):
        if key == "LM" or key == "lm":
            ll = Conductivity.loglambda_LeeMore(self)
        elif key == "LS" or key == "ls":
            ll = Conductivity.loglambda_Spitzer(self)
        elif key == "qLB" or key == "qlb" or key == "QLB":
            ll = Conductivity.logLambda_qLB(self)
        elif key == "GMS" or key == "gms":
            ll = Conductivity.logLambda_GMS(self)
        elif key == "BPS" or key == "bps":
            ll = Conductivity.loglambda_BPS(self)
        elif key == "Born" or key == "born":
            ll = Conductivity.logLambda_qLB(self)
        else: 
            print("Error! Improper Formulation")
            return 
        return ll
            
    def eq_time(self, key):
        ll = Conductivity.calcLogLambda(self, key)
        val = (4/3) * (2 * np.pi / self.hx_mele) ** (1/2)
        num = self.ne * self.Z * self.hx_qele ** 4 * ll
        denom = (self.hx_boltz * self.Te) ** (3/2)
        return val * (num / denom), ll
    
    def LeeMoreConductivity(self, key):
        mu_div_kT = Conductivity.calc_mu(self)
        tau = Conductivity.eq_time(self, key)[0]
        a1 = 13.5
        a2 = 0.976
        a3 = 0.437
        b2 = 0.51
        b3 = 0.126
        if mu_div_kT > 20:
            y = mu_div_kT
        elif mu_div_kT < -15:
            y = np.exp(mu_div_kT)
        else:
            y = np.log(1 + np.exp(mu_div_kT))
        
        A_beta = (a1 + a2*y + a3*y**2 ) / (1 + b2*y + b3*y**2)  
        K = (self.ne*self.hx_boltz*(self.hx_boltz * self.Te) * tau) / self.hx_mele * A_beta
        return K
    
    def SpitzerConductivity(self, key):         
        ll = Conductivity.calcLogLambda(self, key)
        const = (8/np.pi) ** (3/2) * (self.hx_boltz**(7/2) / (self.hx_qele**4 * (self.hx_mele) ** (1/2)))
        sigma = (1/(1+3.3/self.Z)) * (self.Te**(5/2)/(self.Z*ll))
        return const * sigma
    
    def SpitzerElectricConductivity(self, key):
        tau = Conductivity.eq_time(self, key)
        sigma = 2 * (self.hx_qele ** 2 * self.ne * tau[0]) / self.hx_mele
        return sigma

    def LeeMoreElectricConductivity(self, key):
        tau = Conductivity.eq_time(self, key)
        A_alpha = 32/3*math.pi ## non-degenrate limit degenerate limit is 1
        sigma = (2 * (self.hx_qele ** 2 * self.ne * tau[0]) / self.hx_mele) * A_alpha
        return sigma
            
    def cyclotron_frequency(self, B):
        cyc_freq = self.hx_qele * B / self.calc_beta_eff_approxhx_mele
        return cyc_freq





    
    

