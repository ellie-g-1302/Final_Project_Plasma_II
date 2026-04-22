import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import thermal

## Data --------------------------------------------------------------------------------
N = 1000
tele = np.linspace(0.001,10,N) * 11600 * 1000
tion = np.linspace(0.001, 10,N) * 11600
nele = 1e30
Z = 1
A = 2.5
nion = Z * nele

## So we're gonna work at the low temperature, high density limit

## Calculating the Data logLambda-------------------------------------------------------
ls = [thermal.Conductivity(tele[i], tion[i], nion, nele, Z, A).calcLogLambda("ls") for i in range(N)]
lm = [thermal.Conductivity(tele[i], tion[i], nion, nele, Z, A).calcLogLambda("lm") for i in range(N)]
gms = [thermal.Conductivity(tele, tion, nion, nele, Z, A).calcLogLambda("gms") for i in range(N)]

## Calculating the Electrical Conductivity ---------------------------------------------
lsSigmaElectric = [thermal.Conductivity(tele[i], tion[i], nion, nele, Z, A).SpitzerElectricConductivity("ls") for i in range(N)]
lmSigmaElectric = [thermal.Conductivity(tele[i], tion[i], nion, nele, Z, A).LeeMoreElectricConductivity("lm") for i in range(N)]

## Calculating the Thermal Conductivity ------------------------------------------------
lsSigmaThermal = [thermal.Conductivity(tele[i], tion[i], nion, nele, Z, A).SpitzerThermalConductivity("ls") for i in range(N)]
lmSigmaThermal = [thermal.Conductivity(tele[i], tion[i], nion, nele, Z, A).LeeMoreThermalConductivity("lm") for i in range(N)]

## Physical interpretation -------------------------------------------------------------
MagneticReynoldsLS = [thermal.Conductivity(tele[i], tion[i], nion, nele, Z, A).magneticReynoldsNumber(100, 0.01, "ls") for i in range(N)]
MagneticReynoldsLM = [thermal.Conductivity(tele[i], tion[i], nion, nele, Z, A).magneticReynoldsNumber(100, 0.01, "lm") for i in range(N)]
BLS = [thermal.Conductivity(tele[i], tion[i], nion, nele, Z, A).OhmsLaw("ls", 100, 10) for i in range(N)]
BLM = [thermal.Conductivity(tele[i], tion[i], nion, nele, Z, A).OhmsLaw("lm", 100, 10) for i in range(N)]

## ChatGPT Generated Code for Graphing plots (edited)
# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Top-left subplot
axs[0, 0].plot(tele, BLS,c="#7CFFCB")
axs[0, 0].plot(tele, BLM, c="#A2003B")
axs[0, 0].set_xscale("log")
axs[0, 0].set_yscale("log")
axs[0, 0].set_title("Magnetic Field")

# # Top-right subplot
axs[0, 1].plot(tele, lsSigmaElectric,c="#7CFFCB")
axs[0, 1].plot(tele, lmSigmaElectric, c="#A2003B")
axs[0, 1].set_title("Electrical Conductivity")
axs[0, 1].set_xscale("log")
axs[0, 1].set_yscale("log")

# # Bottom-left subplot
axs[1, 0].plot(tele, lsSigmaThermal, c="#7CFFCB")
axs[1, 0].plot(tele, lmSigmaThermal, c="#A2003B")
axs[1, 0].set_title("Thermal Conductivity")
axs[1, 0].set_xscale("log")
axs[1, 0].set_yscale("log")


# # Bottom-right subplot
axs[1, 1].plot(tele, MagneticReynoldsLS, c="#7CFFCB")
axs[1, 1].plot(tele, MagneticReynoldsLM, c="#A2003B")
axs[1, 1].set_title("Magnetic Reynolds")
axs[1, 1].set_xscale("log")
axs[1, 1].set_yscale("log")

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# plt.plot(nele, ls, label = "Spizter")
# plt.plot(nele, lm, label = "LeeMore")
# plt.plot(nele, gms, label = "GMS")
# plt.plot(nele, lsSigma, label = "Spitzer Conductivity")
# plt.plot(nele, lmSigma, label = "Lee-More Conductivity")
# plt.legend()
# plt.grid(which="major", color = "#636363")
# plt.grid(which="minor", linestyle = "dashed", color="#bfbfbf")
# plt.show()





    
# plt.plot(nele, LS, c = "#940034", label = "Spitzer")
# plt.plot(nele, LM, c = "#B9A6FF", label="LeeMore")
# plt.title("Thermal Conductivity")
# plt.xlabel("Electron Density")
# plt.legend()
# plt.ylabel("Thermal Conductivity")
# # plt.yscale("log")
# # plt.xscale("log")
# plt.savefig("/Users/admin/Desktop/Plasma/nolog.png", dpi = 800)

    
