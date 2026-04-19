import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import thermal

## Data -----------------------------------
tele = 250 * 11600
tion = 300 * 1160
nele = np.linspace(1e21,1e30,100)
Z = 1
A = 2.5
nion = Z * nele


thermCon = thermal.Conductivity(tele, tion, nion, nele, Z, A)
KGMS = thermCon.LeeMoreConductivity("GMS")
# plt.plot(nele, llGMS[1])
# plt.xscale("log")
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

    
