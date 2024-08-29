"""
@author: Andres Peñuela
contact: apenuela@uco.es
"""

import numpy as np  # Import numpy for array operations if using numpy functions
from numba import njit

def soil_detachment_rain(clay, silt, sand, M, TF, rain_energy):
    SC = np.exp(-0.035 * M)
    K = (0.1*clay + 0.5*silt + 0.3*sand)/100
    F = K*SC*TF*rain_energy*10**-3
    return F

def soil_detachment_runoff(clay, silt, sand, M, TF, Q, slope):
    DR = (1.0*clay + 1.6*silt + 1.5*sand)/100 # always >0
    SC = np.exp(-0.035 * M)
    H = DR*Q**1.5*np.sin(slope)*SC*TF*10**-3
    return H

def transport_capacity(M,TF,Q,slope):
    SC = np.exp(-0.035 * M)
    TC = SC*TF*Q**2*np.sin(slope)*10**-3
    return TC

@njit # Numba decorator to speed-up the function below
def soil_loss(F, H, TC):    
    
    # Variables and matrices initialization
    G = (F+H)
    ST = H.copy() * 0 # sediment transport
    SL  = H.copy() * 0
    SL_net  = H.copy() * 0
    SL_class = H.copy() * 0

    count_DEP = 0
    count_G = 0
    count_TC = 0

    for r in range(len(flow_rout_up_row)-1):
        
        FD_ix_2D_x  = flow_rout_up_row[r]
        FD_ix_2D_y  = flow_rout_up_col[r]
        FD_ixc_2D_x = flow_rout_down_row[r]
        FD_ixc_2D_y = flow_rout_down_col[r]
        
        if np.isnan(TC[FD_ix_2D_x, FD_ix_2D_y]) or np.isnan(G[FD_ix_2D_x, FD_ix_2D_y]):
            SL[FD_ix_2D_x, FD_ix_2D_y] = np.nan

        elif ST[FD_ix_2D_x, FD_ix_2D_y] > TC[FD_ix_2D_x, FD_ix_2D_y]:
            SL[FD_ix_2D_x, FD_ix_2D_y] = SL[FD_ix_2D_x, FD_ix_2D_y] - (ST[FD_ix_2D_x, FD_ix_2D_y] - TC[FD_ix_2D_x, FD_ix_2D_y])*flow_rout_contrib[r]
            ST[FD_ixc_2D_x, FD_ixc_2D_y] = ST[FD_ixc_2D_x, FD_ixc_2D_y] + TC[FD_ix_2D_x, FD_ix_2D_y]*flow_rout_contrib[r]
            SL_class[FD_ix_2D_x, FD_ix_2D_y] = 0
            count_DEP += 1
            
        else:
            if TC[FD_ix_2D_x, FD_ix_2D_y] >= G[FD_ix_2D_x, FD_ix_2D_y]:
                SL[FD_ix_2D_x, FD_ix_2D_y] = SL[FD_ix_2D_x, FD_ix_2D_y] + G[FD_ix_2D_x, FD_ix_2D_y]*flow_rout_contrib[r]
                ST[FD_ixc_2D_x, FD_ixc_2D_y] = ST[FD_ixc_2D_x, FD_ixc_2D_y] + G[FD_ix_2D_x, FD_ix_2D_y]*flow_rout_contrib[r]
                TC[FD_ixc_2D_x, FD_ixc_2D_y] = TC[FD_ixc_2D_x, FD_ixc_2D_y] + (TC[FD_ix_2D_x, FD_ix_2D_y] - G[FD_ix_2D_x, FD_ix_2D_y])*flow_rout_contrib[r]
                SL_class[FD_ix_2D_x, FD_ix_2D_y] = 1
                count_G += 1
            else:
                SL[FD_ix_2D_x, FD_ix_2D_y] = SL[FD_ix_2D_x, FD_ix_2D_y] + TC[FD_ix_2D_x, FD_ix_2D_y]*flow_rout_contrib[r]
                ST[FD_ixc_2D_x, FD_ixc_2D_y] = ST[FD_ixc_2D_x, FD_ixc_2D_y] + TC[FD_ix_2D_x, FD_ix_2D_y]*flow_rout_contrib[r]
                SL_class[FD_ix_2D_x, FD_ix_2D_y] = 2
                count_TC += 1
        
    return G, ST, TC, SL, SL_class
