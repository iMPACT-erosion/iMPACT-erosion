"""
@author: Andres Peñuela
contact: apenuela@uco.es
"""

import numpy as np  # Import numpy for array operations if using numpy functions
from numba import njit

def soil_capacity(MS, BD, EHD, Et_Eo):
    """
    Calculate the soil water capacity (mm), i.e. how much water the soil can hold.

    Parameters:
    - MS: float, the soil moisture content at field capacity (e.g., 0.2 for 20%)
    - BD: float, bulk density of the soil (tn/m³)
    - EHD: float, effective holding depth of the soil (m)
    - Et_Eo: float, evapotranspiration ratio (e.g., ratio of actual to potential evapotranspiration)

    Returns:
    - soil_capacity: float, the calculated soil water capacity (in mm)
    """

    # Calculate soil water capacity using the formula:
    # soil_capacity = (1000 * MS * EHD * Et_Eo) / BD
    # 1000 is used to convert depth from meters to millimeters
    soil_capacity = (1000 * MS * EHD * Et_Eo) / BD

    # Note: Division by BD: Reflects the inverse relationship between bulk density and available pore space. 
    # A higher bulk density means the soil is more compact, leading to less pore space available for water storage.

    return soil_capacity

@njit
def runoff(total_rain, mean_daily_rain, soil_capacity, flow_acc, cell_size, flow_rout):
    """
    This function calculates how much runoff (water flow) is generated from rainfall and how it flows across a grid of land.

    Parameters:
    - total_rain: float, total rainfall (mm)
    - mean_daily_rain: float, mean daily rainfall (mm/day)
    - soil_capacity: 2D array, soil water capacity per grid cell (mm)
    - flow_acc: 2D array, flow accumulation per grid cell (number of cells contributing to flow)
    - cell_size: float, size of a grid cell (m)
    - flow_rout: array, indices for flow routing
    
    Returns:
    - Q: 2D array, initial runoff generated in each grid cell
    - Q_rout: 2D array, routed runoff accounting for upstream contributions
    """
    flow_rout_up_row, flow_rout_up_col, flow_rout_down_row, flow_rout_down_col, flow_rout_contrib = flow_rout
    
    # Initial runoff calculation for each grid cell
    # The formula combines rainfall, soil capacity, and flow accumulation
    Q = total_rain * np.exp(-soil_capacity / mean_daily_rain) * (flow_acc / (10 * cell_size)) ** 0.1

    # Initialize Q_rout to store the routed runoff
    # Initially, Q_rout is the same as Q, but it will be updated with contributions from upstream cells
    Q_rout = np.copy(Q)

    # Iterate over each cell in the flow routing order (upstream to downstream)
    for r in range(len(flow_rout_up_row)):
        # Get indices for upstream and downstream cells
        FD_ix_2D_x  = flow_rout_up_row[r]
        FD_ix_2D_y  = flow_rout_up_col[r]
        FD_ixc_2D_x = flow_rout_down_row[r]
        FD_ixc_2D_y = flow_rout_down_col[r]

        # Update the downstream cell with the contribution from the upstream cell
        Q_rout[FD_ixc_2D_x, FD_ixc_2D_y]  = Q_rout[FD_ixc_2D_x, FD_ixc_2D_y] + (Q_rout[FD_ix_2D_x, FD_ix_2D_y]*np.exp(-soil_capacity/mean_daily_rain))*(flow_acc[FD_ix_2D_x, FD_ix_2D_y]/(10*cell_size))**0.1*flow_rout_contrib[r]

    return Q, Q_rout