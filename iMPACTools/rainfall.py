"""
@author: Andres Peñuela
contact: apenuela@uco.es
"""

import numpy as np  # Import numpy for array operations if using numpy functions

def net_rainfall(PI, rain, num_rain_days=None):
    """
    Calculate the net rainfall after losses and the mean daily rainfall.
    
    Parameters:
    - PI: float, plant interception (e.g., 0.1 for 10% interception)
    - rain: int, float, list, tuple or numpy array representing rainfall data
    - num_rain_days: int, optional, the number of days with rainfall data provided
    
    Returns:
    - total_rain: float, total net rainfall after losses
    - mean_daily_rain: float, average daily net rainfall over the rainy days
    """

    # Case 1: If rain is a single numeric value and the number of rain days is provided
    if isinstance(rain, (int, float)) and num_rain_days:
        rain_net = rain * (1 - PI)  # Apply the loss percentage to the rainfall
        total_rain = rain_net  # Total rain is just the net rain since it's a single value
        print(rain_net, total_rain, num_rain_days)  # Print intermediate values (can be removed)
        mean_daily_rain = total_rain / num_rain_days  # Calculate the mean daily rain

    # Case 2: If rain is a list, tuple, or numpy array
    elif isinstance(rain, (list, tuple, np.ndarray)):
        rain_net = np.array(rain) * (1 - PI)  # Apply the loss percentage to the rainfall
        num_rain_days = np.sum(rain_net > 1)  # Count days with significant rainfall (net > 1)
        total_rain = np.sum(rain_net)  # Sum of all net rainfall values
        mean_daily_rain = total_rain / num_rain_days  # Calculate mean daily rain

    else:
        raise ValueError("Unsupported type for 'rain'. Must be either int, float (and provide a value for num_rain_days), or list, tuple, numpy array.")

    return total_rain, mean_daily_rain

def rainfall_energy(total_rain, canopy_cover, rain_intensity, plant_height):

    # Direct throughfall
    DT = total_rain*(1 - canopy_cover)
    DT_energy = DT*(11.9+8.73*np.log10(rain_intensity))
    
    # Lead drainage
    LD = total_rain*canopy_cover
    LD_energy = LD*(15.8 * plant_height - 5.87)
    
    rain_energy = DT_energy + LD_energy
    
    return rain_energy
      
# --- EI30 calculation function ---
def calculate_ei30(event_group):
    total_rain = event_group['rain'].sum()
    max_15 = event_group['rain'].rolling(3, min_periods=1).sum().max()
    max_30 = event_group['rain'].rolling(6, min_periods=1).sum().max()
    
    # Check erosive event criteria
    if (total_rain < 12.7) and not ((max_15 >= 6.35) or (max_30 >= 12.7)):
        return 0.0
    
    # Intensities
    I10 = event_group['rain'] / (10/60)  # mm/h
    ir = I10  # interval intensity
    
    # Unit rainfall energy (MJ/ha/mm)
    er = 0.29 * (1 - 0.72 * np.exp(-0.05 * ir))
    vr = event_group['rain']  # mm per interval
    
    # Total event energy
    E = (er * vr).sum()
    
    # Max 30-min intensity (sum of 3 consecutive 10-min intensities)
    I30 = I10.rolling(3, min_periods=1).sum().max()
    
    return E * I30 if pd.notnull(I30) else 0.0