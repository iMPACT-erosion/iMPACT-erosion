# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:12:11 2024

@author: andro
"""

import numpy as np # NumPy is a library for numerical computing in Python.
import numba       # Numba is a library for accelerating numerical computations.

@numba.njit
def is_sink(x, y, dem):
    """
    
    Function to check if a cell is a sink.

    Imagine you pour water onto the map. Some points might be like little bowls, 
    where water collects because they're surrounded by higher ground on all sides. 
    These are called sinks. This part of the code helps us identify these sink 
    points. It checks if a point on the map is surrounded by higher points on 
    all sides, just like how water would collect in a little depression or hole 
    in the ground.
    
    Args:
    - x, y: Coordinates of the cell in the DEM
    - dem: the DEM itself represented as a 2D NumPy array

    Returns:
    - Tuple (sink, min_neigh_elev):
        - sink: True if the cell is a sink, False otherwise
        - min_neigh_elev: Minimum elevation among the neighboring cells
    """

    min_neigh_elev = np.inf  # Initialize minimum neighboring elevation to infinity
    sink = True  # Initialize sink flag to True, assuming the cell is a sink

    # Iterate over each neighbor of the cell
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:  # Skip the current cell
                continue
                
            nx, ny = x + dx, y + dy  # Calculate coordinates of the neighbor cell
            
            # Check if the neighbor's elevation is valid (not NaN)
            if not np.isnan(dem[nx, ny]):
                # If neighbor's elevation is less than the current cell's elevation,
                # the current cell cannot be a sink
                if dem[nx, ny] < dem[x, y]:
                    sink = False
                # If neighbor's elevation is different and less than the current minimum,
                # update the minimum neighboring elevation
                elif dem[nx, ny] != dem[x, y]:
                    min_neigh_elev = min(min_neigh_elev, dem[nx, ny])
                    
    return sink, min_neigh_elev  # Return the sink flag and the minimum neighboring elevation


@numba.njit
def fill_sinks(dem):
    """
    Function to fill sinks in a Digital Elevation Model (DEM).
    
    First, it finds the lowest point on the map. Then, it goes through each 
    point on the map and checks if it's a sink (where water would stop and get 
    stored). If it finds a sink, it raises that point to the same height as the 
    surrounding lowest point. It keeps doing this until there are no more sinks 
    left on the map. Essentially, it's like filling in low spots on the map to 
    make sure water flows smoothly without getting stuck.
    
    Args:
    - dem: 2D numpy array representing the DEM
    
    Returns:
    - Filled DEM with sinks removed
    """
    rows, cols = dem.shape  # Get the dimensions of the DEM (rows, cols)
    
    while True:  # Repeat until no more sinks are found
        found_sinks = False  # We first assume that no sinks are present
        # Loop through each cell in the DEM (except for the limits of the DEM)
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if not np.isnan(dem[x, y]):  # Check if the elevation is not a NaN value.
                    sink, min_neigh_elev = is_sink(x, y, dem)  # Run function to check if the cell is a sink
                    # Check if the cell is a sink
                    if sink:
                        dem[x, y] = min_neigh_elev  # Update the elevation of the sink cell to the minimum 
                        # elevation among its neighbors.
                        found_sinks = True  # Set the flag to True indicating that a sink has been found 

        # Break the loop if no more sinks are found (found_sinks remains False)
        if not found_sinks:
            break
    return dem
