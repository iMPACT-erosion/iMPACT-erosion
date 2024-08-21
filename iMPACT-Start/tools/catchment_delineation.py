# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:43:22 2024

@author: andro
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def catchment_delineation(dem,flow_direction, outlet_row, outlet_col):
    """
    Delineates the catchment area based on flow direction and the location of the catchment outlet.

    Args:
        dem (numpy.ndarray): 2D array of the digital elevation model.
        flow_direction (numpy.ndarray): 2D array representing flow direction values.
        outlet_row (int): Row index of the catchment outlet cell.
        outlet_col (int): Column index of the catchment outlet cell.

    Returns:
        catchment_mask (numpy.ndarray): Binary mask indicating the delineated catchment area.
        catchment_dem (numpy.ndarray: DEM of the delineated catchment area
    """
    
    ######################
    ### Initialization ###
    ######################
    # Initialize a mask (boolean array) to track visited cells during reverse tracing
    visited = np.zeros_like(flow_direction, dtype=np.bool_)

    # Initialize the catchment mask (boolean array) to represent the catchment area
    catchment_mask = np.zeros_like(flow_direction, dtype=np.bool_)*np.nan

    #######################
    ### Reverse tracing ###
    #######################   
    # Start reverse tracing from the outlet cell and explores neighboring cells recursively
    
    # Create the variable stack to keep track of the cells that need to be explored during the reverse tracing process.
    stack = [(outlet_row, outlet_col)] # Add the first catchment cell (outlet) to the stack
    
    while stack: # It continues this process while there are cells (in stack) to explore.
        
        current_row, current_col = stack.pop() # return and remove the last item from the stack

        # Check if the current cell has been visited
        if visited[current_row, current_col]:
            continue

        # Mark the current cell as visited
        visited[current_row, current_col] = True

        # Mark the current cell as part of the catchment
        catchment_mask[current_row, current_col] = 1
        
        ##########################
        ### Neighbor Selection ###
        ##########################

        # Define a 3x3 array to represent the directions of the valid neighboring cells (connected to the current cell)
        neighbours_dir = np.array([(2,    4,  8),
                                   (1,    0, 16),
                                   (128, 64, 32)], dtype=np.int32)

        # Iterate over neighboring cells in a 3x3 grid centered around the current cell
        
        neighboring_cells = [] # Initialize the variable to add valid neighboring cells
        
        for dr in range(-1, 2): # change in the row index (-1: move up, 0: stay, 1: move down)
            for dc in range(-1, 2): # change in the column index (-1: move left, 0: stay, 1: move right)
                
                if dr == 0 and dc == 0: # exclude the current cell itself
                    continue
                    
                new_row, new_col = current_row + dr, current_col + dc
                
                # Determine valid neighboring cells: check if each neighboring cell satisfies certain conditions
                if (0 <= new_row < flow_direction.shape[0] and # It's within the bounds of the array.
                    0 <= new_col < flow_direction.shape[1] and # It's within the bounds of the array.
                    not visited[new_row, new_col] and # It hasn't been visited yet.
                    flow_direction[new_row, new_col] == neighbours_dir[dr+1, dc+1]): # Its flow direction matches neighbours_dir.         
                    
                    neighboring_cells.append((new_row, new_col)) # Valid neighboring cells are added
        
        #############################
        ### Recursive Exploration ###
        #############################
        # Add all the valid neighboring cells to the stack for further exploration
        stack.extend(neighboring_cells)
        
    catchment_dem = dem*catchment_mask

    # Return the catchment_mask, a boolean array indicating the delineated catchment area
    return catchment_mask,catchment_dem

def plot_catchment_delineation(dem,fd,fa,outlet=0):
    """
    Plot the delineated catchment area overlaid on the flow accumulation map.

    Args:
    - dem: Digital elevation model array
    - fd: Flow direction array.
    - fa: Flow accumulation array.
    - catchment: Index of the catchment to delineate. The lower the index the larger the contributing area

    Returns:
    - catchment_mask: Mask representing the delineated catchment area.
    - catchment_dem: DEM of the delineated catchment
    """
    # Step 1: Flatten the 2D array to a 1D list
    flattened = fa.flatten()

    # Step 2: Sort the flattened list in descending order
    fa_sorted = np.sort(flattened)[::-1] 

    # Find the outlet row and column coordinates for the catchment
    outlet_row, outlet_col = np.where(fa == fa_sorted[outlet])
    
    # Check if outlet coordinates were found, return zeros if not
    if len(outlet_row) == 0 or len(outlet_col) == 0:
        return np.zeros_like(fd, dtype=np.bool_)

    # Run the delineation function to obtain the catchment mask
    catchment_mask,catchment_dem = catchment_delineation(dem, fd, outlet_row[0], outlet_col[0]) 

    # Display the flow accumulation map and catchment area
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Create a transparent colormap for overlaying catchment area
    coolwarm_2 = plt.cm.get_cmap("coolwarm").copy()
    coolwarm_2.set_bad(alpha=0.2)

    # Overlay catchment area on flow accumulation map
    ax1.imshow(fa, cmap='coolwarm')
    ax1.imshow(fa * catchment_mask, cmap=coolwarm_2)
    ax1.plot(outlet_col[0],outlet_row[0], 'ro')
    ax1.set_title('Catchment Area for Flow Accumulation = %.d' % fa_sorted[outlet])
    plt.show()
    
    return catchment_mask,catchment_dem