# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:51:06 2024

@author: Andres Peñuela
"""
import numpy as np
from numba import njit, prange

@njit
def direction_lowest_neighbour(x, y, dem):
    """
    Determine the direction to the lowest neighboring cell.

    Args:
        x, y (int): Coordinates of the cell.
        dem (numpy.ndarray): A 2D array representing the elevation data.

    Returns:
        dx, dy, flow_dir (tuple): The direction to the lowest neighbour (dx, dy) and the corresponding flow direction.
    """
    
    rows, cols = dem.shape  # Get the dimensions of the DEM (rows, cols)
    
    # Define the neighbourhood directions (8-connected)
    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),         (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    
    # Flow direction values corresponding to each neighbour
    neighbours_dir = [32, 64, 128,
                      16,      1,
                       8,  4,   2]
    
    # Initialize an array to store elevations of neighbouring cells
    neighbour_elevations = np.full(8, np.inf)

    for idx in range(8):
        dx, dy = neighbours[idx]
        nx, ny = x + dx, y + dy
        # Check if neighbour is within bounds
        if 0 <= nx < rows and 0 <= ny < cols and not np.isnan(dem[nx, ny]):
            neighbour_elevations[idx] = dem[nx, ny]

    # Find the index of the minimum elevation
    min_idx = np.argmin(neighbour_elevations)
    dx, dy = neighbours[min_idx]
    flow_dir = neighbours_dir[min_idx]

    return dx, dy, flow_dir

@njit
def nanargmax(arr):
    """
    Find the index of the maximum value in an array, ignoring NaNs.

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        int: Index of the maximum value.
    """
    max_val = -np.inf  # Initialize max_val to negative infinity to ensure any value will be higher
    max_idx = -1       # Initialize max_idx to -1 to indicate no index found yet

    # Iterate through all elements in the flattened array
    for idx in range(arr.size):
        # Check if the current element is not NaN and greater than max_val
        if not np.isnan(arr.flat[idx]) and arr.flat[idx] > max_val:
            max_val = arr.flat[idx]  # Update max_val with the new maximum value
            max_idx = idx            # Update max_idx with the index of the new maximum value

    return max_idx  # Return the index of the maximum value

@njit
def flow_accumulation_D8(dem, dem_resol=5, num_iterations=None):
    """
    Calculate the flow accumulation using the D8 method for a digital elevation model (DEM).

    Args:
        dem (numpy.ndarray): A 2D array representing the elevation data.
        dem_resol (float, optional): The resolution of the DEM grid, representing the physical distance between adjacent cells. Default is 5.
        num_iterations (int, optional): Maximum number of iterations to perform. If None, process all non-NaN cells.

    Returns:
        slope (numpy.ndarray): The slope gradient for each cell.
        flow_dir (numpy.ndarray): Flow direction values for each cell, encoded as angles or direction indices.
        flow_acc (numpy.ndarray): Flow accumulation values for each cell, representing the number of upstream cells contributing to the flow.
        flow_rout_up_row (numpy.ndarray): Row indices of upstream cells for each processed cell.
        flow_rout_up_col (numpy.ndarray): Column indices of upstream cells for each processed cell.
        flow_rout_down_row (numpy.ndarray): Row indices of downstream cells for each processed cell.
        flow_rout_down_col (numpy.ndarray): Column indices of downstream cells for each processed cell.
        flow_rout_contrib (numpy.ndarray): Contribution count from each cell to its downstream neighbour (initially set to 1 for each cell).
        flow_rout_slope (list): Slope gradients for each flow-routing path between upstream and downstream cells.
    """
    # Create a temporary copy of the DEM to work with
    dem_temp = dem.copy()
    
    # Initialize the flow accumulation array; start with a value of 1 for each cell
    flow_acc = dem.copy() / dem.copy()
    
    # Initialize the flow direction array; initially all zeros
    flow_dir = np.zeros_like(dem, dtype=np.float64)

    # Initialize the slope array; initially all zeros
    slope = np.zeros_like(dem, dtype=np.float64)

    # Calculate the total number of non-NaN cells in the DEM
    num_nonan = np.sum(~np.isnan(dem))

    # If a specific number of iterations is set, limit the number of cells to process
    if num_iterations is not None:
        num_nonan = min(num_iterations, np.sum(~np.isnan(dem)))
        
    # Initialize arrays to store upstream and downstream cell indices for each processed cell
    flow_rout_up_row = np.zeros(num_nonan, dtype=np.int64)
    flow_rout_up_col = np.zeros(num_nonan, dtype=np.int64)
    flow_rout_down_row = np.zeros(num_nonan, dtype=np.int64)
    flow_rout_down_col = np.zeros(num_nonan, dtype=np.int64)
    
    # Initialize an array to track the contribution count from each cell to its downstream neighbour
    flow_rout_contrib = np.zeros(num_nonan, dtype=np.int64)
    
    # Initialize a list to store the slope gradients for each flow-routing path
    flow_rout_slope = []

    for i in range(num_nonan - 1):
        # Find the index of the highest remaining cell in the temporary DEM
        high_cell_index = nanargmax(dem_temp)
        
        # Convert the linear index to a row, column pair
        high_cell = high_cell_index // dem_temp.shape[1], high_cell_index % dem_temp.shape[1]
        
        # Calculate the direction to the lowest neighbouring cell using the D8 method
        dx, dy, flow_dir[high_cell] = direction_lowest_neighbour(high_cell[0], high_cell[1], dem)

        # Add the flow accumulation of the current cell to its lowest neighbouring cell
        flow_acc[high_cell[0] + dx, high_cell[1] + dy] += flow_acc[high_cell]

        # Calculate the slope gradient between the current cell and its lowest neighbouring cell
        dz = dem[high_cell] - dem[high_cell[0] + dx, high_cell[1] + dy]
        slope[high_cell] = dz / np.sqrt((dx * dem_resol)**2 + (dy * dem_resol)**2)  # Slope = height change / distance

        # Store the row and column indices of the current (upstream) cell
        flow_rout_up_row[i] = high_cell[0]
        flow_rout_up_col[i] = high_cell[1]
        
        # Store the row and column indices of the lowest (downstream) neighbouring cell
        flow_rout_down_row[i] = high_cell[0] + dx
        flow_rout_down_col[i] = high_cell[1] + dy
        
        # Set the contribution count to 1 (indicating one cell contributing to downstream flow)
        flow_rout_contrib[i] = 1
        
        # Append the calculated slope gradient for this flow path
        flow_rout_slope.append(slope[high_cell])

        # Mark the processed cell as inactive by setting it to NaN in the temporary DEM
        dem_temp[high_cell[0], high_cell[1]] = np.nan

    return slope, flow_dir, flow_acc, flow_rout_up_row, flow_rout_up_col, flow_rout_down_row, flow_rout_down_col, flow_rout_contrib, flow_rout_slope

def flow_contribution_neighbours(x, y, dem, dem_resol=5):
    """
    Helper function to get the flow contribution to neighbouring cells based on slope.

    Args:
    - x, y: Coordinates of the cell
    - dem: The digital elevation model (DEM)

    Returns:
    - flow_contributions: Array containing the flow contribution to neighbouring cells
    """

    rows, cols = dem.shape  # Get the dimensions of the DEM (rows, cols)

    # Define the neighbourhood directions (8-connected)
    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    # Initialize an array to store slopes in all eight directions
    slopes = np.zeros(8)

    # Calculate slopes in all eight directions
    for j, (dx, dy) in enumerate(neighbours):
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:  # Check if the neighbouring cell is within bounds
            # Calculate the difference in elevation (height difference)
            dz = max(dem[x, y] - dem[nx, ny], 0)  # Take the maximum to account for negative slopes
            # Calculate the slope in this direction
            slopes[j] = dz / np.sqrt((dx*dem_resol) ** 2 + (dy*dem_resol) ** 2)  # Slope = change in height / distance
        else:
            slopes[j] = np.nan  # If the neighbouring cell is out of bounds, assign NaN to its slope

    # Compute the flow contributions based on the slopes
    flow_contributions = slopes / np.nansum(slopes)  # Normalize slopes to get flow contributions

    return slopes, flow_contributions
    
def flow_accumulation_Dinf(dem, dem_resol=5, *num_iterations):
    """
    Calculates the flow accumulation for a digital elevation model (DEM) using the D-infinity method.

    Args:
        dem (numpy.ndarray): A 2D array representing the elevation data.
        dem_resol (float, optional): The resolution of the DEM grid, representing the physical distance between adjacent cells. Default is 5.
        num_iterations (int, optional): Maximum number of iterations to perform. If provided, limits the number of cells to process.

    Returns:
        flow_acc (numpy.ndarray): The flow accumulation values for each cell, representing the sum of all upstream contributions.
        flow_rout_up_row (list): List of row indices for upstream cells corresponding to each flow path.
        flow_rout_up_col (list): List of column indices for upstream cells corresponding to each flow path.
        flow_rout_down_row (list): List of row indices for downstream cells corresponding to each flow path.
        flow_rout_down_col (list): List of column indices for downstream cells corresponding to each flow path.
        flow_rout_contrib (list): List of flow contribution values from upstream to downstream cells.
        flow_rout_slope (list): List of slope values corresponding to each flow path.
    """

    # Create a temporary copy of the DEM to manipulate during processing
    dem_temp = dem.copy()
    
    # Initialize the flow accumulation matrix with ones, setting NaN values for non-DEM cells
    flow_acc = np.ones_like(dem, dtype=float)
    flow_acc[np.isnan(dem)] = np.nan
    
    # Determine the number of non-NaN cells to process
    if num_iterations:
        num_nonan = np.min([num_iterations[0], np.sum(~np.isnan(dem))])
    else:
        # Calculate the total number of non-NaN cells
        num_nonan = np.sum(~np.isnan(dem))

    # Initialize lists to store flow routing information: upstream/downstream indices and contributions
    flow_rout_up_row = []
    flow_rout_up_col = []
    flow_rout_down_row = []
    flow_rout_down_col = []
    flow_rout_contrib = []
    flow_rout_slope = []
    
    # Process the DEM by iterating through all non-NaN cells
    for i in range(num_nonan - 1):

        # Identify the highest cell remaining in the temporary DEM
        high_cell_index = np.nanargmax(dem_temp)
        high_cell = np.unravel_index(high_cell_index, dem_temp.shape)
        
        # Calculate the slope and flow contributions from the highest cell to its neighbours
        slopes, flow_contributions = flow_contribution_neighbours(high_cell[0], high_cell[1], dem)
        
        # Define the neighbourhood directions (D8 method - 8-connected neighbours)
        neighbours = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        
        # Update flow accumulation for each neighbouring cell
        for j, (dx, dy) in enumerate(neighbours):
            nx = high_cell[0] + dx
            ny = high_cell[1] + dy
            
            # Ensure the neighbour is within bounds, not NaN, and has a positive contribution
            if nx < dem.shape[0] and ny < dem.shape[1] and ~np.isnan(flow_contributions[j]) and flow_contributions[j] > 0:
                # Add the flow contribution from the current cell to its neighbour
                flow_acc[nx, ny] += flow_acc[high_cell] * flow_contributions[j]
                
                # Record the indices of the current (upstream) cell and the neighbour (downstream cell)
                flow_rout_up_row.append(high_cell[0])
                flow_rout_up_col.append(high_cell[1])
                flow_rout_down_row.append(nx)
                flow_rout_down_col.append(ny)
                
                # Store the flow contribution and slope for this path
                flow_rout_contrib.append(flow_contributions[j])
                flow_rout_slope.append(slopes[j])

        # Mark the processed cell as inactive by setting it to NaN in the temporary DEM
        dem_temp[high_cell[0], high_cell[1]] = np.nan

    # Optional: Uncomment the following line to apply a threshold and remove pixels contributing to gully formation
    # flow_acc[flow_acc > 1000] = np.nan

    return flow_acc, flow_rout_up_row, flow_rout_up_col, flow_rout_down_row, flow_rout_down_col, flow_rout_contrib, flow_rout_slope