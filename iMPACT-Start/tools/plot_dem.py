# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:55:58 2024

@author: andro
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot3d_dem(dem, data=None, title=None):
    """
    Plot the synthetic digital elevation model (DEM) in 3-D.

    Parameters:
    - dem: 2D array representing the digital elevation model.
    - data: Optional 2D array representing additional data to be plotted.
    - title: Optional string for the title of the second subplot (2D plot).
    """
    # Set elevation and azimuth angles for 3D plot (if needed, adjust these values to improve the visualization)
    elev = 30
    azim = 240

    # Create meshgrid for x and y coordinates
    x, y = np.meshgrid(np.arange(dem.shape[1]), np.arange(dem.shape[0]))
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Digital Elevation Model & '+title if title else 'Digital Elevation Model')
    
    # First subplot: 3D plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    
    if data is not None:
        color = cm.coolwarm((np.flipud(data) - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data)))
    else:
        color = cm.cividis((np.flipud(dem) - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem)))
    
    ax.plot_surface(x, y, np.flipud(dem),
                    rstride=1,
                    cstride=1,
                    cmap='cividis',
                    facecolors=color,
                    linewidth=0.,
                    antialiased=True)
    
    # Set view angle for 3D plot
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Elevation')
    
    # Second subplot (2D plot)
    ax = fig.add_subplot(1, 2, 2)
    if data is not None:
        mesh = ax.imshow(data, cmap='coolwarm')
        ax.set_title(title if title else 'Data')
    else:
        mesh = ax.imshow(dem, cmap='cividis')
        ax.set_title('Elevation')
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    fig.colorbar(mesh) 
    plt.show()

# Example usage (you need to replace this with actual data)
# fd_D8, fa_D8, ix_D8, ixc_D8 = flow_accumulation_D8(dem)
# plot3d_dem(dem, fa_D8, 'Flow Accumulation')
