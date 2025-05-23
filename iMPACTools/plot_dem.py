# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:55:58 2024

@author: andro
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go

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

def plot3d_dem_plotly(dem, data=None, title=None):
    """
    Plot a DEM with optional overlay data in an interactive 3D surface plot using Plotly.

    Parameters:
    - dem: 2D numpy array for elevation.
    - data: Optional 2D numpy array for coloring (same shape as dem).
    - title: Optional string for plot title.
    """
    z = np.flipud(dem)  # Flip to match top-down visual convention

    # Create color data
    if data is not None:
        c = np.flipud(data)
    else:
        c = z

    fig = go.Figure(data=[
        go.Surface(
            z=z,
            surfacecolor=c,
            colorscale='Cividis' if data is None else 'RdBu',
            colorbar=dict(title='Value'),
        )
    ])

    fig.update_layout(
        title=title or "Digital Elevation Model",
        scene=dict(
            xaxis_title='X axis',
            yaxis_title='Y axis',
            zaxis_title='Elevation',
        ),
        width=900,
        height=600
    )

    fig.show()
