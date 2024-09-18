"""
@author: Andres Peñuela
contact: apenuela@uco.es
"""

import numpy as np

def slope_gradient(dem, dem_resol=5):
    """Calculate the slope gradient using the elevation data."""
    # Use numpy's gradient function to compute the slope in x and y directions
    dx, dy = np.gradient(dem, dem_resol)
    # Calculate the magnitude of the gradient
    slope = np.sqrt(dx**2 + dy**2)
    return slope

def profile_curvature(dem, dem_resol=5):
    """Calculate profile curvature from a DEM. Profile curvature is calculated based 
    on the curvature of the land surface in the direction of the steepest slope

    - Positive Profile Curvature: Indicates concave slopes along the steepest descent, 
    where water and soil are likely to accumulate, leading to deposition.
    - Negative Profile Curvature: Indicates convex slopes along the steepest descent, 
    where soil is more likely to erode.
    """
    
    # Calculate the first partial derivatives (using central differences)
    dz_dx = np.gradient(dem, dem_resol, axis=1)
    dz_dy = np.gradient(dem, dem_resol, axis=0)
    
    # Calculate the second partial derivatives
    d2z_dx2 = np.gradient(dz_dx, dem_resol, axis=1)
    d2z_dy2 = np.gradient(dz_dy, dem_resol, axis=0)
    d2z_dxdy = np.gradient(dz_dx, dem_resol, axis=0)
    
    # Calculate the profile curvature
    numerator = d2z_dx2 * dz_dx**2 + d2z_dy2 * dz_dy**2 + 2 * dz_dx * dz_dy * d2z_dxdy
    denominator = (dz_dx**2 + dz_dy**2) * np.sqrt(dz_dx**2 + dz_dy**2)
    profile_curvature = numerator / denominator
    
    return profile_curvature

def plan_curvature(dem, dem_resol=5):
    """
    Calculate plan curvature from a DEM. 
    Plan curvature can be calculated from a Digital Elevation Model (DEM) 
    using the derivatives of the elevation data. The calculation involves
    determining the curvature of the surface in the horizontal plane 
    (perpendicular to the slope). Here's a step-by-step approach to calculate 
    plan curvature:
    - Positive Plan Curvature: Indicates converging terrain (e.g., valleys), 
      where materials tend to accumulate.
    - Negative Plan Curvature: Indicates diverging terrain (e.g., ridges), 
      where materials tend to spread out, leading to potential erosion
    """
    
    # Calculate the first partial derivatives (using central differences)
    dz_dx = np.gradient(dem, dem_resol, axis=1)
    dz_dy = np.gradient(dem, dem_resol, axis=0)
    
    # Calculate the second partial derivatives
    d2z_dx2 = np.gradient(dz_dx, dem_resol, axis=1)
    d2z_dy2 = np.gradient(dz_dy, dem_resol, axis=0)
    d2z_dxdy = np.gradient(dz_dx, dem_resol, axis=0)
    
    # Calculate the plan curvature
    numerator = d2z_dx2 * dz_dy**2 + d2z_dy2 * dz_dx**2 - 2 * dz_dx * dz_dy * d2z_dxdy
    denominator = (dz_dx**2 + dz_dy**2)**1.5
    plan_curvature = numerator / denominator
    
    return plan_curvature