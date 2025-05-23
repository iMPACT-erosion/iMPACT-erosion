"""
@author: Andres Peñuela
contact: apenuela@uco.es
"""

import numpy as np
from scipy.ndimage import sobel

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


def ls_factor(slope, flow_acc, cell_size):
    """
    Calculates the L (slope length), S (slope steepness), and LS (combined) factors
    from slope and flow accumulation rasters.

    Parameters:
    -----------
    slope : 2D array-like
        Slope raster values, expressed as rise/run (not degrees).
    flow_acc : 2D array-like
        Flow accumulation (number of upslope contributing cells).

    Returns:
    --------
    L : 2D array-like
        Slope length factor
    S : 2D array-like
        Slope steepness factor
    LS : 2D array-like
        Combined LS factor
    """

    # --- S Factor (Slope Steepness) ---
    # Convert slope to percentage
    slope_percent = slope * 100

    # Convert slope to radians for trigonometric calculations
    slope_radians = np.arctan(slope)  # slope = rise/run, arctan gives angle in radians

    # Apply Foster et al. (1981) equations for S factor
    S = np.where(
        slope_percent < 9,
        10.8 * np.sin(slope_radians) + 0.03,
        16.8 * np.sin(slope_radians) - 0.50)

    # --- L Factor (Slope Length) ---
    # Calculate slope-dependent coefficient beta
    beta = (np.sin(slope_radians) / 0.0896) / (0.56 + 3 * (np.sin(slope_radians))**0.8)

    # Calculate m exponent depending on slope steepness
    m = beta / (beta + 1)

    # Calculate L factor from flow accumulation
    # 22.13 m is the standard RUSLE unit plot length
    L = (flow_acc*cell_size**2 / 22.13) ** m

    # --- LS Factor ---
    LS = L * S  # Multiply L and S factors

    return L, S, LS


def calculate_aspect(dem, cell_size):
    """Calculate aspect (radians) from DEM."""
    dzdx = sobel(dem, axis=1) / (8 * cell_size)
    dzdy = sobel(dem, axis=0) / (8 * cell_size)
    
    aspect = np.arctan2(dzdy, -dzdx)
    
    return aspect

def ls_factor2(slope, aspect, flow_acc, cell_size):
    """
    Compute the LS-factor for soil erosion modeling using the Desmet and Govers (1996) method.

    Parameters:
    -----------
    slope : 2D array-like
        Slope values expressed as rise/run (not degrees or radians).
        Example: np.tan(slope_angle_in_degrees)

    aspect : 2D array-like
        Aspect in radians (0 = east, pi/2 = north, etc.), from DEM.

    flow_acc : 2D array-like
        Flow accumulation (number of upslope contributing cells).

    cell_size : float
        Size of the DEM raster cell in meters.

    Returns:
    --------
    L : 2D array-like
        Slope length factor (dimensionless)

    S : 2D array-like
        Slope steepness factor (dimensionless)

    LS : 2D array-like
        Combined LS factor (dimensionless)
    """

    # Convert slope to radians for use in trigonometric formulas
    slope_rad = np.arctan(slope)

    # --- S Factor (Slope Steepness) ---
    # RUSLE standard formulas based on slope in radians (Foster et al., 1981 / Renard et al., 1997)
    slope_percent = slope * 100
    S = np.where(
        slope_percent < 9,
        10.8 * np.sin(slope_rad) + 0.03,
        16.8 * np.sin(slope_rad) - 0.50
    )

    # --- m Exponent (Slope-dependent) ---
    # m controls the influence of slope on the L factor; depends on slope angle
    beta = (np.sin(slope_rad) / 0.0896) / (0.56 + 3 * (np.sin(slope_rad) ** 0.8))
    m = beta / (1 + beta)

    # --- L Factor (Slope Length) ---
    # A is upslope contributing area (m²)
    A = flow_acc * (cell_size ** 2)

    # Xi,j from Desmet & Govers: direction correction using aspect
    X = np.abs(np.sin(aspect) + np.cos(aspect))

    # Avoid divide-by-zero issues
    X[X == 0] = 1e-6
    m = np.clip(m, 0, 1)  # Limit m to [0, 1] for stability

    D = cell_size
    L = (((A + D**2)**(m + 1) - A**(m + 1)) / ((D**(m + 2)) * X)) * (1 / (22.13**m))

    # --- Mask for slopes steeper than 50% (≈ 26.6°)
    LS = L * S
    LS[slope > 0.5] = 0  # 0.5 = 50% slope = tan(26.6°)

    return L, S, LS
