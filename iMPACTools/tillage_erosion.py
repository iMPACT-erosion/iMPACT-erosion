"""
@author: Andres Peñuela
contact: apenuela@uco.es
"""

def tillage_erosion(dem, slope, curvature, till_depth, BD, a, b, passes, dem_resol=5):
    """Simulates tillage erosion based on the Van Oost et al. (2000) method.
    
    Parameters:
    dem (array-like): Digital Elevation Model (DEM) representing the terrain.
    slope (array-like): Precomputed slope gradient derived from the DEM.
    curvature (array-like): Precomputed curvature (plan or profile) derived from the DEM.
    till_depth (float): Depth of tillage in meters.
    BD (float): Bulk density of the soil in tn/m³.
    a (float): Empirical coefficient related to slope influence on soil translocation.
    b (float): Empirical coefficient representing the baseline translocation rate.
    passes (int): Number of tillage passes, affecting the magnitude of soil movement.
    dem_resol (float, optional): Resolution of the DEM in meters. Default is 5 meters.
    
    Returns:
    SL_till (array-like): Soil erosion/deposition due to tillage in tn/ha/year.
    """

    # Compute the coefficient B, which represents the combined influence of slope and
    # the empirical coefficients 'a' and 'b' on tillage translocation.
    B = a * slope + b

    # Calculate the tillage translocation coefficient (ktill).
    # This value represents the potential for soil movement due to tillage, influenced by
    # the tillage depth, soil bulk density, and the slope's impact (B).
    ktill = - till_depth * BD * B
    
    # Calculate the soil erosion or depostion due to tillage translocation.
    # This is determined by multiplying the translocation coefficient (ktill) by the curvature,
    # the number of tillage passes, and a scaling factor to convert to tn/ha/year.
    # The resulting SL_till represents the soil erosion (positive) or deposition (negative)
    # resulting from the tillage process.
    SL_till = ktill * curvature * passes * 100  # Soil erosion/deposition in tn/ha/year
    
    return SL_till



