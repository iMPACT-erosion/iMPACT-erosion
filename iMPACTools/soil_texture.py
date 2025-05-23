def soil_texture_calculator(silt, clay):
    """
    Classifies soil texture based on USDA soil texture triangle.
    
    Parameters:
    -----------
    
    silt : float
        Sand content (%)    
    clay : float
        Clay content (%)
    
    Returns:
    --------
        str: Soil texture classification.
    """

    sand = 100 - silt - clay
    
    if not (0 <= silt <= 100 and 0 <= clay <= 100):
        raise ValueError("Silt and clay percentages must be between 0 and 100.")
    
    if silt < 0:
        raise ValueError("The sum of sand and clay percentages cannot exceed 100.")
    
    # Classify based on USDA soil texture triangle
    if clay >= 40:
        if sand > 45:
            return "sandy clay"
        elif silt >= 40:
            return "silty clay"
        else:
            return "clay"
    
    elif clay >= 27 and clay < 40:
        if sand < 20:
            return "silty clay loam"
        elif sand >= 20 and sand < 40:
            return "clay loam"
        elif sand >=40 and clay >= 36:
            return "sandy clay"
        else:
            return "sandy clay loam"
        
    elif clay >= 20 and clay < 27:
        if silt < 28:
            return "sandy clay loam"
        elif silt >= 28 and silt < 50:
            return "loam"
        else:
            return "silty loam"
    
    if clay < 20:
        if sand >= 85:
            return "sand"
        elif sand < 85 and sand >= 50:
            return "sandy loam"
        elif sand < 50:
            if clay >= 7 and silt < 50:
                return "loam"
            elif clay < 7 and silt < 50:
                return "sandy loam"
            elif clay < 14 and silt >= 80:
                return "silt"
            else:
                return "silty loam"


# Permeability codes by texture class (USDA)
permeability_codes = {
    'clay': 6, 'silty clay': 6, 'sandy clay': 5,
    'clay loam': 4, 'silty clay loam': 5, 'sandy clay loam': 4,
    'loam': 3, 'silty loam': 3, 'sandy loam': 2,
    'silt': 3, 'loamy sand': 2, 'sand': 1
}

def permeability_code_calculator(silt, clay):
    """
    Estimate soil permeability code (1-6)

    Parameters:
    -----------

    silt : float
        Sand content (%)    
    clay : float
        Clay content (%)

    Returns:
    --------
    int : Permeability code (1=rapid, 6=very slow)
    """

    sand = 100 - silt - clay
    
    texture_class = soil_texture_calculator(silt, clay)

    return permeability_codes[texture_class]
    

def structure_code_calculator(silt, clay, organic_matter=None):
    """
    Estimate soil structure code (1-4)

    Parameters:
    -----------
    silt : float
        Sand content (%)    
    clay : float
        Clay content (%)
    organic_matter : float, optional
        Organic matter content (%)

    Returns:
    --------
    int : Structure code (1=very fine granular, 4=blocky, platy or massive)
    """
    # This is a simplified approach - actual structure depends on multiple factors
    # and field observation is preferred
    
    # Default structure code (moderate, medium granular)
    structure_code = 3

    sand = 100 - silt - clay
    # Adjust based on clay content if available
    if clay > 27:  # High clay
        structure_code = 4  # More likely blocky or massive
    elif clay < 20:  # Low clay
        structure_code = 2  # More likely fine granular
        if sand > 80:
            structure_code = 1

    # Adjust for organic matter if available
    if organic_matter is not None:
        if organic_matter > 3:  # High organic matter
            structure_code = max(1, structure_code - 1)  # Improves structure

    return structure_code
