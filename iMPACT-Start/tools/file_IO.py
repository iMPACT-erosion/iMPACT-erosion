# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:26:02 2024

@author: andro
"""

import os
from ipywidgets import widgets
import rasterio  

def select_input_file(path):
    # List all files in the 'inputs/' directory and storing them in a list
    files = [f for f in os.listdir(path)]
    
    # Create a Select menu with the list of files as options
    input_file = widgets.Select(
        options=files,  # Set the options of the Select widget to the list of files
        description='select raster:',  # Set the description label of the Select widget
        disabled=False  # Set the initial state of the widget to enabled
    )
    
    # Display the Select widget
    display(input_file)
    
    # Return the Select widget for further use
    return input_file


def open_raster(path,input_file):
    # Open the raster map file
    with rasterio.open(path+input_file) as src:
        # Read the raster data as a numpy array
        raster = src.read(1)  # Read the first band (index 0)

        # Get metadata of the raster map: the `metadata` variable contains 
        # metadata information such as the raster's spatial reference system, 
        # data type, and geotransform. You can use this metadata for various 
        # purposes, such as georeferencing the raster map.
        raster_metadata = src.meta
    return raster, raster_metadata

def save_as_raster(path,name,data,metadata):
    # Specify the output file path and name
    output_file = path+name

    # Write the modified raster data to a new file with the same metadata
    with rasterio.open(output_file, 'w', **metadata) as dst:
        # Write the modified raster data to the new file
        dst.write(data, 1) 