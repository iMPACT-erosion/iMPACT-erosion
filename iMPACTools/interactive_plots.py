#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np  # for numerical operations, especially with arrays
import matplotlib.pyplot as plt  # for data visualization
from scipy.stats import gamma, norm # for statistical distributions

def plot_normal_fitting(temp_data, mean, std_dev):

    # Remove NaN values to work with non-missing temperature data only
    non_missing_temp_data = temp_data['temp'].dropna()

    # Set up x-axis values for plotting PDFs and CDFs based on non-missing data range
    x_temp = np.linspace(non_missing_temp_data.min(), non_missing_temp_data.max(), 100)
    
    # Generate PDF and CDF values for a normal distribution with specified mean and std_dev
    pdf_fitted = norm.pdf(x_temp, loc=mean, scale=std_dev)
    cdf_fitted = norm.cdf(x_temp, loc=mean, scale=std_dev)
    
    # Calculate empirical CDF for the non-missing temperature data
    sorted_temp_data = np.sort(non_missing_temp_data)
    empirical_cdf = np.arange(1, len(sorted_temp_data) + 1) / len(sorted_temp_data)
    
    # Interpolate fitted CDF to the data points for MSE calculation
    interpolated_cdf_fitted = np.interp(sorted_temp_data, x_temp, cdf_fitted)
    mse_cdf = np.square(np.subtract(empirical_cdf, interpolated_cdf_fitted)).mean() 
    
    # Create a figure with a gridspec layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot histogram and fitted Normal PDF
    ax0 = fig.add_subplot(gs[0, 0])  # Top left plot
    ax0.hist(non_missing_temp_data, bins=30, density=True, color='red', label='Temperature Data (no gaps)')
    ax0.plot(x_temp, pdf_fitted, color='salmon', lw=3, label=f'Fitted Normal: mean={mean:.2f}, std={std_dev:.2f}')
    ax0.set_title('Normal PDF Fitting', fontsize=20)
    ax0.set_xlabel('Temperature (°C)', fontsize=20)
    ax0.set_ylabel('Density', fontsize=20)
    ax0.set_ylim([0, 0.08])
    ax0.legend(fontsize=15)
    
    # Plot empirical CDF and fitted Normal CDF
    ax1 = fig.add_subplot(gs[0, 1])  # Top right plot
    ax1.plot(sorted_temp_data, empirical_cdf, marker='.', linestyle='none', color='red', label='Empirical CDF')
    ax1.plot(sorted_temp_data, interpolated_cdf_fitted, color='salmon', lw=2, label=f'Fitted Normal CDF: mean={mean:.2f}, std={std_dev:.2f}')
    ax1.set_title(f'Normal CDF Fitting - MSE: {mse_cdf:.4f}', fontsize=20)
    ax1.set_xlabel('Temperature (°C)', fontsize=20)
    ax1.set_ylabel('Cumulative Probability', fontsize=20)
    ax1.legend(fontsize=15)
    
    # Generate random values for missing temperature values based on the fitted Normal distribution
    missing_temp_indices = temp_data[temp_data['temp'].isna()].index
    missing_temp_values = norm.rvs(loc=mean, scale=std_dev, size=len(missing_temp_indices))

    # Fill in the missing temperature values in a copy of the raw data
    filled_temp_data = temp_data.copy()
    filled_temp_data.loc[missing_temp_indices, 'temp'] = missing_temp_values

    # Plot the original and filled temperature data
    ax2 = fig.add_subplot(gs[1, :])  # Bottom full-width plot
    ax2.plot(filled_temp_data, color='salmon', label='Filled Data')
    ax2.plot(temp_data, color='red', label='Observed Data')
    ax2.set_ylabel('Temperature (°C)', fontsize=20)
    ax2.set_ylim([-10, 40])
    ax2.legend(fontsize=15)
    
    plt.tight_layout()
    plt.show()

    
def plot_gamma_fitting(rain_data,alpha, beta):
    
    # Fit the Normal distribution to the non-missing data
    non_missing_rain_data = rain_data['rain'].dropna()  # Remove NaN values

    # Visualize the data and the fitted distribution
    x_rain = np.linspace(0, non_missing_rain_data.max(), 100)

    # Generate x values for the Gamma PDF and CDF
    pdf_fitted = gamma.pdf(x_rain, alpha, loc=0, scale=beta)
    cdf_fitted = gamma.cdf(x_rain, alpha, loc=0, scale=beta)
    
    # Calculate empirical CDF
    sorted_data = np.sort(rain_data['rain'].dropna())
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Interpolate fitted CDF to the data points for MSE calculation
    interpolated_cdf_fitted = np.interp(sorted_data, x_rain, cdf_fitted)
    mse_cdf = np.square(np.subtract(empirical_cdf, interpolated_cdf_fitted)).mean() 
    
    # Create a figure with a gridspec for the narrow top and wide bottom plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot histogram and fitted Gamma PDF
    ax0 = fig.add_subplot(gs[0, 0])  # Top left plot
    ax0.hist(rain_data['rain'], bins=100, density=True, color = 'blue', label='Rainfall Data (with filled gaps)')
    ax0.plot(x_rain, pdf_fitted, lw=3, label=f'Fitted Gamma: shape={alpha:.2f}, scale={beta:.2f}')
    ax0.set_title('Gamma PDF Fitting', fontsize=20)
    ax0.set_xlabel('Rainfall (mm)', fontsize=20)
    ax0.set_ylabel('Density', fontsize=20)
    ax0.legend(fontsize=15)
    
    # Plot empirical CDF and fitted Gamma CDF
    ax1 = fig.add_subplot(gs[0, 1])  # Top right plot
    ax1.plot(sorted_data, empirical_cdf, marker='.', linestyle='none', color='blue',label='Empirical CDF of Rainfall Data')
    ax1.plot(sorted_data, interpolated_cdf_fitted, lw=2, label=f'Fitted Gamma CDF: shape={alpha:.2f}, scale={beta:.2f}')
    ax1.set_title(f'Gamma CDF Fitting\nMSE: {mse_cdf:.4f}', fontsize=20)
    ax1.set_xlabel('Rainfall (mm)', fontsize=20)
    ax1.set_ylabel('Cumulative Probability', fontsize=20)
    ax1.legend(fontsize=15)
    
    # Generate random values for missing rainfall values based on the fitted Gamma distribution
    missing_rain_indices = rain_data[rain_data['rain'].isna()].index
    missing_rain_values = gamma.rvs(alpha, loc=0, scale=beta, size=len(missing_rain_indices))

    # Fill in the missing rainfall values
    filled_rain_data = rain_data.copy()
    filled_rain_data.loc[missing_rain_indices, 'rain'] = missing_rain_values

    # Plot the original and filled rain data
    # Merge top-right empty space with bottom plot
    ax2 = fig.add_subplot(gs[1, :])  # Bottom full-width plot
    ax2.plot(filled_rain_data, label='Filled Data')
    ax2.plot(rain_data, color='blue', label='Observed Data')
    ax2.set_ylabel('Rainfall (mm)', fontsize=20)
    ax2.legend(fontsize=15)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.show()

# Function to display the interactive plot
def plot_interactive_K(silt, clay, organic_matter, structure, permeability):
    
    K1, Ks, Kp, K = soil_erodibility(silt, clay, organic_matter, structure, permeability)
    
    soil_texture = soil_texture_calculator(silt,clay)
    
    # Create a figure with 4 bars (K1, Ks, Kp, Total K)
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_width = 0.4  # Set width for bars
    index = [0, 0.5, 1, 2]  # Positions for the bars on the x-axis
    
    ax.bar(index[0], K1, bar_width, color="skyblue", label="K1 (Texture)")
    ax.bar(index[1], Ks, bar_width, color="lightgreen", label="Ks (Structure)")
    ax.bar(index[2], Kp, bar_width, color="lightcoral", label="Kp (Permeability)")
    ax.bar(index[3], K, bar_width*2, color="lightblue", label="Total K")
    
    ax.set_xticks(index)
    ax.set_xticklabels(["K1", "Ks", "Kp", "Total K"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.01, 0.1)  # Set y-axis limit to avoid bars going too high
    ax.set_ylabel('(t ha h) / (ha MJ mm)')
    
    ax.set_title(f"Soil Erodibility Factor (K): {K:.3f} - Soil texture: {soil_texture}",fontweight='bold')
    ax.legend(loc="upper left")
    plt.show()

def plot_interactive_CP(land_use, canopy_cover, ground_residue, support_practice, slope):
    C = calculate_c_factor(land_use, canopy_cover, ground_residue)
    P = calculate_p_factor(support_practice, slope)

    fig, ax = plt.subplots(figsize=(8, 5))

    factors = ['C-Factor', 'P-Factor']
    values = [C, P]

    bars = ax.bar(factors, values, color=['olivedrab', 'sandybrown'], width=0.5)

    ax.set_ylabel('Factor Value (Dimensionless)')
    ax.set_title('Estimated C and P Factors', fontweight='bold')
    ax.set_ylim(0, 1.1) # P and C factors are between 0 and 1

    # Add text labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom')

    # Display input summary
    inputs_summary = (f"Inputs:\n"
                      f" C-Factor: Land Use='{land_use}', Canopy={canopy_cover:.0f}%, Residue={ground_residue:.0f}%\n"
                      f" P-Factor: Practice='{support_practice}', Slope={slope:.1f}%")
    fig.text(0.5, -0.05, inputs_summary, ha='center', va='bottom', fontsize=9, wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for fig.text
    plt.show()