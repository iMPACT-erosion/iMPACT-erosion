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
    ax0.hist(non_missing_temp_data, bins=30, density=True, color='darkgreen', label='Temperature Data (no gaps)')
    ax0.plot(x_temp, pdf_fitted, color='green', lw=3, label=f'Fitted Normal: mean={mean:.2f}, std={std_dev:.2f}')
    ax0.set_title('Normal PDF Fitting')
    ax0.set_xlabel('Temperature (°C)')
    ax0.set_ylabel('Density')
    ax0.set_ylim([0, 0.08])
    ax0.legend()
    
    # Plot empirical CDF and fitted Normal CDF
    ax1 = fig.add_subplot(gs[0, 1])  # Top right plot
    ax1.plot(sorted_temp_data, empirical_cdf, marker='.', linestyle='none', color='darkgreen', label='Empirical CDF of Temperature Data')
    ax1.plot(sorted_temp_data, interpolated_cdf_fitted, color='green', lw=2, label=f'Fitted Normal CDF: mean={mean:.2f}, std={std_dev:.2f}')
    ax1.set_title(f'Normal CDF Fitting\nMSE: {mse_cdf:.4f}')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Cumulative Probability')
    ax1.legend()
    
    # Generate random values for missing temperature values based on the fitted Normal distribution
    missing_temp_indices = temp_data[temp_data['temp'].isna()].index
    missing_temp_values = norm.rvs(loc=mean, scale=std_dev, size=len(missing_temp_indices))

    # Fill in the missing temperature values in a copy of the raw data
    filled_temp_data = temp_data.copy()
    filled_temp_data.loc[missing_temp_indices, 'temp'] = missing_temp_values

    # Plot the original and filled temperature data
    ax2 = fig.add_subplot(gs[1, :])  # Bottom full-width plot
    ax2.plot(filled_temp_data, color='green', label='Filled Data')
    ax2.plot(temp_data, color='darkgreen', label='Observed Data')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_ylim([-10, 40])
    ax2.legend()
    
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
    ax0.set_title('Gamma PDF Fitting')
    ax0.set_xlabel('Rainfall (mm)')
    ax0.set_ylabel('Density')
    ax0.legend()
    
    # Plot empirical CDF and fitted Gamma CDF
    ax1 = fig.add_subplot(gs[0, 1])  # Top right plot
    ax1.plot(sorted_data, empirical_cdf, marker='.', linestyle='none', color='blue',label='Empirical CDF of Rainfall Data')
    ax1.plot(sorted_data, interpolated_cdf_fitted, lw=2, label=f'Fitted Gamma CDF: shape={alpha:.2f}, scale={beta:.2f}')
    ax1.set_title(f'Gamma CDF Fitting\nMSE: {mse_cdf:.4f}')
    ax1.set_xlabel('Rainfall (mm)')
    ax1.set_ylabel('Cumulative Probability')
    ax1.legend()
    
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
    ax2.set_ylabel('Rainfall (mm)')
    ax2.legend()
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.show()
