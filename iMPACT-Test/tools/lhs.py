# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:48:24 2024

@author: Usuario
"""

import numpy as np
from numba import njit

def preprocess_data(X):
    
    # Reshape the list of variables into a matrix (N x k) where each row represents a sampling location (N locations) 
    # and each column represents a different variable or property (k variables).
    k = np.shape(X)[0]
    rows = np.shape(X)[1]
    cols = np.shape(X)[2]
    N = rows*cols
    X = np.dstack(X)
    X = np.reshape(X, [N,k])
    
    # Remove rows with NaN values
    X = X[~np.isnan(X).any(axis=1)]
    
    return X

def cLHS(X, n, max_iter=1000, initial_temp=1.0, cooling_rate=0.99, p=0.5):
    """
    Perform Conditioned Latin Hypercube Sampling (cLHS).
    
    Parameters:
    X: list
        List of variables or properties. All variables must be 2D arrays with the same size (rows x cols)
    n: int
        The number of samples to select.
    max_iter: int, optional
        The maximum number of iterations (default is 1000).
    initial_temp: float, optional
        The initial temperature for the annealing schedule (default is 1.0).
    cooling_rate: float, optional
        The rate at which the temperature decreases (default is 0.99).
    p: float, optional
        The probability of performing a random swap (default is 0.5).
    
    Returns:
    x: ndarray
        The sampled sites (n x k).
    """
    
    N,k = X.shape

    # Step 1: Divide the quantile distribution of X into n strata
    strata = np.array([np.percentile(X[:, j], np.linspace(0, 100, n + 1)) for j in range(k)]).T

    # Step 2: Initialize n random samples and the reservoir of unsampled points
    selected_indices = np.random.choice(N, n, replace=False) # Randomly select n samples
    x = X[selected_indices] # Selected samples
    reservoir_indices = np.setdiff1d(np.arange(N), selected_indices) # Indices of points not selected
    r = X[reservoir_indices] # Reservoir of unsampled points

    # Calculate initial Z matrix which counts how many samples fall into each stratum for each variable
    Z = np.zeros((n, k), dtype=int)  # Initialize Z matrix with zeros
    for i in range(n):  # Loop over samples
        for j in range(k):  # Loop over variables
            for q in range(n):  # Loop over strata
                if strata[q, j] <= x[i, j] < strata[q + 1, j]:  # Check if sample falls in stratum
                    Z[q, j] += 1  # Increment Z matrix
                    break  # Exit loop once stratum is found

    current_obj = objective_function(Z)  # Initial objective function value
    temp = initial_temp  # Starting temperature for simulated annealing

    for iteration in range(max_iter):  # Main loop for iterations
        # Step 3: Propose a new solution
        new_x = x.copy()  # Copy current samples
        if np.random.rand() < p:  # With probability p, perform random swap
            swap_idx = np.random.choice(n, 1)[0]  # Select a random sample index
            res_idx = np.random.choice(len(r), 1)[0]  # Select a random reservoir index
            new_x[swap_idx] = r[res_idx]  # Swap sample with reservoir point
        else:  # Otherwise, perform systematic replacement
            worst_fit_idx = np.argmax(np.sum((Z - 1) ** 2, axis=1))  # Find worst-fit sample index
            res_idx = np.random.choice(len(r), 1)[0]  # Select a random reservoir index
            new_x[worst_fit_idx] = r[res_idx]  # Replace worst-fit sample with reservoir point

        # Calculate new Z matrix for the proposed solution
        new_Z = np.zeros((n, k), dtype=int)  # Initialize new Z matrix with zeros
        for i in range(n):  # Loop over samples
            for j in range(k):  # Loop over variables
                for q in range(n):  # Loop over strata
                    if strata[q, j] <= new_x[i, j] < strata[q + 1, j]:  # Check if sample falls in stratum
                        new_Z[q, j] += 1  # Increment new Z matrix
                        break  # Exit loop once stratum is found

        new_obj = objective_function(new_Z)  # New objective function value
        delta_obj = new_obj - current_obj  # Change in the objective function

        # Step 4: Perform annealing schedule
        if delta_obj < 0 or np.random.rand() < np.exp(-delta_obj / temp):  # Accept new solution conditionally
            x = new_x  # Update samples
            Z = new_Z  # Update Z matrix
            current_obj = new_obj  # Update objective function value

        # Cool down the temperature
        temp *= cooling_rate  # Decrease temperature

        # Step 5: Check stopping criteria (optional)
        if current_obj <= 0.001:  # Stop if objective function is sufficiently low
            break  # Exit main loop

    return x  # Return final samples

### Conditioned latin hypercube with Numba ###

@njit
def compute_percentiles(data, percentiles):
    """Compute the percentiles of a 1D array."""
    data_sorted = np.sort(data)
    return np.interp(percentiles, np.linspace(0, 100, len(data_sorted)), data_sorted)

@njit
def objective_function(Z):
    """Objective function to measure the stratification.
    The goal is to minimize this function, where perfect stratification results in a value of 0."""
    return np.sum((Z - 1) ** 2)  # Sum of squared differences from 1

@njit
def cLHS_numba(X, n, max_iter=1000, initial_temp=1.0, cooling_rate=0.99, p=0.5):
    """
    Perform Conditioned Latin Hypercube Sampling (cLHS).
    
    Parameters:
    X: ndarray
        Array of variables or properties (N x k).
    n: int
        The number of samples to select.
    max_iter: int, optional
        The maximum number of iterations (default is 1000).
    initial_temp: float, optional
        The initial temperature for the annealing schedule (default is 1.0).
    cooling_rate: float, optional
        The rate at which the temperature decreases (default is 0.99).
    p: float, optional
        The probability of performing a random swap (default is 0.5).
    
    Returns:
    x: ndarray
        The sampled sites (n x k).
    """
    
    N, k = X.shape

    # Step 1: Divide the quantile distribution of X into n strata
    strata = np.empty((n + 1, k))
    percentiles = np.linspace(0, 100, n + 1)
    for j in range(k):
        strata[:, j] = compute_percentiles(X[:, j], percentiles)

    # Step 2: Initialize n random samples and the reservoir of unsampled points
    selected_indices = np.random.choice(N, n, replace=False)  # Randomly select n samples
    x = X[selected_indices]  # Selected samples
    reservoir_indices = np.array([i for i in range(N) if i not in selected_indices])  # Indices of points not selected
    r = X[reservoir_indices]  # Reservoir of unsampled points

    # Calculate initial Z matrix which counts how many samples fall into each stratum for each variable
    Z = np.zeros((n, k), dtype=np.int32)  # Initialize Z matrix with zeros
    for i in range(n):  # Loop over samples
        for j in range(k):  # Loop over variables
            for q in range(n):  # Loop over strata
                if strata[q, j] <= x[i, j] < strata[q + 1, j]:  # Check if sample falls in stratum
                    Z[q, j] += 1  # Increment Z matrix
                    break  # Exit loop once stratum is found

    current_obj = objective_function(Z)  # Initial objective function value
    temp = initial_temp  # Starting temperature for simulated annealing

    for iteration in range(max_iter):  # Main loop for iterations
        # Step 3: Propose a new solution
        new_x = x.copy()  # Copy current samples
        if np.random.rand() < p:  # With probability p, perform random swap
            swap_idx = np.random.randint(n)  # Select a random sample index
            res_idx = np.random.randint(len(r))  # Select a random reservoir index
            new_x[swap_idx] = r[res_idx]  # Swap sample with reservoir point
        else:  # Otherwise, perform systematic replacement
            worst_fit_idx = np.argmax(np.sum((Z - 1) ** 2, axis=1))  # Find worst-fit sample index
            res_idx = np.random.randint(len(r))  # Select a random reservoir index
            new_x[worst_fit_idx] = r[res_idx]  # Replace worst-fit sample with reservoir point

        # Calculate new Z matrix for the proposed solution
        new_Z = np.zeros((n, k), dtype=np.int32)  # Initialize new Z matrix with zeros
        for i in range(n):  # Loop over samples
            for j in range(k):  # Loop over variables
                for q in range(n):  # Loop over strata
                    if strata[q, j] <= new_x[i, j] < strata[q + 1, j]:  # Check if sample falls in stratum
                        new_Z[q, j] += 1  # Increment new Z matrix
                        break  # Exit loop once stratum is found

        new_obj = objective_function(new_Z)  # New objective function value
        delta_obj = new_obj - current_obj  # Change in the objective function

        # Step 4: Perform annealing schedule
        if delta_obj < 0 or np.random.rand() < np.exp(-delta_obj / temp):  # Accept new solution conditionally
            x = new_x  # Update samples
            Z = new_Z  # Update Z matrix
            current_obj = new_obj  # Update objective function value

        # Cool down the temperature
        temp *= cooling_rate  # Decrease temperature

        # Step 5: Check stopping criteria (optional)
        if current_obj <= 0.001:  # Stop if objective function is sufficiently low
            break  # Exit main loop

    return x  # Return final samples