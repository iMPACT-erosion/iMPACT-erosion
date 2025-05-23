import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize

def idw(points, values, xi, power=2):
    """
    Classic power-based Inverse Distance Weighting (IDW) interpolation over a 2D grid.

    Parameters:
        points : (n, 2) array
            Known data point coordinates.
        values : (n,) array
            Values at the data points.
        xi : tuple of 2D arrays (grid_x, grid_y)
            Coordinates where interpolation is to be computed.
        power : float
            Power parameter for inverse distance weighting (default = 2).

    Returns:
        zi : 2D array
            Interpolated grid of values.
    """
    grid_x, grid_y = xi
    xi_flat = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    dists = np.linalg.norm(xi_flat[:, None, :] - points[None, :, :], axis=2)

    # Avoid division by zero by setting minimum distance
    dists[dists == 0] = 1e-10

    # Compute inverse distance weights
    weights = 1.0 / dists**power
    weights_sum = np.sum(weights, axis=1)

    # Compute interpolated values
    zi_flat = np.sum(weights * values[None, :], axis=1) / weights_sum
    zi = zi_flat.reshape(grid_x.shape)

    return zi


def loocv_idw(points, values, power_values=[1, 1.5, 2, 2.5, 3]):
    """
    Perform leave-one-out cross-validation to estimate the uncertainty produced by
    the IDW spatial interpolation
    """

    results = {}

    for p in power_values:
        residuals = []
        for i in range(len(points)):
            # Create leave-one-out dataset
            train_points = np.delete(points, i, axis=0)
            train_values = np.delete(values, i)
            
            # Create single-point grid for test location
            test_grid = (np.array([[points[i, 0]]]), np.array([[points[i, 1]]]))
            
            # Predict left-out value
            pred = idw(train_points, train_values, test_grid, power=p)
            residuals.append(values[i] - pred[0, 0])
        
        # Calculate error metrics
        results[p] = {
            'rmse': np.sqrt(np.mean(np.square(residuals))),
            'mae': np.mean(np.abs(residuals)),
            'residuals': np.array(residuals)
        }
    
    return results

def spatial_uncertainty(points, values, xi, power_range=[1, 1.5, 2, 2.5, 3]):
    """
    Generate spatial uncertainty map through ensemble modeling
    """

    # Create interpolation grid
    grid_x, grid_y = xi
    
    # Generate ensemble predictions
    predictions = []
    for p in power_range:
        pred_grid = idw(points, values, xi, power=p)
        predictions.append(pred_grid)
    
    # Calculate standard deviation across parameter variations
    uncertainty = np.std(predictions, axis=0)
    uncertainty_grid = uncertainty.reshape(grid_x.shape)
    
    return uncertainty_grid
    
### Kriging ###

# Variogram models
def spherical_variogram(h, sill, range_):
    h = np.asarray(h)
    return sill * (1.5 * (h / range_) - 0.5 * (h / range_)**3) * (h <= range_) + sill * (h > range_)

def gaussian_variogram(h, sill, range_):
    return sill * (1 - np.exp(-(h**2 / range_**2)))

def exponential_variogram(h, sill, range_):
    return sill * (1 - np.exp(-h / range_))

# Kriging interpolation
def ordinary_kriging(points, values, xi, variogram_model, sill=1, range_=0.0, num_bins=20):
    """
    Ordinary Kriging interpolation using a given variogram model.

    Parameters:
        points : (n, 2) array
            Coordinates of known data points.
        values : (n,) array
            Values at known data points.
        xi : tuple of 2D arrays (grid_x, grid_y)
            Grid coordinates for prediction.
        variogram_model : callable
            Variogram model function (e.g. exponential_variogram).
        sill : float
            Sill value for variogram. In ordinary kriging this parameter has no influence
        range_ : float
            Range value for variogram (optional, set to 0.0 to auto-fit).
        num_bins : int
            Number of bins for the experimental variogram.

    Returns:
        zi : 2D array
            Interpolated values on the grid.
        sill : float
            Fitted or provided sill value.
        range_ : float
            Fitted or provided range value.
    """

    # Convert inputs to NumPy arrays for safety
    points = np.asarray(points)
    values = np.asarray(values)

    # Unpack the grid coordinates
    grid_x, grid_y = xi

    # Flatten the grid to a list of (x, y) coordinates for interpolation
    xi_flat = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # STEP 1: Compute pairwise distances between known data points
    dists = pdist(points)

    # Compute empirical semivariances between all pairs of values
    semivariance = pdist(values.reshape(-1, 1), metric='sqeuclidean') / 2

    # STEP 2: Bin distances to build the experimental variogram
    bin_edges = np.linspace(0, np.max(dists), num_bins + 1)         # Define bin edges
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])            # Compute bin centers
    bin_indices = np.digitize(dists, bin_edges)                     # Assign each distance to a bin

    # STEP 3: Calculate average semivariance per bin (experimental variogram)
    experimental = np.array([
        np.mean(semivariance[bin_indices == i]) if np.any(bin_indices == i) else np.nan
        for i in range(1, num_bins + 1)
    ])

    # STEP 4: Fit the variogram model if sill or range are not provided
    if range_ == 0.0:
        # Objective function: minimize squared error between model and experimental points
        def objective(params):
            sill, range_ = params
            model = variogram_model(bin_centers, sill, range_)
            return np.nanmean((model - experimental)**2)

        # Initial guesses and bounds for optimization
        d0 = 50   # Initial guess for range
        v0 = 100  # Initial guess for sill
        res = minimize(objective, [d0, v0], bounds=[(1e-6, None), (1e-6, None)])
        range_, sill = res.x  # Extract optimized values
        model = variogram_model(bin_centers, sill, range_)  # Fitted model
    else:
        # Use provided sill and range
        model = variogram_model(bin_centers, sill, range_)

    # STEP 5: Plot experimental vs fitted variogram
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, model, label='Fitted model', color='black')
    plt.scatter(bin_centers, experimental, color='red', label='Experimental')
    plt.xlabel('Distance')
    plt.ylabel('Semivariance')
    plt.title('Experimental vs. Fitted Variogram')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # STEP 6: Build the kriging system (matrix of semivariances between known points)
    d_matrix = squareform(dists)  # Convert to square distance matrix
    K = variogram_model(d_matrix, sill, range_)  # Apply variogram to compute matrix

    # Add small noise to diagonal to avoid numerical issues (regularization)
    K += 1e-10 * np.eye(len(points))

    # Invert kriging matrix
    K_inv = np.linalg.inv(K)

    # STEP 7: Compute variogram between grid points and known points
    d_pred = cdist(xi_flat, points)  # Distances from grid points to known points
    k_vectors = variogram_model(d_pred, sill, range_)  # Apply variogram

    # STEP 8: Compute predicted values using kriging weights
    z_pred = np.dot(k_vectors, np.dot(K_inv, values))

    # STEP 9: Reshape predictions to the original grid shape
    zi = z_pred.reshape(grid_x.shape)

    return zi, sill, range_
