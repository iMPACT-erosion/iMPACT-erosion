import numpy as np
import matplotlib.pyplot as plt

# Function to classify raster values into bins
def classify_raster(values, num_bins):
    bins = np.percentile(values, np.linspace(0, 100, num_bins + 1)[1:-1])
    return np.digitize(values, bins, right=True), bins

# Function to plot histograms
def plot_histograms(raster_values, num_bins):
    fig, axes = plt.subplots(1, len(raster_values), figsize=(15, 5))
    if len(raster_values) == 1:
        axes = [axes]
    
    for ax, (key, values) in zip(axes, raster_values.items()):
        classes, bins = classify_raster(values, num_bins)
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
        for bin_edge in bins:
            ax.axvline(bin_edge, color='red', linestyle='dashed', linewidth=1)
        ax.set_title(f"Histogram of {key}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()


# Function to perform stratified sampling
def stratified_sampling(raster_dict, transform, num_bins=3, max_samples=50):
    valid_mask = np.all([~np.isnan(raster) for raster in raster_dict.values()], axis=0)
    raster_values = {key: raster[valid_mask] for key, raster in raster_dict.items()}
    plot_histograms(raster_values, num_bins)  # Plot histograms before sampling
    
    class_labels = []
    num_features = len(raster_dict)
    
    for i, (key, values) in enumerate(raster_values.items()):
        classes, _ = classify_raster(values, num_bins)
        class_labels.append(classes * (num_bins**(num_features - i - 1)))
   
    strata_labels = sum(class_labels)
    unique_strata, strata_counts = np.unique(strata_labels, return_counts=True)
    strata_proportions = strata_counts / strata_counts.sum()
    samples_per_stratum = (strata_proportions * max_samples).round().astype(int)
    
    sample_indices = []
    for strata, count in zip(unique_strata, samples_per_stratum):
        indices = np.where(strata_labels == strata)[0]
        if len(indices) > 0:
            selected = np.random.choice(indices, min(count, len(indices)), replace=False)
            sample_indices.extend(selected)
            
    total_samples = len(sample_indices)
    
    row_indices, col_indices = np.where(valid_mask)
    selected_rows, selected_cols = row_indices[sample_indices], col_indices[sample_indices]
    selected_coords = [transform * (col, row) for row, col in zip(selected_rows, selected_cols)]
    return total_samples, selected_coords, selected_cols, selected_rows