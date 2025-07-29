import argparse
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Clear GPU memory
torch.cuda.empty_cache()


# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=2000):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file1', type=str, help='Path to the data matrix file (.pt)')
parser.add_argument('--file2', type=str, help='Path to the labels CSV file')
args = parser.parse_args()

# Check device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data matrix
try:
    data_matrix = torch.load(args.file1)
    if len(data_matrix.shape) != 3:
        raise ValueError("Input data must have shape [n, m, q]")
    n, m, q = data_matrix.shape
    data_matrix = data_matrix.reshape(n, m * q).float()  # Flatten to [n, m*q]
    print(f"Original data shape: {data_matrix.shape}")
except Exception as e:
    print(f"Error loading or processing {args.file1}: {e}")
    exit(1)

# Load labels
try:
    labels_df = pd.read_csv(args.file2)
    if "Antigen" not in labels_df or "label" not in labels_df:
        raise ValueError("CSV must contain 'Antigen' and 'label' columns")
    labels = labels_df["Antigen"]
    targets = labels_df["label"]
except Exception as e:
    print(f"Error loading or processing {args.file2}: {e}")
    exit(1)

# Validate data alignment
if len(data_matrix) != len(labels):
    print(f"Error: Data matrix ({len(data_matrix)}) and labels ({len(labels)}) have different lengths")
    exit(1)

# Create DataLoader for batch processing
batch_size = 128
dataset = TensorDataset(data_matrix)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Initialize MLP
mlp = MLP(input_dim=m * q, output_dim=128).to(device)
mlp.eval()  # Set to evaluation mode

# Compress data with MLP in batches
mapped_data = []
with torch.no_grad():
    for batch in dataloader:
        inputs = batch[0].to(device)
        outputs = mlp(inputs)
        mapped_data.append(outputs.cpu())
mapped_data = torch.cat(mapped_data).numpy()  # [n, 2000]
print(f"Mapped data shape: {mapped_data.shape}")

# Define target protein sequences
target_labels = [
    "AVFDRKSDAK", "KLGGALQAK", "GILGFVFTL", "YVLDHLIVV",
    "GLCTLVAML", "IVTDFSVIK", "LLWNGPMAV", "RAKFKQLL"
]

# Filter data
mask = (targets == 1) & (labels.isin(target_labels))
filtered_data_matrix = mapped_data[mask]
filtered_labels = labels[mask].values
filtered_targets = targets[mask]

print(f"Filtered data shape: {filtered_data_matrix.shape}")
print(f"Filtered labels count: {len(filtered_labels)}")
print(f"Filtered target values: {np.unique(filtered_targets)}")

# Perform UMAP and visualize
if len(filtered_data_matrix) > 0:
    # Dynamic UMAP parameters
    n_samples = len(filtered_data_matrix)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_result = reducer.fit_transform(filtered_data_matrix)

    # Assign colors
    unique_labels = target_labels
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))

    # Plot
    plt.figure(figsize=(10, 8))
    scatters = {}
    for label in unique_labels:
        mask_label = filtered_labels == label
        if mask_label.any():
            scatter = plt.scatter(
                umap_result[mask_label, 0], umap_result[mask_label, 1],
                c=[label_to_color[label]], label=label, alpha=0.6
            )
            scatters[label] = scatter

    # Warn about missing sequences
    missing_labels = [label for label in unique_labels if label not in filtered_labels]
    if missing_labels:
        print(f"Warning: No data for sequences: {missing_labels}")

    # Add legend and labels
    plt.legend(
        handles=[scatters[label] for label in unique_labels if label in scatters],
        title="Protein Sequences", loc="best"
    )
    plt.title("UMAP Dimensionality Reduction for Specific Protein Sequences (target=1)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.savefig("umap_plot2.png", dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("Warning: No data satisfies conditions (target=1 and in target_labels).")