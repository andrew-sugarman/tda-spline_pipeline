import re
from pathlib import Path
import pandas as pd
import numpy as np
import skfda
from skfda.representation.basis import BSplineBasis
import matplotlib.pyplot as plt

# ---------------------------
# 1. Batch Processing Setup
# ---------------------------
# Define the root output folder containing one folder per section scan.
# Each folder is named following the pattern: [sampleID]_[sectionX]_[additionalinfo]
root_dir = Path("/home/andrew/thesis/prostate_cancer/data/multiseg_proc_output")  # <--- Change this to your actual output folder

# Regex to extract sample and section info from the section scan folder name.
# Example: "jrw004_section1_bac2-1.25_crop1_reslice_bottom" -> sample: "jrw004", section: "section1"
section_folder_re = re.compile(r"^(?P<sample>[^_]+)_(?P<section>section\d+)_.*")

# Regex to extract chunk info from the CSV file name.
# Example: "chunk_Z0-256_Y0-256_X0-256_ec.csv" will capture "Z0-256_Y0-256_X0-256"
chunk_re = re.compile(r"chunk_(?P<chunk>.+)_ec")

# List to store metadata and curve data for each chunk.
records = []

# Iterate over each section scan folder in the root directory.
for section_folder in root_dir.iterdir():
    if not section_folder.is_dir():
        continue

    # Parse sample and section from the folder name.
    m = section_folder_re.match(section_folder.name)
    if m:
        sample_id = m.group("sample")
        section_id = m.group("section")
    else:
        sample_id = section_folder.name
        section_id = None

    # Look for the alpha complex filtration results.
    # We assume that within each section folder, there is a subfolder (or nested subfolder)
    # named "topology_results" that contains the Euler characteristic CSV files.
    topology_dirs = list(section_folder.rglob("topology_results"))
    if not topology_dirs:
        print(f"No 'topology_results' folder found in {section_folder}")
        continue
    # Use the first found "topology_results" folder.
    topology_dir = topology_dirs[0]

    # Process each CSV file that ends with '_ec.csv'
    for csv_file in topology_dir.glob("*_ec.csv"):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        # Ensure the CSV has the expected columns.
        if "Filtration Value" not in df.columns or "Raw Euler Characteristic" not in df.columns:
            print(f"Skipping {csv_file.name} (missing required columns)")
            continue

        # Get the filtration values and raw Euler curve.
        filtration = df["Filtration Value"].values
        raw_ec = df["Raw Euler Characteristic"].values

        # Extract chunk info from the CSV file name.
        chunk_match = chunk_re.search(csv_file.stem)
        if chunk_match:
            chunk_id = chunk_match.group("chunk")
        else:
            chunk_id = csv_file.stem

        # Save a record with metadata and the curve data.
        records.append({
            "sample": sample_id,
            "section": section_id,
            "chunk": chunk_id,
            "filtration": filtration,
            "raw_ec": raw_ec,
            "file": str(csv_file)
        })

# Build a metadata DataFrame for review.
metadata_df = pd.DataFrame(records)
print("Metadata summary:")
print(metadata_df.head())
print("Total records processed:", len(metadata_df))

# ----------------------------------------------------
# 2. Fitting B-spline Basis Functions to Each Chunk
# ----------------------------------------------------
# We will process each chunk’s raw Euler curve individually.
# Here we fit a B-spline basis (using, for example, 10 basis functions) to each curve.
# Then we plot all fitted curves on one large plot, coloring by sample.

plt.figure(figsize=(12, 8))

# Map each unique sample to a color using a colormap.
unique_samples = metadata_df["sample"].unique()
cmap = plt.cm.get_cmap("tab10", len(unique_samples))
sample_colors = {s: cmap(i) for i, s in enumerate(unique_samples)}

# This set will help us add one legend entry per sample.
legend_plotted = set()

# Number of basis functions for B-spline fitting.
n_basis = 10

for record in records:
    sample = record["sample"]
    filtration = record["filtration"]
    raw_ec = record["raw_ec"]

    # Create a functional data object for the raw Euler curve.
    fd = skfda.FDataGrid(data_matrix=raw_ec.reshape(1, -1), grid_points=filtration)
    
    # Define a B-spline basis on the domain of the current curve.
    basis = BSplineBasis(domain_range=(filtration.min(), filtration.max()), n_basis=n_basis)
    
    # Convert the functional data to its B-spline basis representation.
    fd_basis = fd.to_basis(basis)
    
    # Evaluate the B-spline representation on a fine grid for a smooth curve.
    fine_grid = np.linspace(filtration.min(), filtration.max(), 200)
    spline_vals = fd_basis(fine_grid)[0]
    
    # Plot the spline curve; add a legend label only once per sample.
    label = sample if sample not in legend_plotted else None
    if label is not None:
        legend_plotted.add(sample)
    plt.plot(fine_grid, spline_vals, color=sample_colors[sample], label=label, alpha=0.7)

plt.xlabel("Filtration Value")
plt.ylabel("Raw Euler Characteristic")
plt.title("B-spline Fit of Raw Euler Characteristic Curves (per chunk) by Sample")
plt.legend(title="Sample ID")
plt.show()

### ADDITIONAL COMPARISONS IN THE SPACE OF FUNCTIONS ###

import numpy as np
import skfda
from skfda.representation.basis import BSplineBasis

# List to store functional data objects for each chunk along with metadata.
fd_records = []

# Number of basis functions (adjust as needed)
n_basis = 10

for record in records:
    sample = record["sample"]
    section = record["section"]
    chunk = record["chunk"]
    filtration = record["filtration"]
    raw_ec = record["raw_ec"]
    
    # Create a functional data object from the raw Euler curve.
    fd = skfda.FDataGrid(data_matrix=raw_ec.reshape(1, -1), grid_points=filtration)
    
    # Fit a B-spline basis to obtain a smooth representation.
    basis = BSplineBasis(domain_range=(filtration.min(), filtration.max()), n_basis=n_basis)
    fd_basis = fd.to_basis(basis)
    
    # Save the FD object and its metadata.
    fd_records.append({
        "sample": sample,
        "section": section,
        "chunk": chunk,
        "fd_basis": fd_basis
    })

# Assume all curves share the same filtration range.
global_grid = np.linspace(records[0]["filtration"].min(), records[0]["filtration"].max(), 200)

# Evaluate each functional data object on the common grid.
fd_values = []
for rec in fd_records:
    # Evaluate the B-spline representation on the common grid.
    values = rec["fd_basis"](global_grid)[0]  # result is a 1D array for one function
    fd_values.append(values)

# Create a single FDataGrid object combining all curves.
fd_aggregated = skfda.FDataGrid(data_matrix=np.array(fd_values), grid_points=global_grid)





#### IM PICKLE RICK ####
import pickle

# Save the aggregated functional data object to a file.
with open("fd_aggregated.pkl", "wb") as f:
    pickle.dump(fd_aggregated, f) 
#### - HAH GOTEEM - ####




from skfda.preprocessing.dim_reduction.feature_extraction import FPCA

# Specify the number of principal components to extract.
n_components = 3
fpca = FPCA(n_components=n_components)
fpca_scores = fpca.fit_transform(fd_aggregated)

print("FPCA scores shape:", fpca_scores.shape)
print("Explained variance ratio:", fpca.explained_variance_ratio_)

fpca.components_.plot()

plt.title("Functional Principal Components")
plt.show()

from sklearn.cluster import KMeans

# Decide on the number of clusters (for example, 3)
kmeans = KMeans(n_clusters=6, random_state=0)
clusters = kmeans.fit_predict(fpca_scores)

# Add the cluster assignments to your metadata DataFrame.
cluster_assignments = clusters  # This will be an array with one entry per curve.

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(fpca_scores[:, 0], fpca_scores[:, 1], c=clusters, cmap="viridis", s=50)
plt.xlabel("FPCA Component 1")
plt.ylabel("FPCA Component 2")
plt.title("Clustering of Euler Characteristic Curves (FPCA Space)")
plt.colorbar(label="Cluster")
plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Assume that fpca_scores is a NumPy array of shape (n_samples, n_components)
# and that fd_records is a list of dictionaries where each dict has a "sample" key.
# Here, we extract the sample IDs in the same order as the FPCA scores.
samples = [record["sample"] for record in fd_records]

# Identify unique samples and map them to colors.
unique_samples = sorted(set(samples))
cmap = plt.cm.get_cmap("tab10", len(unique_samples))
sample_color_map = {s: cmap(i) for i, s in enumerate(unique_samples)}

# Create a color list for each observation based on its sample ID.
colors = [sample_color_map[s] for s in samples]

plt.figure(figsize=(8, 6))
plt.scatter(fpca_scores[:, 0], fpca_scores[:, 1], c=colors, s=50)

plt.xlabel("FPCA Component 1")
plt.ylabel("FPCA Component 2")
plt.title("FPCA Scores Colored by Sample ID")

# Create legend handles for each unique sample.
handles = [mpatches.Patch(color=sample_color_map[s], label=s) for s in unique_samples]
plt.legend(handles=handles, title="Sample ID")

plt.show()

from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Assume fpca_scores is a NumPy array of shape (n_samples, n_components)
# and that fd_records (or metadata_df) contains the corresponding sample IDs.
samples = [record["sample"] for record in fd_records]
unique_samples = sorted(set(samples))
cmap = plt.cm.get_cmap("tab10", len(unique_samples))
sample_color_map = {s: cmap(i) for i, s in enumerate(unique_samples)}
colors = [sample_color_map[s] for s in samples]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(fpca_scores[:, 0], fpca_scores[:, 1], fpca_scores[:, 2],
                c=colors, s=5)

ax.set_xlabel("FPCA Component 1")
ax.set_ylabel("FPCA Component 2")
ax.set_zlabel("FPCA Component 3")
ax.set_title("3D FPCA Scores Colored by Sample ID")

# Create legend handles for each unique sample.
handles = [mpatches.Patch(color=sample_color_map[s], label=s) for s in unique_samples]
plt.legend(handles=handles, title="Sample ID")
plt.show()
