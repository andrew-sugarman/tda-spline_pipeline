#!/usr/bin/env python
"""
multi_mask_pipeline.py

A script to process multiple 3D segmentation masks in a folder. For each mask:
1) Extract centroids & create overlapping chunks.
2) Compute alpha complexes & SEC curves for each chunk.
3) Generate interactive dashboards for visualization.
4) Produce a summary report with aggregated plots & statistics.

Requires: numpy, pandas, scikit-image, matplotlib, plotly, gudhi, etc.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from skimage import io, measure
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import gudhi as gd
from scipy.integrate import cumulative_trapezoid

# ---------------------
# Parameters
# ---------------------
CHUNK_SIZE = 256
DESIRED_OVERLAP = 16
STRIDE = CHUNK_SIZE - DESIRED_OVERLAP

# Choose filtration steps for the alpha complex
FILTRATION_STEPS = np.linspace(0, 100, 25)

# ---------------------
# Helper Functions
# ---------------------
def generate_chunk_boundaries(image_shape, chunk_size=CHUNK_SIZE, stride=STRIDE):
    """
    Generate overlapping cube boundaries for a 3D volume given 'chunk_size' and 'stride' (chunk_size - overlap).
    Returns a list with three sub-lists, one per dimension: [(start, end, padded_flag, pad_amount), ...]
    """
    boundaries = []
    for dim_len in image_shape:
        dim_boundaries = []
        for start in range(0, dim_len, stride):
            end = start + chunk_size
            if end > dim_len:
                pad_amount = end - dim_len
                end = dim_len
                padded_flag = True
            else:
                pad_amount = 0
                padded_flag = False
            dim_boundaries.append((start, end, padded_flag, pad_amount))
        boundaries.append(dim_boundaries)
    return boundaries

def extract_centroids(segmentation_mask):
    """
    Given a 3D segmentation mask (Z x Y x X), extract region centroids.
    Returns a pandas DataFrame with columns [Z, Y, X, Label].
    """
    regions = measure.regionprops(segmentation_mask)
    centroids = [r.centroid for r in regions]
    centroids_array = np.array(centroids)
    df_centroids = pd.DataFrame(centroids_array, columns=["Z", "Y", "X"])
    df_centroids["Label"] = [r.label for r in regions]
    return df_centroids

def create_chunks_and_save_pointclouds(df_centroids, image_shape, output_dir):
    """
    Using the full centroid list, split it into overlapping 3D chunks and save each chunk as a CSV.
    Also records a metadata CSV with chunk boundaries, border flags, etc.

    Returns the path to the metadata CSV.
    """
    chunk_boundaries = generate_chunk_boundaries(image_shape, CHUNK_SIZE, STRIDE)

    # Prepare output dirs
    pc_dir = os.path.join(output_dir, "pointclouds")
    os.makedirs(pc_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "chunk_metadata.csv")
    metadata_list = []

    # Loop over all chunk combos
    for z_info in chunk_boundaries[0]:
        for y_info in chunk_boundaries[1]:
            for x_info in chunk_boundaries[2]:
                z_start, z_end, z_pad_flag, z_pad_amount = z_info
                y_start, y_end, y_pad_flag, y_pad_amount = y_info
                x_start, x_end, x_pad_flag, x_pad_amount = x_info

                # Filter centroids within this chunk
                in_chunk = (
                    (df_centroids["Z"] >= z_start) & (df_centroids["Z"] < z_end) &
                    (df_centroids["Y"] >= y_start) & (df_centroids["Y"] < y_end) &
                    (df_centroids["X"] >= x_start) & (df_centroids["X"] < x_end)
                )
                chunk_pts = df_centroids[in_chunk].copy()

                # Define a filename for the chunk's point cloud
                chunk_file = f"chunk_Z{z_start}-{z_end}_Y{y_start}-{y_end}_X{x_start}-{x_end}.csv"
                chunk_path = os.path.join(pc_dir, chunk_file)
                chunk_pts.to_csv(chunk_path, index=False)

                # Metadata
                is_border = {
                    "Z": (z_start == 0 or z_end == image_shape[0]),
                    "Y": (y_start == 0 or y_end == image_shape[1]),
                    "X": (x_start == 0 or x_end == image_shape[2]),
                }
                metadata_list.append({
                    "chunk_bounds": {"Z": [z_start, z_end],
                                     "Y": [y_start, y_end],
                                     "X": [x_start, x_end]},
                    "is_border": is_border,
                    "is_padded": (z_pad_flag or y_pad_flag or x_pad_flag),
                    "padding": {"Z": z_pad_amount, "Y": y_pad_amount, "X": x_pad_amount},
                    "num_points": len(chunk_pts),
                    "point_cloud_file": chunk_path
                })

    md_df = pd.DataFrame(metadata_list)
    md_df.to_csv(metadata_path, index=False)
    return metadata_path

def compute_alpha_complex_sec(csv_file, filtration_steps=FILTRATION_STEPS):
    """
    Given a point cloud CSV (with Z, Y, X columns), build an alpha complex,
    compute the Euler characteristic at each r, and produce the 'Smoothed EC' (SEC).
    Returns the SEC curve as a numpy array, or None if the CSV is empty.
    """
    df = pd.read_csv(csv_file)
    if df.empty:
        return None

    points = df[['Z','Y','X']].values
    alpha_complex = gd.AlphaComplex(points=points)
    simplex_tree = alpha_complex.create_simplex_tree()

    euler_characteristics = []
    for r in filtration_steps:
        # Count simplices in or below this filtration radius
        V = sum(1 for simplex,filt in simplex_tree.get_skeleton(0) if len(simplex)==1 and filt <= r)
        E = sum(1 for simplex,filt in simplex_tree.get_skeleton(1) if len(simplex)==2 and filt <= r)
        F = sum(1 for simplex,filt in simplex_tree.get_skeleton(2) if len(simplex)==3 and filt <= r)
        T = sum(1 for simplex,filt in simplex_tree.get_skeleton(3) if len(simplex)==4 and filt <= r)
        ec = V - E + F - T
        euler_characteristics.append(ec)

    euler_characteristics = np.array(euler_characteristics)
    # Center and integrate => "Smoothed EC"
    centered_ec = euler_characteristics - np.mean(euler_characteristics)
    sec_curve = cumulative_trapezoid(centered_ec, filtration_steps, initial=0)
    return sec_curve

def process_one_mask(mask_path, output_parent_dir):
    """
    End-to-end processing for a single segmentation mask file.
    1) Load mask
    2) Extract centroids
    3) Chunk + save CSV point clouds
    4) Compute SEC curves for all chunks
    5) Produce aggregated plots (Plotly or Matplotlib)
    """
    # -------------------------------------
    # A) Prepare output directories
    # -------------------------------------
    mask_name = os.path.splitext(os.path.basename(mask_path))[0]
    output_dir = os.path.join(output_parent_dir, f"{mask_name}_output")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------
    # B) Load segmentation mask
    # -------------------------------------
    print(f"\n[INFO] Processing mask: {mask_path}")
    seg_mask = io.imread(mask_path)  # shape: (Z, Y, X)
    print(f" --> Loaded mask with shape {seg_mask.shape}.")

    # -------------------------------------
    # C) Extract centroids
    # -------------------------------------
    df_centroids = extract_centroids(seg_mask)
    print(f" --> Found {len(df_centroids)} labeled regions.")

    # Save full centroids for reference
    full_centroids_csv = os.path.join(output_dir, "full_centroids.csv")
    df_centroids.to_csv(full_centroids_csv, index=False)

    # -------------------------------------
    # D) Create overlapping chunks & save point clouds
    # -------------------------------------
    metadata_csv = create_chunks_and_save_pointclouds(df_centroids, seg_mask.shape, output_dir)
    print(f" --> Chunk metadata saved to {metadata_csv}")

    # -------------------------------------
    # E) For each chunk, compute alpha complex + SEC
    # -------------------------------------
    pc_dir = os.path.join(output_dir, "pointclouds")
    topology_dir = os.path.join(output_dir, "topology_results")
    os.makedirs(topology_dir, exist_ok=True)

    chunk_files = glob.glob(os.path.join(pc_dir, "*.csv"))
    all_sec_curves = []
    for cf in chunk_files:
        sec = compute_alpha_complex_sec(cf, FILTRATION_STEPS)
        if sec is None:
            continue
        # Save the SEC to CSV
        base = os.path.splitext(os.path.basename(cf))[0]
        sec_csv = os.path.join(topology_dir, f"{base}_sec.csv")
        pd.DataFrame({
            "Filtration Value": FILTRATION_STEPS,
            "Smoothed EC (3D)": sec
        }).to_csv(sec_csv, index=False)
        all_sec_curves.append(sec)

    print(f" --> Computed SEC curves for {len(all_sec_curves)} chunks.")

    # -------------------------------------
    # F) Generate aggregated dashboards
    # -------------------------------------
    #  (i) Mean SEC + error bands
    #  (ii) Boxplot at selected filtration steps
    if all_sec_curves:
        curves_array = np.array(all_sec_curves)
        mean_sec = curves_array.mean(axis=0)
        std_sec = curves_array.std(axis=0)

        # Example Plotly figure for aggregated SEC
        fig = go.Figure()
        # Add mean curve
        fig.add_trace(go.Scatter(
            x=FILTRATION_STEPS,
            y=mean_sec,
            mode='lines',
            name='Mean SEC',
            line=dict(color='blue')
        ))
        # Add error band
        fig.add_trace(go.Scatter(
            x=np.concatenate([FILTRATION_STEPS, FILTRATION_STEPS[::-1]]),
            y=np.concatenate([mean_sec + std_sec, (mean_sec - std_sec)[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Std. Dev.'
        ))
        fig.update_layout(
            title=f"Aggregated SEC Curves for {mask_name}",
            xaxis_title="Filtration Value (r)",
            yaxis_title="Smoothed EC (3D)",
            legend_title="Curves"
        )
        fig_path = os.path.join(output_dir, "aggregated_SEC.html")
        fig.write_html(fig_path)
        print(f" --> Aggregated SEC plot saved as {fig_path}")
    else:
        print(" --> No SEC curves computed (possibly empty chunk files). Skipping dashboard.")

    # -------------------------------------
    # G) Optionally: Return summary info
    # -------------------------------------
    return {
        "mask_path": mask_path,
        "num_regions": len(df_centroids),
        "num_chunks": len(chunk_files),
        "num_sec": len(all_sec_curves),
        "output_dir": output_dir
    }


def create_final_notebook(all_summaries, output_notebook="final_report.ipynb"):
    """
    (Optional) Demonstration stub for how you might create a summary Jupyter Notebook
    from the aggregated results. In a real-world scenario, you might use nbformat or papermill
    to programmatically generate or fill in a notebook template with these results.
    """
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell

    nb = new_notebook()
    nb.cells.append(new_markdown_cell("# Multi-Scan Processing Report\n"))
    
    lines = [
        "This notebook summarizes the processing of multiple segmentation masks.\n",
        "## Summary Table"
    ]
    summaries_df = pd.DataFrame(all_summaries)
    lines.append(summaries_df.to_markdown(index=False))
    
    nb.cells.append(new_markdown_cell("\n".join(lines)))

    with open(output_notebook, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[INFO] Wrote final notebook report to {output_notebook}")


def main():
    # ---------------------
    # Parse command line
    # ---------------------
    parser = argparse.ArgumentParser(
        description="Process multiple 3D segmentation masks in a folder, generating overlapping chunks, SEC curves, and summary reports."
    )
    parser.add_argument("--input_folder", required=True, help="Folder containing 3D segmentation masks (e.g., .tif)")
    parser.add_argument("--output_folder", required=True, help="Output folder for results.")
    parser.add_argument("--make_notebook", action="store_true", help="Flag to create a final Jupyter Notebook report.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Find all TIF (or other 3D) files
    mask_paths = glob.glob(os.path.join(args.input_folder, "*.tif"))
    mask_paths += glob.glob(os.path.join(args.input_folder, "*.tiff"))
    if not mask_paths:
        print(f"No .tif or .tiff files found in {args.input_folder}. Exiting.")
        return

    all_summaries = []
    for mp in mask_paths:
        summary = process_one_mask(mp, args.output_folder)
        all_summaries.append(summary)

    # Optionally create a final Jupyter Notebook summarizing everything
    if args.make_notebook:
        notebook_path = os.path.join(args.output_folder, "final_report.ipynb")
        create_final_notebook(all_summaries, notebook_path)

    print("\nAll done!")


if __name__ == "__main__":
    main()
