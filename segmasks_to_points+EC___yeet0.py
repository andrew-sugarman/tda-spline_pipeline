#!/usr/bin/env python
"""
multi_mask_pipeline_directory_traversal.py

This script processes 3D segmentation masks organized in a folder structure where each section-scan folder 
(e.g., "jrw004_section1_bac2-1.25_crop1_reslice_bottom") contains exactly one segmentation mask (a .tif/.tiff file).
For each segmentation mask, it:
  1. Loads the mask (assumed to be a 3D image in .tif/.tiff format).
  2. Extracts nuclear centroids using region properties.
  3. Chunks the original image space into overlapping sub-volumes based on a predefined chunk size and overlap.
  4. Filters the centroids into each chunk and saves the resulting point cloud as a CSV.
  5. Generates metadata (chunk boundaries, border flags, number of centroids, etc.) and computes the raw Euler 
     characteristic curve for each chunk.
  6. Optionally creates an aggregated dashboard and a summary Jupyter Notebook report.

The script mirrors the input folder structure. For example, if your input is:
    output_topo_speedrun_G3_fullmem_tile+memgrow/jrw004_section1_bac2-1.25_crop1_reslice_bottom/labels.tif
then the output for that file will be placed in:
    <output_root>/jrw004_section1_bac2-1.25_crop1_reslice_bottom/labels_output/

Usage:
    python multi_mask_pipeline_directory_traversal.py --input_root <input_folder> --output_root <output_folder> [--make_notebook]

Dependencies:
    numpy, pandas, scikit-image, matplotlib, plotly, gudhi, scipy, nbformat
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
    Also records a metadata CSV with chunk boundaries, border flags, number of centroids, etc.
    Returns the path to the metadata CSV.
    """
    chunk_boundaries = generate_chunk_boundaries(image_shape, CHUNK_SIZE, STRIDE)

    # Prepare output directories
    pc_dir = os.path.join(output_dir, "pointclouds")
    os.makedirs(pc_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "chunk_metadata.csv")
    metadata_list = []

    # Loop over all chunk combinations
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

                # Metadata: record the chunk boundaries and number of centroids
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
                    "num_centroids": len(chunk_pts),
                    "point_cloud_file": chunk_path
                })

    md_df = pd.DataFrame(metadata_list)
    md_df.to_csv(metadata_path, index=False)
    return metadata_path

def compute_alpha_complex_ec(csv_file, filtration_steps=FILTRATION_STEPS):
    """
    Given a point cloud CSV (with Z, Y, X columns), build an alpha complex,
    compute the raw Euler characteristic at each filtration value.
    Returns the raw Euler characteristic curve as a numpy array, or None if the CSV is empty.
    """
    df = pd.read_csv(csv_file)
    if df.empty:
        return None

    points = df[['Z','Y','X']].values
    alpha_complex = gd.AlphaComplex(points=points)
    simplex_tree = alpha_complex.create_simplex_tree()

    euler_characteristics = []
    for r in filtration_steps:
        # Count vertices, edges, faces, and tetrahedra up to the current filtration value
        V = sum(1 for simplex, filt in simplex_tree.get_skeleton(0) if len(simplex) == 1 and filt <= r)
        E = sum(1 for simplex, filt in simplex_tree.get_skeleton(1) if len(simplex) == 2 and filt <= r)
        F = sum(1 for simplex, filt in simplex_tree.get_skeleton(2) if len(simplex) == 3 and filt <= r)
        T = sum(1 for simplex, filt in simplex_tree.get_skeleton(3) if len(simplex) == 4 and filt <= r)
        ec = V - E + F - T
        euler_characteristics.append(ec)

    return np.array(euler_characteristics)

def process_one_mask(mask_path, output_parent_dir):
    """
    End-to-end processing for a single segmentation mask file.
      A) Load mask.
      B) Extract centroids.
      C) Chunk the full image space and save the chunked point cloud CSVs.
      D) Compute raw Euler characteristic curves for each chunk.
      E) Generate an aggregated dashboard.
    The output is saved in a folder named [maskname]_output inside output_parent_dir.
    Returns a summary dictionary.
    """
    # A) Prepare output directories
    mask_name = os.path.splitext(os.path.basename(mask_path))[0]
    output_dir = os.path.join(output_parent_dir, f"{mask_name}_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Processing mask: {mask_path}")
    # B) Load segmentation mask (assumed 3D: Z x Y x X)
    seg_mask = io.imread(mask_path)
    print(f" --> Loaded mask with shape {seg_mask.shape}.")

    # C) Extract centroids
    df_centroids = extract_centroids(seg_mask)
    print(f" --> Found {len(df_centroids)} labeled regions.")

    # Save full centroids for reference
    full_centroids_csv = os.path.join(output_dir, "full_centroids.csv")
    df_centroids.to_csv(full_centroids_csv, index=False)

    # D) Create overlapping chunks & save point clouds along with metadata
    metadata_csv = create_chunks_and_save_pointclouds(df_centroids, seg_mask.shape, output_dir)
    print(f" --> Chunk metadata saved to {metadata_csv}")

    # E) Compute raw Euler characteristic curves for each chunk (topological analysis)
    pc_dir = os.path.join(output_dir, "pointclouds")
    topology_dir = os.path.join(output_dir, "topology_results")
    os.makedirs(topology_dir, exist_ok=True)

    chunk_files = glob.glob(os.path.join(pc_dir, "*.csv"))
    all_ec_curves = []
    for cf in chunk_files:
        ec = compute_alpha_complex_ec(cf, FILTRATION_STEPS)
        if ec is None:
            continue
        base = os.path.splitext(os.path.basename(cf))[0]
        ec_csv = os.path.join(topology_dir, f"{base}_ec.csv")
        pd.DataFrame({
            "Filtration Value": FILTRATION_STEPS,
            "Raw Euler Characteristic": ec
        }).to_csv(ec_csv, index=False)
        all_ec_curves.append(ec)

    print(f" --> Computed raw Euler characteristic curves for {len(all_ec_curves)} chunks.")

    # F) Generate aggregated dashboard for raw Euler characteristic curves
    if all_ec_curves:
        curves_array = np.array(all_ec_curves)
        mean_ec = curves_array.mean(axis=0)
        std_ec = curves_array.std(axis=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=FILTRATION_STEPS,
            y=mean_ec,
            mode='lines',
            name='Mean Raw EC'
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([FILTRATION_STEPS, FILTRATION_STEPS[::-1]]),
            y=np.concatenate([mean_ec + std_ec, (mean_ec - std_ec)[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Std. Dev.'
        ))
        fig.update_layout(
            title=f"Aggregated Raw Euler Characteristic for {mask_name}",
            xaxis_title="Filtration Value (r)",
            yaxis_title="Raw Euler Characteristic"
        )
        fig_path = os.path.join(output_dir, "aggregated_EC.html")
        fig.write_html(fig_path)
        print(f" --> Aggregated dashboard saved as {fig_path}")
    else:
        print(" --> No Euler characteristic curves computed. Skipping dashboard generation.")

    return {
        "mask_path": mask_path,
        "num_regions": len(df_centroids),
        "num_chunks": len(chunk_files),
        "num_ec": len(all_ec_curves),
        "output_dir": output_dir
    }

def create_final_notebook(all_summaries, output_notebook="final_report.ipynb"):
    """
    (Optional) Create a summary Jupyter Notebook report aggregating the results from all masks.
    """
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell

    nb = new_notebook()
    nb.cells.append(new_markdown_cell("# Multi-Mask Processing Report\n"))
    
    lines = [
        "This notebook summarizes the processing of multiple segmentation masks.",
        "## Summary Table"
    ]
    summaries_df = pd.DataFrame(all_summaries)
    lines.append(summaries_df.to_markdown(index=False))
    
    nb.cells.append(new_markdown_cell("\n".join(lines)))

    with open(output_notebook, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[INFO] Wrote final notebook report to {output_notebook}")

def process_all_segmentation_masks(input_root, output_root, make_notebook=False):
    """
    Traverse the input_root directory, where each immediate subfolder is a section-scan folder 
    containing exactly one segmentation mask (.tif/.tiff). Process each mask using process_one_mask 
    and save outputs in a mirrored folder structure under output_root.
    """
    all_summaries = []
    # List immediate subdirectories in the input_root
    for subdir in os.listdir(input_root):
        subdir_path = os.path.join(input_root, subdir)
        if os.path.isdir(subdir_path):
            # Find the tif file (expect exactly one per folder)
            tif_files = glob.glob(os.path.join(subdir_path, '*.tif')) + \
                        glob.glob(os.path.join(subdir_path, '*.tiff'))
            if len(tif_files) != 1:
                print(f"[WARN] Folder {subdir_path} has {len(tif_files)} tif files, expected exactly one. Skipping.")
                continue
            seg_path = tif_files[0]
            print(f"\n[INFO] Processing segmentation mask: {seg_path}")
            
            # Mirror the section folder in the output directory
            rel_path = os.path.relpath(subdir_path, input_root)
            output_section_folder = os.path.join(output_root, rel_path)
            os.makedirs(output_section_folder, exist_ok=True)
            
            summary = process_one_mask(seg_path, output_section_folder)
            all_summaries.append(summary)
    if make_notebook and all_summaries:
        final_notebook_path = os.path.join(output_root, "final_report.ipynb")
        create_final_notebook(all_summaries, final_notebook_path)

def main():
    parser = argparse.ArgumentParser(
        description="Process section-scan 3D segmentation masks to extract centroids, chunk the image space, and compute topological features."
    )
    parser.add_argument("--input_root", required=True, help="Root folder containing section-scan folders (each with one segmentation mask)")
    parser.add_argument("--output_root", required=True, help="Output root folder where processed results will be saved")
    parser.add_argument("--make_notebook", action="store_true", help="Generate a final Jupyter Notebook report summarizing the results")
    args = parser.parse_args()

    process_all_segmentation_masks(args.input_root, args.output_root, args.make_notebook)
    print("\nAll segmentation masks processed.")

if __name__ == "__main__":
    main()
