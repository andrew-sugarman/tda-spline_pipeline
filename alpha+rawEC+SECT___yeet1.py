#!/usr/bin/env python
"""
process_alpha_filtration.py

This script computes the alpha complex filtration for 3D point cloud CSV files.
For each point cloud (chunk):
  1. Build an Alpha Complex using the Gudhi library.
  2. Compute the raw Euler characteristic (EC) at a series of filtration steps.
  3. Compute the smoothed Euler characteristic (SEC) by centering and integrating the raw EC.
  4. Save the results in a CSV with columns:
       - Filtration Value
       - Raw EC (3D)
       - Smoothed EC (3D)

Input Arguments:
  --input_folder: Folder containing point cloud CSV files (e.g., the 'pointclouds' subfolder from extraction).
  --output_folder: Folder where the topology results will be saved.

Output:
  A subfolder named "topology_results" is created inside the specified output folder.
  For each point cloud CSV, a corresponding CSV file is saved with the computed filtration curves.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import gudhi as gd
from scipy.integrate import cumulative_trapezoid

# Define filtration steps (can be adjusted if needed)
FILTRATION_STEPS = np.linspace(0, 100, 25)

def compute_alpha_complex_filtration(csv_file, filtration_steps=FILTRATION_STEPS):
    """
    Given a point cloud CSV (with columns: Z, Y, X), build an alpha complex,
    compute the raw Euler characteristic (EC) at each filtration step,
    and compute the smoothed EC (SEC) by centering and integrating the raw EC.

    Returns:
      filtration_steps: numpy array of filtration values.
      raw_ec: numpy array of raw Euler characteristic values.
      sec_curve: numpy array of the smoothed Euler characteristic.
    Returns None if the CSV file is empty.
    """
    df = pd.read_csv(csv_file)
    if df.empty:
        return None

    points = df[['Z','Y','X']].values
    alpha_complex = gd.AlphaComplex(points=points)
    simplex_tree = alpha_complex.create_simplex_tree()

    raw_ec = []
    for r in filtration_steps:
        # Count simplices in or below the filtration radius r
        V = sum(1 for simplex, filt in simplex_tree.get_skeleton(0) if len(simplex)==1 and filt <= r)
        E = sum(1 for simplex, filt in simplex_tree.get_skeleton(1) if len(simplex)==2 and filt <= r)
        F = sum(1 for simplex, filt in simplex_tree.get_skeleton(2) if len(simplex)==3 and filt <= r)
        T = sum(1 for simplex, filt in simplex_tree.get_skeleton(3) if len(simplex)==4 and filt <= r)
        ec = V - E + F - T
        raw_ec.append(ec)
    raw_ec = np.array(raw_ec)

    # Compute smoothed EC by centering and cumulative integration
    centered_ec = raw_ec - np.mean(raw_ec)
    sec_curve = cumulative_trapezoid(centered_ec, filtration_steps, initial=0)

    return filtration_steps, raw_ec, sec_curve

def process_point_clouds(input_folder, output_folder):
    """
    Process all point cloud CSV files in the input_folder.
    For each file, compute the alpha complex filtration and save the results as a CSV
    with columns: Filtration Value, Raw EC (3D), Smoothed EC (3D).

    The output files are saved in a subfolder 'topology_results' within the output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    topology_dir = os.path.join(output_folder, "topology_results")
    os.makedirs(topology_dir, exist_ok=True)

    # Get list of all CSV files in input_folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_folder}. Exiting.")
        return

    for csv_file in csv_files:
        result = compute_alpha_complex_filtration(csv_file, FILTRATION_STEPS)
        if result is None:
            print(f"Skipping empty file: {csv_file}")
            continue

        filtration_vals, raw_ec, sec_curve = result

        # Define output file name
        base = os.path.splitext(os.path.basename(csv_file))[0]
        output_csv = os.path.join(topology_dir, f"{base}_filtration.csv")

        # Save the results to CSV
        pd.DataFrame({
            "Filtration Value": filtration_vals,
            "Raw EC (3D)": raw_ec,
            "Smoothed EC (3D)": sec_curve
        }).to_csv(output_csv, index=False)
        print(f"Saved filtration results to {output_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Process point cloud CSV files to compute alpha complex filtration (raw and smoothed Euler characteristic curves)."
    )
    parser.add_argument("--input_folder", required=True, help="Folder containing point cloud CSV files (e.g., the 'pointclouds' folder).")
    parser.add_argument("--output_folder", required=True, help="Output folder to save topology results (filtration curves).")
    args = parser.parse_args()

    process_point_clouds(args.input_folder, args.output_folder)
    print("Alpha complex filtration processing completed.")

if __name__ == "__main__":
    main()
