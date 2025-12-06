"""
During the development of this project I used ChatGPT for high level guidance and small code examples, which I adapted and tested myself.
-----------------------------------------------------------
This script inspects MRI volumes (.nii files) in either a T1-only or T2-only dataset.

It will extract basic metadata (shape, voxel spacing, and depth in slices),
and write a CSV summary + a histogram of slice counts (z-depths).

Usage:
    python inspect_volumes.py [dataset_root]

If no argument is given, it defaults to "ms_T1_dataset" (T1-only dataset).
You can also pass "ms_T2_dataset" to analyze the T2-only dataset.
-----------------------------------------------------------
"""

# Library imports that will be used throughout the file
import sys
from pathlib import Path
import nibabel as nib  # Library for reading medical images (.nii in our specific case)
import numpy as np
import matplotlib.pyplot as plt
import csv

"""
-----------------------------------------------------------
Define the dataset root folder.
This is either provided as a command-line argument or defaults to "ms_T1_dataset"
Ex: python inspect_volumes.py ms_T2_dataset
-----------------------------------------------------------
"""
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ms_T1_dataset")

# Automatically determine scan type based on filename pattern or folder name
if "T2" in ROOT.name.upper():
    SCAN_TYPE = "T2"
else:
    SCAN_TYPE = "T1"

print(f"Scanning dataset: {ROOT} (scan type: {SCAN_TYPE})")

"""
-----------------------------------------------------------
Function: get_volume_info
Given a path to a .nii file, this loads the image and extracts:
    - The shape of the volume (x, y, z)
    - The voxel resolution (zooms, spacing between voxels in mm)
-----------------------------------------------------------
"""
def get_volume_info(p):
    try:
        img = nib.load(str(p))  # Try to load the image from the path provided in the argument
        img = nib.as_closest_canonical(img)  # Reorient to canonical (RAS+) orientation
        arr = img.get_fdata()  # Convert to a NumPy array (float data)

        # If the file has 4 dimensions (e.g., fMRI with time or multiple volumes),
        # take the first volume only and squeeze it down to 3D.
        if arr.ndim == 4:
            arr = np.squeeze(arr[..., 0])

        return arr.shape, img.header.get_zooms()  # Return (dimensions, voxel spacing)
    except Exception as e:
        # If something goes wrong (corrupt file, unreadable format, etc.)
        print(f"Error reading {p}: {e}")
        return None, None

"""
-----------------------------------------------------------
Main inspection loop
Walk through each class (ms/control) and collect information about each patient's scan.
Only one scan type is expected per dataset (T1 or T2).
-----------------------------------------------------------
"""
patients = []  # List of patients metadata dictionaries
z_counts = []  # List of z-axis depths (number of axial slices) for histogram

for cls in ["ms", "control"]:
    class_folder = ROOT / cls
    if not class_folder.exists():
        print(f"Missing folder: {class_folder}")
        continue  # Skip if this class folder doesn't exist in the dataset

    # Each patient folder looks like: patient_xx/
    for patient in sorted([p for p in class_folder.iterdir() if p.is_dir()]):
        # Look for scan file matching T1 or T2
        scan_files = list(patient.glob(f"*{SCAN_TYPE}*.nii"))

        if not scan_files:
            print(f"No {SCAN_TYPE} scan found for {patient}")
            continue

        scan_path = scan_files[0]
        shape, zoom = get_volume_info(scan_path)
        z = shape[2] if shape and len(shape) >= 3 else None

        # Build row with both T1 and T2 columns, but fill only the relevant one
        if SCAN_TYPE == "T1":
            row = {
                "patient": patient.name,
                "class": cls,
                "t1_shape": shape,
                "t1_zoom": zoom,
                "t2_shape": None,
                "t2_zoom": None,
                "z_depth": z
            }
        else:  # SCAN_TYPE == "T2"
            row = {
                "patient": patient.name,
                "class": cls,
                "t1_shape": None,
                "t1_zoom": None,
                "t2_shape": shape,
                "t2_zoom": zoom,
                "z_depth": z
            }

        patients.append(row)

        if z is not None:
            z_counts.append(z)

"""
-----------------------------------------------------------
Write all patient info into a CSV summary file
-----------------------------------------------------------
"""
outcsv = ROOT / "inspect_summary.csv"
with open(outcsv, "w", newline="") as f:
    keys = ["patient", "class", "t1_path", "t1_shape", "t1_zoom", "t2_shape", "t2_zoom", "z_depth"]
    writer = csv.DictWriter(f, keys)
    writer.writeheader()
    for r in patients:
        writer.writerow(r)

print("Wrote summary to", outcsv)

"""
-----------------------------------------------------------
Create a histogram of z-depths (axial slice count distribution)
This shows how many slices each scan has, useful for preprocessing decisions.
-----------------------------------------------------------
"""
if z_counts:
    import statistics
    print("Z-depth stats (axial slices):",
          "min", min(z_counts),
          "median", statistics.median(z_counts),
          "max", max(z_counts))

    plt.figure(figsize=(8, 3))
    plt.hist(z_counts, bins=20)
    plt.xlabel("Number of axial slices (z depth)")
    plt.ylabel("Number of volumes")
    plt.title(f"Z-depth distribution across {SCAN_TYPE} volumes")
    plt.tight_layout()

    hist_path = ROOT / "z_depth_histogram.png"
    plt.savefig(hist_path)
    print(f"Saved z-depth histogram: {hist_path}")
else:
    print("No z-depths found (possible reading error).")
