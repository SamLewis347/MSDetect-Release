"""
During the development of this project I used ChatGPT for high level guidance and small code examples, which I adapted and tested myself.
-----------------------------------------------------------
This script preprocesses MRI volumes (.nii files) into 2D PNG slices

This file supports both standard single channel axial slices
as well as "2.5D" preprocessing, where each slice is represented as 3 channels
(previous, current, next)

It also writes a manifest.csv file that records data such as:
    - Patient ID
    - Class label (MS = 1, control = 0)
    - Paths to slice folders
    - Number of total slices generated
    - Which modality (T1 or T2) was used

Usage:
    # Example for T1 dataset
    python preprocess_mri_to_pngs.py --input /path/to/ms_T1_dataset \
        --out /path/to/prepared_pngs/T1 \
        --n_slices 18 \
        --use_25d \
        --size 224 \
        --modality T1 \

    # Example for T2 dataset
    python preprocess_mri_to_pngs.py --input /path/to/ms_T1_dataset \
        --out /path/to/prepared_pngs/T1 \
        --n_slices 18 \
        --use_25d \
        --size 224 \
        --modality T2 \
-----------------------------------------------------------
"""

# Library imports that will be used throughout the file
import gc
import os
import argparse
import nibabel as nib # Library for handling medical images (in our case NifTI)
import numpy as np
import csv  # Library for writing the csv file
import scipy.ndimage
import traceback
import shutil
import datetime
from sklearn.feature_extraction.image import extract_patches_2d
from PIL import Image   # Library to save PNGs

"""
-----------------------------------------------------------
Argument parser:
Define configurable options such as:
    - dataset input folder
    - output folder
    - modality (T1 or T2) since datasets are separated
    - number of slices to extract
    - slice selection strategy
    - use of 2.5D vs single-channel
    - output image size
    - which axis to treat as axial (after canonical reorientation)
    - Whether or not to patch the data into smaller pieces for sliding window image model
    - Size of the patch window (ex: 32x32)
    - Threshold for when we patch as to if a patch is "interesting" (ex: a patch is interesting if 50% or more of the patch image is brain tissue)
-----------------------------------------------------------
"""
def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str, help="Un-pre-processed dataset root folder location")
    p.add_argument("--out", required=True, type=str, help="The folder that you want the dataset root folder to be created")
    p.add_argument("--modality", required=True, choices=("T1","T2"), help="Which modality is the un-pre-processed-dataset")
    p.add_argument("--n_slices", type=int, default=18, help="Number of axial slices to extract per volume")
    p.add_argument("--use_25d", action="store_true", help="Produce 2.5D (3-channel) stacks")
    p.add_argument("--size", type=int, default=224, help="Output image size (square), e.g. 224x224")
    p.add_argument("--axis", type=int, default=2, help="Axis to treat as axial AFTER canonicalization")
    p.add_argument("--patching", action="store_true", help="Select whether you want to further preprocess the resulting slices into patches")
    p.add_argument("--slider_size", type=int, default=32, help="Select the size of the sliding window that goes over the image data to split it into image segments")
    p.add_argument("--threshold", type=float, default=0.5, help="Select the threshold for the minimum amount of black space a patch image can have")
    return p.parse_args(args)

"""
-----------------------------------------------------------
Function: normalize_triplet_to_uint8
Converts a floating-point slice or 3-channel “triplet” to an
8-bit unsigned integer (0–255) image suitable for saving as PNG.
Steps:
    1. Clip intensity outliers (1st–99th percentile)
    2. Normalize values to 0–1 range
    3. Scale to 0–255 and cast to uint8
-----------------------------------------------------------
"""
def normalize_triplet_to_uint8(triplet):
    # Compute the intensity cutoff values at 1% and 99%
    low = np.percentile(triplet, 1)
    high = np.percentile(triplet, 99)

    # Clip all values outside this range
    arr = np.clip(triplet, low, high)

    # Shift so the minimum is 0
    arr = arr - arr.min()

    # Avoid the dividing by 0 case in case the slice is uniform
    if arr.max() > 0:
        arr = arr / arr.max()

    # Convert 0-1 floats into 0-255 integers
    arr = (arr * 255).astype(np.uint8)
    return arr

"""
-----------------------------------------------------------
Function: choose_indices
Given a start and end index (z_start, z_end), choose N
evenly spaced slice indices between them. This ensures
every volume contributes the same number of slices.
-----------------------------------------------------------
"""
def choose_indices(z_start, z_end, N):
    # Create N evenly spaced fractional positions between 0 and 1
    positions = np.linspace(0, 1, N)

    # Convert these to integer indices between start and end slice
    indicies = [int(round(z_start + p * (z_end - z_start))) for p in positions]
    return indicies
    
"""
-----------------------------------------------------------
Function: find_brain_bounds
Tries to detect which slices actually contain brain tissue.
Many MRI volumes have empty black slices at the top and bottom.
We look at slice intensity and ignore nearly empty ones.
threshold = fraction of max intensity used to detect boundaries.
-----------------------------------------------------------
"""
def find_brain_bounds(arr, axis=2, threshold=0.05):
    z = arr.shape[axis]
    proj = []

    # Loop through all slices and record each slice's maximum pixel intensity
    for i in range(z):
        if axis == 2:
            proj.append(arr[:, :, i].max())
        elif axis == 1:
            proj.append(arr[:, i, :].max())
        else:
            proj.append(arr[i, :, :].max())

    proj = np.array(proj)
    max_val = proj.max()

    # Find all slices where the max intensity > threshold x overall max intensity
    meaningful = np.where(proj > max_val*threshold)[0]

    # If no meaningful slices found (shouldn't happen), just use the entire range
    if len(meaningful) == 0:
        return 0, z-1
    
    # Return the first and last meaningful slice incidies
    return meaningful[0], meaningful[-1]

"""
-----------------------------------------------------------
Function: load_volume_get_array
Loads a NIfTI MRI, canonicalizes to RAS+, standardizes
the visual orientation, resamples to 1mm isotropic spacing,
and normalizes intensities.
Returns: float32 3D NumPy array (H, W, Z)
-----------------------------------------------------------
"""
def load_volume_get_array(nifti_path):
    # --- Step 1. Load the NIfTI volume ---
    img = nib.load(str(nifti_path))
    orig_axcodes = nib.aff2axcodes(img.affine)
    print(f"[INFO] Original orientation for {os.path.basename(nifti_path)}: {orig_axcodes}")

    # --- Step 2. Canonicalize to RAS+ ---
    img = nib.as_closest_canonical(img)

    # --- Step 3. Extract float32 array ---
    arr = img.get_fdata(dtype=np.float32)

    # --- Step 4. Handle 4D (take first volume only) ---
    if arr.ndim == 4:
        arr = np.squeeze(arr[..., 0])
    if arr.ndim != 3:
        raise ValueError(f"{os.path.basename(nifti_path)} is not a 3D MRI volume.")

    # --- Step 5. Resample to 1×1×1 mm if needed ---
    zooms = img.header.get_zooms()[:3]
    target_zooms = (1.0, 1.0, 1.0)
    if not np.allclose(zooms, target_zooms, atol=1e-3):
        scale_factors = np.array(zooms) / np.array(target_zooms)
        arr = scipy.ndimage.zoom(arr, zoom=scale_factors, order=1)

    # --- Step 6. Normalize intensities ---
    arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
    low, high = np.percentile(arr, (1, 99))
    arr = np.clip(arr, low, high)

    return arr

"""
-----------------------------------------------------------
Function: process_and_save_volume
Converts a single MRI volume into multiple PNG slices.
Steps:
    1. Load and normalize MRI
    2. Find z-range containing brain
    3. Pick N evenly spaced slices
    4. Create 2.5D triplets or single-channel stacks
    5. Normalize + resize
    6. Save as slice_01.png, slice_02.png, ...
Returns: number of slices saved
-----------------------------------------------------------
"""
def process_and_save_volume(nifti_path, out_dir, n_slices, use_25d, size, axis):
    # Load and preprocess the 3D MRI array
    arr = load_volume_get_array(nifti_path)
    
    # Find approximate bounds where brain tissue exists
    z_start, z_end = find_brain_bounds(arr, axis)

    # Number of slices along the chosen axis (usually z)
    z = arr.shape[axis]

    # Choose slice indices evenly spaced between bounds
    inds = choose_indices(z_start, z_end, n_slices)

    # Create the output folder if it doesn’t exist
    os.makedirs(out_dir, exist_ok=True)
    count = 0 # Counter for how many slices are saved

     # Loop through each selected slice index
    for i, idx in enumerate(inds, start=1):
         # For 2.5D: pick previous, current, and next slices (edge protected)
        i0 = max(0, idx - 1)
        i1 = idx
        i2 = min(z - 1, idx + 1)

        # Extract 2D slices along the chosen axis
        if axis == 2:
            s0, s1, s2 = arr[:, :, i0], arr[:, :, i1], arr[:, :, i2]
        elif axis == 1:
            s0, s1, s2 = arr[:, i0, :], arr[:, i1, :], arr[:, i2, :]
        else:
            s0, s1, s2 = arr[i0, :, :], arr[i1, :, :], arr[i2, :, :]

        # Combine slices into a 3-channel image
        if use_25d:
            # Each channel = [previous, current, next slice]
            triplet = np.stack([s0, s1, s2], axis=-1)
        else:
            # Replicate the same slice into all three channels (grayscale)
            triplet = np.stack([s1, s1, s1], axis=-1)

        # Convert to uint8 0–255 scale for saving
        trip_u8 = normalize_triplet_to_uint8(triplet)

        # Convert NumPy array → PIL Image and resize to target dimensions
        img = Image.fromarray(trip_u8)
        img = img.resize((size, size), Image.LANCZOS)

        # Save PNG file into patient folder
        fname = os.path.join(out_dir, f"slice_{i:02d}.png")
        img.save(fname)
        count += 1

    return count

"""
-----------------------------------------------------------
Function: image_splitter
    splits original image into multiple smaller images of 
    size <image_size> to increase the size of the training
    dataset
-----------------------------------------------------------
"""

def image_splitter(out_patient, window_size, threshold):
    for i, file in enumerate(sorted(os.listdir(out_patient))):
        # create patch dir for current slice
        slice_folder = os.path.join(out_patient, f"Slice_{i}")
        os.makedirs(slice_folder, exist_ok=True)

        # get the patch to the current slice
        slc_path = os.path.join(out_patient, file)

        # Load image and turn that image into an array
        with Image.open(slc_path) as slc: # Using with will cause the file to be closed immediately after reading which helps performance
            slc = np.array(slc)

        # creates a new array of patches with shape (patch#, h, w, ch)
        patches = extract_patches_2d(slc, (window_size, window_size))

        # save patches to slice dir
        for j, patch in enumerate(patches):
            if (image_evaluation(patch, 20, threshold, 40)):
                Image.fromarray(patch).save(os.path.join(slice_folder, f"patch_{j}.png"))

        # Remove the original slice after sucessful patch creation
        os.remove(slc_path)
        
"""
-----------------------------------------------------------
Function:
    Looks at <image> and decides if that image contains enough
    of the brain in the image (non blackspace) for it to be
    "interesting" enough to be sample data. This will be done 
    using the <threshold> and if there is more % brain content
    in the image than the <threshold>
-----------------------------------------------------------
"""
def image_evaluation(patch, intensity_threshold, min_brain_percentage, var_threshold):
    # Fracton of pixels above the threshold
    brain_percentage = np.sum(patch > intensity_threshold) / patch.size

    # Local intensity variance (helps to reject flat background or uniform noise)
    var = np.var(patch)

    return brain_percentage > min_brain_percentage and var > var_threshold

"""
-----------------------------------------------------------
Function: preprocess_single_file
Lightweight version of the preprocessing pipeline for inference.
    Works on ONE .nii file and returns a NumPy array of preprocessed slices.
    Called by app.py for the testing pipeline

    Args:
        file_path (str or Path): path to a single .nii or .nii.gz file
        n_slices (int): number of slices to extract
        use_25d (bool): whether to use 2.5D (3-channel) triplets
        size (int): target slice size (square)
        axis (int): axis to treat as axial (2 = default)
    
    Returns:
        np.ndarray: array of shape (n_slices, size, size, 3)
-----------------------------------------------------------
"""
def preprocess_single_file(file_path, n_slices=20, use_25d=False, size=224, axis=2, out_dir=None, use_all_slices=False):
    # Step 1: Load and preprocess the MRI volume as a numpy array
    arr = load_volume_get_array(file_path)

    # Step 2: Remove the slices where there is little to no tissue in the scan
    z_start, z_end = find_brain_bounds(arr, axis)
    z = arr.shape[axis]

    # Either use ALL slices in brain bounds OR sample n evenly spaced slices
    if use_all_slices:
        indices = list(range(z_start, z_end + 1)) # Include every slice
        print(f"[INFO] Using all {len(indices)} slices between bounds [{z_start}, {z_end}]")
    else:
        indices = choose_indices(z_start, z_end, n_slices)
        print(f"[INFO] Sampling {n_slices} evenly spaced slices")

    # Step 4: If an output directory is specified, create it (this would be only for debugging or visualization)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    slices_out = []

    # Step 5: Iterate over each selected index of the brain scan
    for i, index in enumerate(indices): # If we use 2.5D we get the previous, current, and next slice to make a "2.5D triplet"

        # Compute neighbor slice indicies with boundary protection
        i0 = max(0, index - 1) # Previous slice
        i1 = index # Current slice
        i2 = min(z - 1, index + 1) # Next slice

        # Extract slice(s) along the chosen axis
        if axis == 2:
            s0, s1, s2 = arr[:, :, i0], arr[:, :, i1], arr[:, :, i2]
        elif axis == 1:
            s0, s1, s2 = arr[:, i0, :], arr[:, i1, :], arr[:, i2, :]
        else: # Axis = 0
            s0, s1, s2 = arr[i0, :, :], arr[i1, :, :]

        # Create triplet or grayscale stack
        if use_25d:
            triplet = np.stack([s0, s1, s2], axis=-1)
        else:
            triplet = np.stack([s1, s1, s1], axis=-1)

        # Step 7: Normalize from raw MRI intensities -> uint8 (0-255)
        trip_u8 = normalize_triplet_to_uint8(triplet)
        img = Image.fromarray(trip_u8) # Convert to PIL image for resizing

        # Step 8: Resize slice to model's fixed input resolution (ex: 224 x 224)
        img = img.resize((size, size), Image.LANCZOS)

        # Step 9: Either save slice as PNG (for debugging) OR store it in-memory to retur as NumPy arrays
        if out_dir is not None:
            img.save(os.path.join(out_dir, f"slice_{i:03d}.png"))
        else:
            # Store as float32 array for neural network input
            slices_out.append(np.asarray(img, dtype=np.float32))

    # Step 10: Return all collected slices (if not saving to a disk) (shape: n_slices, size, size, 3)
    if out_dir is None:
        return np.stack(slices_out, axis=0)
    else:
        return None

"""
-----------------------------------------------------------
Function: main
Top-level function that coordinates the whole pipeline.
It loops through every patient, processes each scan,
and writes all results to a manifest.csv file.
-----------------------------------------------------------
"""

def main(args):
    root = args.input   # Root dataset directory (contains “control” and “ms”)
    out_root = args.out # Output directory for PNGs + manifest
    rows = []           # Will store metadata for CSV
    iterations = 0
    try:
        # Iterate through both possible class labels
        for cls in ["control", "ms"]:
            class_in = os.path.join(root, cls)
            if not os.path.exists(class_in):
                continue  # Skip if class folder missing

            # Each patient folder inside contains one MRI (.nii) file
            for pid in sorted(os.listdir(class_in)):
                iterations += 1
                patient_dir = os.path.join(class_in, pid)
                if not os.path.isdir(patient_dir):
                    continue

                nii_files = [f for f in os.listdir(patient_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
                if not nii_files:
                    print(f"[WARNING] No NIfTI file for {patient_dir}, skipping")
                    continue

                nii_file = os.path.join(patient_dir, nii_files[0])
                out_patient = os.path.join(out_root, cls, pid)  # Where to save this patient’s slices
                os.makedirs(out_patient, exist_ok=True)

                # Process MRI and save PNGs
                n_written = process_and_save_volume(
                    nii_file, out_patient, args.n_slices, args.use_25d, args.size, args.axis
                )

                # Split saved slices into smaller image patches if the args.patching is true
                if args.patching:
                    image_splitter(out_patient, args.slider_size, args.threshold)

                # Store one row of metadata for this patient
                rows.append({
                    "patient_id": pid,
                    "label": 1 if cls == "ms" else 0,
                    "modality": args.modality,
                    "slice_dir": os.path.abspath(out_patient),
                    "n_slices": n_written
                })

                print(f"Wrote patient {pid} ({cls}, {args.modality}): slices={n_written}")
                if iterations % 10 == 0:
                    gc.collect()
                    

        # ---------- Write manifest.csv ----------
        os.makedirs(out_root, exist_ok=True)
        manifest_path = os.path.join(out_root, "manifest.csv")
        with open(manifest_path, "w", newline="") as f:
            fieldnames = ["patient_id", "label", "modality", "slice_dir", "n_slices"]
            writer = csv.DictWriter(f, fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print("Done. Prepared PNGs + manifest at:", out_root)
    
    except Exception as e:
        print("\n[ERROR] Preprocessing failed due to an unexpected error:")
        print(traceback.format_exc())

        # Cleanup partially written output
        if os.path.exists(out_root):
            print(f"[CLEANUP] Removing incomplete output directory: {out_root}")
            shutil.rmtree(out_root, ignore_errors=True)

        # Re-raise to make sure CLI sees the failure
        raise e

"""
-----------------------------------------------------------
Program entry point
When the script is run directly (not imported), this block
executes. It reads arguments from command line and launches
the main() preprocessing pipeline.
-----------------------------------------------------------
"""
if __name__ == "__main__":
    args = parse_args()
    
    # Create a timestamped dataset root folder automatically
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"preprocessed_{args.modality}_dataset"
    dataset_root = os.path.join(args.out, dataset_name)

    try:
        os.makedirs(dataset_root, exist_ok=False)
    except FileExistsError:
        print(f"[WARNING] Folder {dataset_root} already exists — using unique fallback name.")
        dataset_root += "_dup"
        os.makedirs(dataset_root, exist_ok=True)

    # Update args.out so that the rest of the pipeline writes into that folder
    args.out = dataset_root

    print(f"[INFO] Created dataset root folder: {dataset_root}")

    try:
        main(args)
    except Exception as e:
        print("[ERROR] Pipeline failed — cleaning up dataset root folder.")
        if os.path.exists(dataset_root):
            shutil.rmtree(dataset_root, ignore_errors=True)
        raise e
