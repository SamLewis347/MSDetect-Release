"""
-----------------------------------------------------------
This script will execute the pipeline for splitting our dataset into training
and validation datasets. 

This file will take in a dataset folder in the following format via a user specified path:
    - input path can be adjusted via argparse arguments
dataset_root/
    control/
        patient_#/
            patient_#_MRI Modality (T1 or T2).nii
        ...
        ...
        ...
        patient_#/
            patient_#_MRI Modality (T1 or T2).nii
    ms/
        patient_#/
            patient_#_MRI Modality (T1 or T2).nii
        ...
        ...
        ...
        patient_#/
            patient_#_MRI Modality (T1 or T2).nii

This file will then:
(1) split the data into training and validation datasets
    - The training:validation ratio can be adjusted via argparse arguments

(2) save the split datasets to a user specified path
    - Output path can be adjusted via argparse arguments

(3) Run the preprocessing script on the newly split datasets
-----------------------------------------------------------
"""

"""
-----------------------------------------------------------
Library imports that will be used throughout the file
-----------------------------------------------------------
"""
import argparse
import os
import shutil
import random
from backend.utils.preprocess_mri_to_png import main as preprocess_main, parse_args as preprocess_parse_args # Importing the main() function of our preprocessing script so that we can run preprocessing after the dataset is split

"""
-----------------------------------------------------------
Argument parser:
    - Path to the root directory of the input (RAW) dataset
    - Path to the directory you want the split and preprocessed datasets root directories to be located
    - Percentage of the raw patient files that will go into the validation dataset (ex: 30 = 30% validation data)
-----------------------------------------------------------
"""
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str, help="Path to the root directory of the raw dataset.")
    p.add_argument("--output", required=True, type=str, help="Path to the directory where you want the root directory of the training and validation datasets to be saved to.")
    p.add_argument("--validation_percentage", type=float, default=0.3, help="The percentage of patients that you want to be placed in the validation dataset (default for this value is 30%).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--cleanup", action="store_true", help="If set, deletes raw split .nii data after preprocessing.")
    return p.parse_args()

"""
-----------------------------------------------------------
Function: split dataset
This function will take in the raw dataset from the input path specified in the argparser
and randomly select <valiation_percentage> of patients (split equally between MS and control) 
and place them into the validation dataset. The remaining patients will be placed inside the training dataset.
-----------------------------------------------------------
"""
def split_dataset(path_to_input_dataset, desired_output_dir, validation_percentage, seed):
    # Set the random seed for reproducibility
    random.seed(seed) # Unless otherwise specified this value will be 42

    # Labels array
    labels = ["control", "ms"]

    # Create the root directories for each datasets output
    training_root = os.path.join(desired_output_dir, "ms_training_dataset")
    validation_root = os.path.join(desired_output_dir, "ms_validation_dataset")

    # Create the root folder, "ms" and "control" subfolders for each dataset
    for root in [training_root, validation_root]:
        for label in labels:
            os.makedirs(os.path.join(root, label), exist_ok=True)

    # Loop over the labels array
    for label in labels:

        # Get the directory for the current label we are on (ms or control)
        label_directory = os.path.join(path_to_input_dataset, label)

        # Array to store the patients
        patients = []

        # Loop through each patient folder in this labels directory
        for patient in os.listdir(label_directory):

            # Append the patient folder to the directory of the current label to get the path to the current patient
            patient_path = os.path.join(label_directory, patient)

            # If the path to the current patient is a dir (it should be, but we want to explicitly ignore loose files just in case)
            if os.path.isdir(patient_path):

                # Append that patient to the patients array
                patients.append(patient_path)
        
        # Shuffle the patient list to ensire a random distribution
        random.shuffle(patients)

        # Calculate how many patients will go into the validation set
        total_number_of_patients = len(patients)
        number_to_go_into_validation = int(total_number_of_patients * validation_percentage)

        # Split the shuffled list into validation and training subsets
        validation_patients = patients[:number_to_go_into_validation]
        training_patients = patients[number_to_go_into_validation:]

        # Copy each training patient folder into the training directory under the correct label
        for patient in training_patients:
            destination = os.path.join(training_root, label, os.path.basename(patient))
            shutil.copytree(patient, destination)
        
        # Copy each valiation patient folder into the validation directory under the correct label
        for patient in validation_patients:
            destination = os.path.join(validation_root, label, os.path.basename(patient))
            shutil.copytree(patient, destination)

        # Print a summary for this class
        print(f"{label.upper()}: {len(training_patients)} train, {len(validation_patients)} val")

    # Print some completion messages
    print("\nSplit complete!")
    print(f"Training dataset saved to: {training_root}")
    print(f"Validation dataset saved to: {validation_root}")
    return training_root, validation_root

"""
-----------------------------------------------------------
Function: preprocess split datasets
This function will run the preprocessing script on the new split datasets for 
training and validation. Based on cleanup flag it will either delete the original
raw split datasets, or keep them
-----------------------------------------------------------
"""
def preprocess_split_datasets(training_root, validation_root, cleanup):
    splits = [("training", training_root), ("Validation", validation_root)]

    for split, split_directory in splits:
        print(f"\n Preprocessing {split} dataset...")

        # Output folder for processed data
        preprocessed_output_directory = os.path.join(os.path.dirname(split_directory), f"preprocessed_{split}")
        os.makedirs(preprocessed_output_directory, exist_ok=True)

        # Use default args from the preprocessing script
        args = preprocess_parse_args([
        "--input", split_directory,
        "--out", preprocessed_output_directory,
        "--patching",  # Force patching on
        "--modality", "T1"
        ])

        preprocess_main(args)
        print(f"Preprocessing complete for {split}: {preprocessed_output_directory}")

        if cleanup:
            print(f"Cleaning up raw NIfTI data: {split_directory}")
            shutil.rmtree(split_directory, ignore_errors=True)
            print(f"Deleted {split_directory}")
        
        print("\nAll splits preprocessed successfully!")


"""
-----------------------------------------------------------
Main execution
-----------------------------------------------------------
"""
if __name__ == "__main__":
    args = parse_args()

    # Step 1: Split dataset
    train_root, val_root = split_dataset(
        args.input,
        args.output,
        args.validation_percentage,
        args.seed
    )

    # Step 2: Preprocess splits
    preprocess_split_datasets(train_root, val_root, cleanup=args.cleanup)