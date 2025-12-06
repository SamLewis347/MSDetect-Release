import os
import argparse
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as imread
from matplotlib import cm
import pandas as pd
from PIL import Image
from multiprocessing import Pool
import numpy as np
import random
import h5py
from tqdm import tqdm
import keras
from keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import IPython.display

# get root directory of project
ROOT = os.getcwd()

"""
----------------
Argparser
----------------
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", required=True, help="name of directory of the preprocessed training files after project root folder 'Early-Detection-Of...'")
    parser.add_argument("--hypertune", action='store_true')
    parser.add_argument("--train", action='store_true', help='train a new model')
    parser.add_argument("--make_dataset", action='store_true', help='remake the dataset files')
    parser.add_argument("--batch_size", type=int , default=128, help="batch size used for training")
    parser.add_argument("--resume", action="store_true", help="decide whether to resume training from previous checkpoint")
    return parser.parse_args()

"""
--------------------------------------------
Function load_patch:
    utility function for create_data_stream
--------------------------------------------
"""
def load_patch(path):
    patch_path, label = path
    arr = np.asarray(Image.open(patch_path))
    return arr, label

"""
-----------------------------------------
Function: create_dataset_stream
    new dataset creation function that
    saves converted dataset to h5 files
-----------------------------------------
"""
def create_dataset_stream(base_dir, output_file, threads=8):
    # Estimate total patches
    print('getting all patch paths')
    patch_entries = []   # list of (patch_file_path, label)
    i = 0
    # walk through each patch in the indicated directory
    for label, class_name in enumerate(['control', 'ms']):
        class_dir = os.path.join(base_dir, class_name)
        for patient in os.listdir(class_dir):
            patient_path = os.path.join(class_dir, patient)
            for image in os.listdir(patient_path):
                img_path = os.path.join(patient_path, image)
                for patch in os.listdir(img_path):
                    patch_path = os.path.join(img_path, patch)
                    if patch_path.lower().endswith('.png'):
                        # append found path to array
                        patch_entries.append((patch_path, label))
                        i += 1

    total_patches = len(patch_entries)

    # create and open new file for save data
    """
    NOTE: lzf compression was used for write speeds at the cost of lower compression
          I deemed the inflation as negligible due to the training and val files
          combined still being around 45Gb for our whole dataset
    """
    with h5py.File(output_file, 'w') as f:
        x_ds = f.create_dataset('train_x', 
                                shape=(total_patches, 32, 32, 3),
                                dtype=np.uint8, 
                                compression='lzf', 
                                chunks=(256, 32, 32, 3))
        y_ds = f.create_dataset('train_y', 
                                shape=(total_patches,), 
                                dtype=np.int32, 
                                compression='lzf')

        # multithreaded loading of files to speed up runtime
        with Pool(processes=threads) as pool:
            for idx, (arr, label) in enumerate(
                tqdm(pool.imap(load_patch, patch_entries, chunksize=128),
                     total=total_patches,
                     desc="Loading patches")
            ):
                x_ds[idx] = arr
                y_ds[idx] = label

"""
-----------------------------------------
Function: data_gen
    file to create batches of training
    data for model from saved file
-----------------------------------------
"""
def data_gen(path, batch_size=32, shuffle=True):
    # open data file and gather metrics
    f = h5py.File(path, 'r')
    x_ds = f['train_x']
    y_ds = f['train_y']
    data_len = len(x_ds)

    # generator shuffles entries to be fed to model to model
    def gen():
        indices = np.arange(data_len)
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            yield x_ds[idx], y_ds[idx]

    # normalize outputs for data
    output_sig = (
        tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    # resulting dataset from generator
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)

    # shuffle and batch for model
    ds = ds.shuffle(4096) if shuffle else ds
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

"""
--------------------------------------------------------
Class BatchCheckpoint
    used to save weights mid-epoch for testing
    related purposes to save time when needing
    to retrain a model to test fixes.

    Saves to seperate weights file every n steps
    
    **recieved help from ChatGPT to get working and to 
    understand how a class could be used to make this work
--------------------------------------------------------
"""
class BatchCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, n=500):
        super().__init__()
        self.filepath = filepath
        self.n = n
        self.batch_counter = 0

    # check if weights need to be saved after each step
    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.batch_counter % self.n == 0:
            self.model.save_weights(self.filepath)
            print(f"\nSaved weights at batch {self.batch_counter}")

"""
-----------------------------------------
Function: model_builder
    main model builder
-----------------------------------------
"""
def model_builder(patch_size, resume=False):
    inputs = keras.Input((patch_size, patch_size, 3))

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same', name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, output)

    if resume:
        model.load_weights("backend/models/cp_mid.weights.h5")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    #set up checkpoints to save current weights
    checkpoint_path = "backend/models/cp.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            save_best_only=True,
                                            verbose=1)
    return model, checkpoint_path, callback

"""
-----------------------------------------
Function: train_model
    trains the model
-----------------------------------------
"""
def train_model(model, training_dataset, val_dataset, callback, batch_size):
    # open training file
    with h5py.File('training_patches.h5', 'r') as f:
        train_len = f['train_x'].shape[0]
    
    # track iterations per epoch for progress tracking
    steps = train_len // batch_size

    # mid-epoch weights
    batch_callback = BatchCheckpoint(filepath="backend/models/cp_mid.weights.h5",
                                 n=500)

    # model training
    history = model.fit(training_dataset,
                    steps_per_epoch=steps,
                    epochs=1,
                    validation_data=val_dataset,
                    callbacks=[callback, batch_callback],
                    )

    # creates the graph for the model's accuracy
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()
    history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.show()

"""
-----------------------------------------
Function: test_model
    tests model on single patients
    and outputs a heatmap per slice
    Stride is used to skip over some pixels to help runtime
        set to 1 if you want no skipping
-----------------------------------------
"""
def test_model(model, checkpoint_path, slice_path, patch_size=32, stride=8, display=True):
    print("loading weights...")
    model.load_weights(checkpoint_path)

    img = np.asarray(Image.open(slice_path).convert("RGB"))
    h, w, _ = img.shape

    patches = []
    coords = []

    # creates and stores values for each patch and its location in the image
    for row in range(0, h - patch_size + 1, stride):
        for col in range(0, w - patch_size + 1, stride):
            patch = img[row:row+patch_size, col:col+patch_size]
            patches.append(patch)
            coords.append((row, col))
    
    patches = np.array(patches, dtype=np.float32) / 255.0

    preds = []
    batch_size = 128

    # sends patches to the model for prediction, then saves them to new array
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        p = model.predict(batch, verbose=0).flatten()
        preds.extend(p)
    
    # initialize heatmap data
    heatmap_sum   = np.zeros((h, w), dtype=np.float32)
    heatmap_count = np.zeros((h, w), dtype=np.float32)

    # collect info on patch contribution for each pixel's activation
    for (row, col), p in zip(coords, preds):
        heatmap_sum[row:row+patch_size, col:col+patch_size] += p
        heatmap_count[row:row+patch_size, col:col+patch_size] += 1

    # average values to create smoother coloring for display
    heatmap = heatmap_sum / (heatmap_count + 1e-8)
    print("heatmap generated.")

    # normalize numbers
    hm = heatmap
    hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    heatmap_color = cm.jet(hm_norm)[..., :3]  # drop alpha channel

    # overlays heatmap onto original image
    overlay = (1 - 0.5) * img/255.0 + 0.5 * heatmap_color
    overlay = np.clip(overlay, 0, 1)


    original_image = Image.fromarray(img)
    overlayed_image = Image.fromarray(overlay)
    
    if display:
        # display original image next to overlay
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Original Slice")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Heatmap Overlay")
        plt.imshow(overlay)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return original_image, overlayed_image

def predict_patients_slices(model, checkpoint_path, slices_array, patch_size=32, stride=8, return_originals = False):
    """
    Run the MS inference on every slice of a preprocessed MRI volume.

    Args:
        model: Keras model
        checkpoint_path: path to saved weights
        slices_array: numpy array (N, H, H, 3) from the preprocess_single_file() function inside utils/preprocess_mri_to_png.py
        patch_size: size of each sliding patch
        stride: how far to move the patch each step
        return_originals: include original slices in output or just the predicted heatmaps

    Returns:
        List of dicts, one per slice
        [
            {
                "heatmap": np.ndarray (H, W, 3),
                "overlay": np.ndarray (H, W, 3),
                "raw_slice": optional
            }
        ] 
    """

    # Load the model weights before making predictions
    print("Loading model weights...")
    model.load_weights(checkpoint_path)

    results = []
    total_slices = slices_array.shape[0]

    # Iterate through each preprocessed slice
    for i in range(total_slices):
        print(f"[INFO] Preprocessing slice {i+1}/{total_slices}")

        # Confirm that slice is converted to uint8 (0-255)
        img = slices_array[i].astype(np.uint8) # Shape (224, 224, 3)
        h, w, _ = img.shape

        patches = [] # List of the patch image arrays
        coords = [] # Matching list of (row, col) positions

        # --- SLIDING WINDOW PATCH EXTRACTION ---
        # Slide a patch_sized window over the full slice
        # For every (row, col) location, extract a patch and remember where it was
        for row in range(0, h - patch_size + 1, stride):
            for col in range (0, w - patch_size + 1, stride):
                patch = img[row:row+patch_size, col:col+patch_size]
                patches.append(patch)
                coords.append((row, col))

        # Convert to float in [0, 1] range for the model
        patches = np.array(patches, dtype=np.float32) / 255.0

        # --- MODEL PREDICTIONS ---
        # The model predicts one value per patch (MS probability)
        predictions = []
        batch_size = 128

        # Predict in batches to avoid CPU/GPU memory issues
        for j in range (0, len(patches), batch_size):
            batch = patches[j:j+batch_size]
            prediction = model.predict(batch, verbose=0).flatten()
            predictions.extend(prediction)

        # --- BUILD HEATMAP BASED ON PREDICTIONS ---
        # Two 2D arrays:
        #   heatmap_sum: Accumulates probability scores
        #   heatmap_count: counts how many times each pixel was covered
        heatmap_sum = np.zeros((h, w), dtype=np.float32)
        heatmap_count = np.zeros((h, w), dtype=np.float32)

        # Each patch covers a specific region of the slice
        # We can "deposit" each probability into its own region
        for (row, col), p in zip(coords, predictions):
            heatmap_sum[row:row+patch_size, col:col+patch_size] += p
            heatmap_count[row:row+patch_size, col:col+patch_size] += 1
        
        # Avoid division by zero error
        heatmap = heatmap_sum / (heatmap_count + 1e-8)

        # Normalize heatmap to [0,1]
        hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Apply jet colormap (returns RGBA, keep RGB only)
        heatmap_color = cm.jet(hm_norm)[..., :3]

        # --- OVERLAY HEATMAP ON ORIGINAL SLICE ---
        # Convert original slice to [0,1], blend 50/50 with heatmap.
        overlay = (0.5 * (img / 255.0)) + (0.5 * heatmap_color)
        overlay = np.clip(overlay, 0, 1)

        # Convert overlay back to uint8 for display/frontend transmission
        result = {
            "heatmap": (heatmap_color * 255).astype(np.uint8),
            "overlay": (overlay * 255).astype(np.uint8)
        }

        # Optional: return raw slice for frontend or debugging
        if return_originals:
            result["raw_slice"] = img

        results.append(result)

    print("[INFO] Finished MS inference for patient.")
    return results

def main(args):
    batch_size = args.batch_size
    patients = args.patients

    #get directory of each data set
    base_dir = patients
    train_dir = os.path.join(base_dir, 'preprocessed_training')
    val_dir = os.path.join(base_dir, 'preprocessed_validation')

    # process data into usable arrays
    if args.make_dataset:
        create_dataset_stream(train_dir, 'training_patches.h5', threads=8)
        create_dataset_stream(val_dir, 'val_patches.h5', threads=8)

    # create datasets
    training = data_gen('training_patches.h5', batch_size=batch_size, shuffle=True)
    val = data_gen('val_patches.h5', batch_size=batch_size, shuffle=True)

    # build model
    model, checkpoint_path, callback = model_builder(32, args.resume)

    # train or test the model
    if args.train:
        train_model(model, training, val, callback, batch_size)
    else:
        print("[INFO] Model ready for inference")
        print("[INFO] Use app.py to run predictions on uploaded .nii files")

if __name__ == "__main__":
    args = parse_args()
    main(args)