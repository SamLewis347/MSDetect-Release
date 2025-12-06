"""
app.py
Backend API entry point for MRI model inference.

Routes:
    GET / - Sanity check route to confirm that the API is functional
    POST /predict - Accepts an MRI volume (.nii) file and returns MS prediction with heatmaps
    POST /preview - Generate scrollable axial preview of uploaded MRI
"""

import os
import tempfile
import base64
import io
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# Import the preprocessing function
from utils.preprocess_mri_to_png import preprocess_single_file

# Import the model and prediction function
from models.patch_based_tensor import model_builder, predict_patients_slices

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Model configuration
# Construct path explicitly to avoid Windows path issues
backend_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_CHECKPOINT_PATH = os.path.join(backend_dir, "weights", "cp_mid.weights.h5")

# Print for debugging
print(f"[DEBUG] Looking for model at: {MODEL_CHECKPOINT_PATH}")
if not os.path.exists(MODEL_CHECKPOINT_PATH):
    print(f"[WARNING] Model file not found at {MODEL_CHECKPOINT_PATH}")
    print(f"[INFO] Please ensure the weights file exists at this location")

PATCH_SIZE = 32

# Load model once at startup
print("[INFO] Loading model at startup...")
model, _, _ = model_builder(patch_size=PATCH_SIZE, resume=False)
print("[INFO] Model loaded successfully!")


@app.route("/", methods=["GET"])
def home():
    """Sanity check to confirm backend is running"""
    return jsonify({
        "message": "Backend API is running!",
        "status": "ok"
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts an uploaded MRI file (.nii or .nii.gz), runs inference,
    and returns heatmaps for each slice.
    
    Returns:
        JSON with:
        - filename: original filename
        - slices: list of objects with base64-encoded overlay and raw images
        - count: number of slices processed
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]

    # Validate file type
    if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
        return jsonify({
            "error": "Invalid file type. Please upload a .nii or .nii.gz file"
        }), 400
    
    # Save to temporary location
    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        print(f"[INFO] Starting inference for {file.filename}...")
        
        # Step 1: Preprocess the .nii file to get array of slices
        print("[INFO] Preprocessing MRI volume...")
        slices_array = preprocess_single_file(
            file_path=temp_path,
            n_slices=20,
            use_25d=False,
            size=224,
            axis=2,
            use_all_slices=False
        )
        
        print(f"[INFO] Preprocessed {slices_array.shape[0]} slices")
        
        # Step 2: Run inference on all slices
        print("[INFO] Running model inference...")
        results = predict_patients_slices(
            model=model,
            checkpoint_path=MODEL_CHECKPOINT_PATH,
            slices_array=slices_array,
            patch_size=PATCH_SIZE,
            stride=8,
            return_originals=True  # Include raw slices for frontend toggle
        )
        
        # Step 3: Convert both overlay and raw images to base64 for frontend
        print("[INFO] Encoding results for frontend...")
        encoded_slices = []
        
        for i, result in enumerate(results):
            # Convert overlay numpy array to PIL Image and encode
            overlay_img = Image.fromarray(result["overlay"])
            overlay_buffer = io.BytesIO()
            overlay_img.save(overlay_buffer, format="PNG")
            overlay_buffer.seek(0)
            overlay_encoded = base64.b64encode(overlay_buffer.read()).decode("utf-8")
            
            # Convert raw slice numpy array to PIL Image and encode
            raw_img = Image.fromarray(result["raw_slice"])
            raw_buffer = io.BytesIO()
            raw_img.save(raw_buffer, format="PNG")
            raw_buffer.seek(0)
            raw_encoded = base64.b64encode(raw_buffer.read()).decode("utf-8")
            
            encoded_slices.append({
                "slice_index": i,
                "overlay": f"data:image/png;base64,{overlay_encoded}",
                "raw": f"data:image/png;base64,{raw_encoded}"
            })
        
        print(f"[INFO] Inference complete for {file.filename}")
        
        return jsonify({
            "filename": file.filename,
            "slices": encoded_slices,
            "count": len(encoded_slices),
            "status": "success"
        })

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[INFO] Cleaned up temporary file")


@app.route("/preview", methods=["POST"])
def preview():
    """Generate and return a scrollable axial preview of the uploaded MRI file."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Check extension
    if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
        return jsonify({
            "error": "Invalid file type. Please upload a .nii or .nii.gz file"
        }), 400

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
        temp_path = os.path.join(temp_dir, f"uploaded{suffix}")
        file.save(temp_path)

        try:
            # Use existing preprocessing to save axial slices as PNGs
            preprocess_single_file(
                temp_path,
                n_slices=20,
                use_25d=False,
                size=224,
                axis=2,
                out_dir=temp_dir,
                use_all_slices=False
            )

            # Collect all PNGs and encode them to base64
            png_files = sorted(Path(temp_dir).glob("*.png"))
            preview_images = []
            for png_path in png_files:
                with open(png_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                    preview_images.append(f"data:image/png;base64,{encoded}")

            return jsonify({
                "slices": preview_images, 
                "count": len(preview_images),
                "status": "success"
            })

        except Exception as e:
            print(f"[ERROR] Preview generation failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)