"""
app.py
Backend API entry point for MRI model inference.
CRITICAL FIX: TensorFlow/Keras hangs in Gunicorn workers.
Solution: Load model AFTER fork (lazy loading per worker)
"""

import os
import sys
import tempfile
import base64
import io
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# Configure TensorFlow BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# CRITICAL: Prevent TensorFlow from using multiple threads on single CPU
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['KMP_AFFINITY'] = 'none'

# Import the preprocessing function
from utils.preprocess_mri_to_png import preprocess_single_file

# Import the model and prediction function
from models.patch_based_tensor import model_builder, predict_patients_slices

app = Flask(__name__)
CORS(app,resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "https://ms-detect.web.app",
            "https://ms-detect.firebaseapp.com"
        ]
    }
})

# Model configuration
backend_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_CHECKPOINT_PATH = os.path.join(backend_dir, "weights", "cp_mid.weights.h5")
PATCH_SIZE = 32

# CRITICAL: Global model variable that gets loaded LAZILY per worker
_model = None

def get_model():
    """
    Lazy load model per worker to avoid Gunicorn fork issues.
    TensorFlow/Keras doesn't work well when model is loaded before fork.
    """
    global _model
    
    if _model is None:
        print(f"[INFO] Worker PID {os.getpid()}: Loading model for the first time...", flush=True)
        sys.stdout.flush()
        
        # Verify model file exists
        if not os.path.exists(MODEL_CHECKPOINT_PATH):
            error_msg = f"Model weights not found at {MODEL_CHECKPOINT_PATH}"
            print(f"[ERROR] {error_msg}", flush=True)
            raise FileNotFoundError(error_msg)
        
        try:
            # Build model architecture
            print(f"[INFO] Worker {os.getpid()}: Building model architecture...", flush=True)
            _model, _, _ = model_builder(patch_size=PATCH_SIZE, resume=False)
            
            # Load weights
            print(f"[INFO] Worker {os.getpid()}: Loading weights from {MODEL_CHECKPOINT_PATH}...", flush=True)
            _model.load_weights(MODEL_CHECKPOINT_PATH)
            
            # CRITICAL: Warm up the model with a dummy prediction
            # This forces TensorFlow to compile the graph BEFORE handling real requests
            print(f"[INFO] Worker {os.getpid()}: Warming up model with dummy prediction...", flush=True)
            sys.stdout.flush()
            
            dummy_input = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
            _ = _model.predict(dummy_input, verbose=0)
            
            print(f"[SUCCESS] Worker {os.getpid()}: Model ready for inference!", flush=True)
            sys.stdout.flush()
            
        except Exception as e:
            print(f"[ERROR] Worker {os.getpid()}: Failed to load model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
    
    return _model


@app.route("/", methods=["GET"])
def home():
    """Sanity check to confirm backend is running"""
    return jsonify({
        "message": "Backend API is running!",
        "status": "ok",
        "worker_pid": os.getpid()
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts an uploaded MRI file (.nii or .nii.gz), runs inference,
    and returns heatmaps for each slice.
    """
    print(f"[DEBUG] Worker {os.getpid()}: /predict endpoint called", flush=True)
    sys.stdout.flush()

    if "file" not in request.files:
        print("[DEBUG] No file in request", flush=True)
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    print(f"[DEBUG] File received: {file.filename}", flush=True)

    # Validate file type
    if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
        print("[DEBUG] Invalid file type", flush=True)
        return jsonify({
            "error": "Invalid file type. Please upload a .nii or .nii.gz file"
        }), 400
    
    # Save to temporary location
    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name

    print(f"[DEBUG] File saved to: {temp_path}", flush=True)
    sys.stdout.flush()

    try:
        print(f"[INFO] Starting inference for {file.filename}...", flush=True)
        sys.stdout.flush()
        
        # Step 1: Get the model (lazy loads if needed)
        print("[INFO] Getting model instance...", flush=True)
        model = get_model()
        print("[INFO] Model instance acquired", flush=True)
        sys.stdout.flush()
        
        # Step 2: Preprocess the .nii file to get array of slices
        print("[INFO] Preprocessing MRI volume...", flush=True)
        sys.stdout.flush()
        
        slices_array = preprocess_single_file(
            file_path=temp_path,
            n_slices=20,
            use_25d=False,
            size=224,
            axis=2,
            use_all_slices=False
        )
        
        print(f"[INFO] Preprocessed {slices_array.shape[0]} slices", flush=True)
        print(f"[DEBUG] Slices array shape: {slices_array.shape}, dtype: {slices_array.dtype}", flush=True)
        sys.stdout.flush()
        
        # Step 3: Run inference on all slices
        print("[INFO] Running model inference...", flush=True)
        sys.stdout.flush()
        
        results = predict_patients_slices(
            model=model,
            checkpoint_path=MODEL_CHECKPOINT_PATH,
            slices_array=slices_array,
            patch_size=PATCH_SIZE,
            stride=8,
            return_originals=True,
            skip_load=True,  # Weights already loaded
        )

        print(f"[INFO] Inference returned {len(results)} results", flush=True)
        sys.stdout.flush()
        
        # Step 4: Convert both overlay and raw images to base64 for frontend
        print("[INFO] Encoding results for frontend...", flush=True)
        encoded_slices = []
        
        for i, result in enumerate(results):
            overlay_img = Image.fromarray(result["overlay"])
            overlay_buffer = io.BytesIO()
            overlay_img.save(overlay_buffer, format="PNG")
            overlay_buffer.seek(0)
            overlay_encoded = base64.b64encode(overlay_buffer.read()).decode("utf-8")
            
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
        
        print(f"[SUCCESS] Inference complete for {file.filename}", flush=True)
        sys.stdout.flush()
        
        return jsonify({
            "filename": file.filename,
            "slices": encoded_slices,
            "count": len(encoded_slices),
            "status": "success"
        })

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[INFO] Cleaned up temporary file", flush=True)


@app.route("/preview", methods=["POST"])
def preview():
    """Generate and return a scrollable axial preview of the uploaded MRI file."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
        return jsonify({
            "error": "Invalid file type. Please upload a .nii or .nii.gz file"
        }), 400

    with tempfile.TemporaryDirectory() as temp_dir:
        suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
        temp_path = os.path.join(temp_dir, f"uploaded{suffix}")
        file.save(temp_path)

        try:
            preprocess_single_file(
                temp_path,
                n_slices=20,
                use_25d=False,
                size=224,
                axis=2,
                out_dir=temp_dir,
                use_all_slices=False
            )

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
            print(f"[ERROR] Preview generation failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)