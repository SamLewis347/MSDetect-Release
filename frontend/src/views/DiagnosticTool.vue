<template>
  <div class="relative min-h-screen text-slate-100 flex flex-col items-center py-8 sm:py-12 lg:py-16 px-4 sm:px-6">
    <!-- Header -->
    <div class="pt-6 mt-6 sm:mt-10 w-full max-w-7xl mb-6 sm:mb-10 flex justify-between items-center">
      <div>
        <h1 class="text-2xl sm:text-3xl lg:text-4xl font-extrabold"><span class="text-white">MS</span><span class="text-sky-400">Detect</span> Portal</h1>
        <p class="text-slate-400 text-xs sm:text-sm mt-1">Upload and analyze MRI scans using AI-assisted segmentation.</p>
      </div>
    </div>

    <!-- Unified Workspace -->
    <div class="w-full max-w-7xl bg-slate-950/80 border border-white/10 rounded-2xl sm:rounded-3xl shadow-2xl overflow-hidden flex flex-col lg:flex-row">
      <!-- Upload Section -->
      <div class="lg:w-1/3 p-4 sm:p-6 lg:p-8 border-b lg:border-b-0 lg:border-r border-white/10 flex flex-col justify-between">
        <div>
          <h2 class="text-xl sm:text-2xl font-semibold mb-2 text-white">Upload MRI</h2>
          <p class="text-slate-400 mb-4 sm:mb-5 text-xs sm:text-sm">Click or drag your MRI (.nii / .nii.gz) here for AI preview and prediction.</p>

          <div
            @click="triggerFileInput"
            @dragover.prevent="isDragging = true"
            @dragleave.prevent="isDragging = false"
            @drop.prevent="handleDrop"
            :class="[
              'rounded-xl p-4 sm:p-6 w-full flex flex-col items-center justify-center text-center cursor-pointer transition-all border-2 border-dashed min-h-[120px] sm:min-h-[140px]',
              isDragging ? 'ring-4 ring-sky-400/30 border-sky-400 bg-slate-800/30' : 'border-sky-500/20 bg-slate-900/40 hover:border-sky-400/30',
              (loadingPreview || loading) ? 'pointer-events-none opacity-60' : ''
            ]"
          >
            <input ref="fileInput" type="file" accept=".nii,.nii.gz" class="hidden" @change="handleFileChange" />
            <i class="fa-solid fa-cloud-arrow-up text-sky-400 text-3xl sm:text-4xl mb-2 sm:mb-3"></i>
            <p class="text-white font-medium text-sm sm:text-base">Click or drag to upload</p>
            <p class="text-slate-400 text-xs mt-1">Accepted: .nii, .nii.gz</p>
          </div>

          <!-- File Info -->
          <div v-if="fileName" class="mt-4 text-xs sm:text-sm text-slate-300">
            <div class="truncate">File: <span class="text-sky-300">{{ fileName }}</span></div>
          </div>

          <!-- Preview Loading State -->
          <div v-if="loadingPreview" class="mt-4 flex items-center gap-2 text-xs sm:text-sm text-sky-400">
            <div class="w-4 h-4 border-2 border-sky-400 border-t-transparent rounded-full animate-spin"></div>
            <span>Generating preview...</span>
          </div>
        </div>

        <!-- Actions -->
        <div v-if="fileName && !loadingPreview" class="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 mt-6">
          <button
            class="bg-gradient-to-r from-sky-600 to-sky-500 px-4 py-2.5 sm:py-2 rounded-md text-white text-sm sm:text-base hover:scale-[1.02] transition-all shadow disabled:opacity-50 disabled:cursor-not-allowed"
            @click="submitFile"
            :disabled="loading"
          >
            {{ loading ? "Analyzing..." : "Upload & Predict" }}
          </button>
          <button
            class="px-3 py-2.5 sm:py-2 rounded-md border border-white/10 text-sm text-slate-200 hover:bg-white/5 disabled:opacity-50 disabled:cursor-not-allowed"
            @click="resetSelection"
            :disabled="loading"
          >
            Reset
          </button>
        </div>
      </div>

      <!-- Viewer Section -->
      <div class="lg:w-2/3 p-4 sm:p-6 lg:p-8 relative flex flex-col min-h-[500px] sm:min-h-[600px]">
        <!-- Bezel Header -->
        <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-2 sm:gap-0">
          <!-- Status Lights -->
          <div class="flex items-center gap-2">
            <div
              class="w-3.5 h-3.5 rounded-full shadow transition-all duration-500 animate-[pulseLight_2s_ease-in-out_infinite]"
              :class="{
                'bg-rose-500/70 shadow-[0_0_6px_2px_rgba(239,68,68,0.4)]':
                  !fileName && !previewSlices.length && !predictionSlices.length,
                'bg-amber-400/70 shadow-[0_0_6px_2px_rgba(250,204,21,0.4)]':
                  (fileName && previewSlices.length && !predictionSlices.length) || loadingPreview || loading,
                'bg-emerald-400/70 shadow-[0_0_6px_2px_rgba(34,197,94,0.4)]':
                  predictionSlices.length > 0 && !loading
              }"
            ></div>
            <span class="text-xs text-slate-400 ml-1">
              {{
                loadingPreview
                  ? 'Loading Preview'
                  : loading
                  ? 'Analyzing...'
                  : !fileName && !previewSlices.length
                  ? 'Awaiting Upload'
                  : fileName && previewSlices.length && !predictionSlices.length
                  ? 'Ready for Analysis'
                  : 'Analysis Complete'
              }}
            </span>
          </div>

          <!-- Right-side info -->
          <div class="text-xs text-slate-400 truncate max-w-full">
            <span class="hidden sm:inline">Patient: <span class="text-white">Demo</span> • Study: </span>
            <span class="text-white truncate">{{ fileName || "—" }}</span>
          </div>
        </div>


        <!-- MRI Viewer -->
        <div
          class="relative flex-grow border border-white/10 rounded-2xl bg-slate-950 flex items-center justify-center overflow-hidden shadow-inner min-h-[300px] sm:min-h-[380px]"
          @wheel.prevent="handleScroll"
        >
          <!-- Loading States -->
          <div v-if="loadingPreview" class="flex flex-col items-center gap-3 sm:gap-4 px-4">
            <div class="w-12 h-12 sm:w-16 sm:h-16 border-4 border-sky-400 border-t-transparent rounded-full animate-spin"></div>
            <div class="text-slate-400 text-xs sm:text-sm text-center">Processing MRI volume...</div>
            <div class="text-slate-500 text-xs text-center">Extracting axial slices</div>
          </div>

          <div v-else-if="loading" class="flex flex-col items-center gap-3 sm:gap-4 px-4">
            <div class="w-12 h-12 sm:w-16 sm:h-16 border-4 border-sky-400 border-t-transparent rounded-full animate-spin"></div>
            <div class="text-slate-400 text-xs sm:text-sm text-center">Running AI analysis...</div>
            <div class="text-slate-500 text-xs text-center">This may take a moment</div>
            <!-- Progress bar -->
            <div class="w-48 sm:w-64 h-2 bg-slate-800 rounded-full overflow-hidden mt-2">
              <div class="h-full bg-gradient-to-r from-sky-600 to-sky-400 animate-[progress_2s_ease-in-out_infinite]"></div>
            </div>
          </div>

          <!-- Show prediction results if available, otherwise show preview -->
          <img
            v-else-if="currentDisplayImage"
            :src="currentDisplayImage"
            class="object-contain max-h-[60vh] sm:max-h-[75vh] scale-125 sm:scale-[1.5] transition-transform duration-300 ease-out"
          />
          <div v-else class="text-slate-500/70 text-xs sm:text-sm px-4 text-center">No scan loaded — upload to preview.</div>

          <!-- Label -->
          <div v-if="currentDisplayImage && !loadingPreview && !loading" class="absolute top-3 left-3 text-xs bg-black/50 text-slate-200 px-2 py-1 rounded-md">
            Axial
          </div>
        </div>

        <!-- Viewer Controls -->
        <div class="flex flex-col sm:flex-row items-stretch sm:items-center justify-between mt-4 gap-3 sm:gap-0">
          <div class="flex items-center gap-2 sm:gap-3 justify-center sm:justify-start">
            <button
              class="flex-1 sm:flex-none px-3 py-2 bg-slate-800/60 border border-white/10 rounded-md text-sm hover:bg-slate-700/60 disabled:opacity-50 disabled:cursor-not-allowed"
              @click="currentSlice = Math.max(0, currentSlice - 1)"
              :disabled="totalSlices === 0 || loading || loadingPreview"
            >
              <i class="fa-solid fa-arrow-left"></i>
            </button>
            <button
              class="flex-1 sm:flex-none px-3 py-2 bg-slate-800/60 border border-white/10 rounded-md text-sm hover:bg-slate-700/60 disabled:opacity-50 disabled:cursor-not-allowed"
              @click="currentSlice = Math.min(totalSlices - 1, currentSlice + 1)"
              :disabled="totalSlices === 0 || loading || loadingPreview"
            >
              <i class="fa-solid fa-arrow-right"></i>
            </button>
            <button
              class="flex-1 sm:flex-none px-3 py-2 bg-sky-600/80 text-white rounded-md text-xs sm:text-sm shadow hover:bg-sky-500/80 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
              @click="toggleHeatmap"
              :disabled="!predictionSlices.length || loading"
            >
              {{ showHeatmap ? "Hide" : "Show" }} Heatmap
            </button>
          </div>
          <div class="text-xs text-slate-400 text-center sm:text-left">
            Slice <span class="text-white">{{ currentSlice + 1 }}</span> / <span class="text-white">{{ totalSlices || "—" }}</span>
          </div>
        </div>

        <!-- Findings -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4 sm:mt-6">
          <div class="col-span-1 md:col-span-2 bg-slate-900/60 border border-white/10 rounded-lg p-3">
            <div class="text-xs text-slate-400">AI Findings</div>
            <div v-if="loading" class="mt-2">
              <div class="text-base sm:text-lg font-bold text-amber-400">Processing...</div>
              <div class="text-xs sm:text-sm text-slate-300 mt-1">
                Running inference on MRI volume
              </div>
            </div>
            <div v-else-if="predictionSlices.length > 0" class="mt-2">
              <div class="text-base sm:text-lg font-bold text-sky-300">Analysis Complete</div>
              <div class="text-xs sm:text-sm text-slate-300 mt-1">
                Processed <span class="font-semibold">{{ predictionSlices.length }}</span> slices with heatmap overlays
              </div>
              <p class="text-xs text-slate-400 mt-2">Toggle "Show Heatmap" to view AI-detected regions. This is a model output — confirm with a clinician.</p>
            </div>
            <div v-else class="text-xs sm:text-sm text-slate-400 mt-2">No prediction yet — upload & run analysis.</div>
          </div>

          <div class="bg-slate-900/60 border border-white/10 rounded-lg p-3">
            <div class="text-xs text-slate-400">Scan Info</div>
            <div class="mt-2 text-xs sm:text-sm text-slate-200">
              <div class="truncate">Filename: <span class="text-slate-300">{{ fileName || "—" }}</span></div>
              <div class="mt-1">Slices: <span class="text-slate-300">{{ totalSlices || "—" }}</span></div>
              <div v-if="predictionSlices.length" class="mt-1">Status: <span class="text-emerald-400">Analyzed</span></div>
              <div v-else-if="loading" class="mt-1">Status: <span class="text-amber-400">Analyzing</span></div>
              <div v-else-if="loadingPreview" class="mt-1">Status: <span class="text-amber-400">Loading</span></div>
            </div>
          </div>
        </div>

        <div class="text-xs text-slate-500 mt-3 border-t border-white/5 pt-3 text-center sm:text-left">
          Powered by Flask + TensorFlow • Demo only
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from "vue";
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5001";

const fileInput = ref(null);
const selectedFile = ref(null);
const fileName = ref("");
const isDragging = ref(false);
const loading = ref(false);
const loadingPreview = ref(false);

// Preview slices (from /preview endpoint)
const previewSlices = ref([]);

// Prediction slices (from /predict endpoint) - array of {raw, overlay, slice_index}
const predictionSlices = ref([]);

const currentSlice = ref(0);
const showHeatmap = ref(false);

// Computed property to determine which image to display
const currentDisplayImage = computed(() => {
  // If we have prediction results, show those (either raw or overlay based on toggle)
  if (predictionSlices.value.length > 0) {
    const slice = predictionSlices.value[currentSlice.value];
    if (!slice) return null;
    return showHeatmap.value ? slice.overlay : slice.raw;
  }

  // Otherwise show preview slices
  if (previewSlices.value.length > 0) {
    return previewSlices.value[currentSlice.value];
  }

  return null;
});

// Computed property for total slices
const totalSlices = computed(() => {
  if (predictionSlices.value.length > 0) {
    return predictionSlices.value.length;
  }
  return previewSlices.value.length;
});

const handleScroll = (e) => {
  if (totalSlices.value === 0 || loading.value || loadingPreview.value) return;
  if (e.deltaY > 0 && currentSlice.value < totalSlices.value - 1) {
    currentSlice.value++;
  } else if (e.deltaY < 0 && currentSlice.value > 0) {
    currentSlice.value--;
  }
};

const triggerFileInput = () => {
  if (!loading.value && !loadingPreview.value) {
    fileInput.value.click();
  }
};

const handleFileChange = async (event) => {
  const file = event.target.files[0];
  if (file) {
    selectedFile.value = file;
    fileName.value = file.name;
    predictionSlices.value = [];
    showHeatmap.value = false;
    await generatePreview();
  }
};

const handleDrop = async (e) => {
  isDragging.value = false;
  if (loading.value || loadingPreview.value) return;

  const file = e.dataTransfer.files[0];
  if (file) {
    selectedFile.value = file;
    fileName.value = file.name;
    predictionSlices.value = [];
    showHeatmap.value = false;
    await generatePreview();
  }
};

const generatePreview = async () => {
  if (!selectedFile.value) return;
  loadingPreview.value = true;
  const formData = new FormData();
  formData.append("file", selectedFile.value);
  try {
    const res = await axios.post(`${API_URL}/preview`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    previewSlices.value = res.data.slices || [];
    currentSlice.value = 0;
  } catch (err) {
    console.error("Preview failed:", err);
    alert("Preview failed. Check backend logs.");
  } finally {
    loadingPreview.value = false;
  }
};

const submitFile = async () => {
  if (!selectedFile.value) return alert("Select a file first.");
  loading.value = true;
  predictionSlices.value = [];

  const formData = new FormData();
  formData.append("file", selectedFile.value);

  try {
    const res = await axios.post(`${API_URL}/predict`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    // Store the prediction results (array of {slice_index, raw, overlay})
    if (res.data.slices && res.data.slices.length > 0) {
      predictionSlices.value = res.data.slices;
      currentSlice.value = 0;
      showHeatmap.value = false; // Start with raw images
      console.log(`[INFO] Received ${predictionSlices.value.length} analyzed slices`);
    } else {
      alert("No results returned from prediction.");
    }
  } catch (err) {
    console.error("Prediction failed:", err);
    alert("Prediction failed. Check backend logs.");
  } finally {
    loading.value = false;
  }
};

const resetSelection = () => {
  if (loading.value) return;

  selectedFile.value = null;
  fileName.value = "";
  previewSlices.value = [];
  predictionSlices.value = [];
  currentSlice.value = 0;
  showHeatmap.value = false;
  if (fileInput.value) fileInput.value.value = "";
};

const toggleHeatmap = () => {
  if (predictionSlices.value.length > 0 && !loading.value) {
    showHeatmap.value = !showHeatmap.value;
  }
};
</script>

<style>
  @keyframes pulseLight {
    0%, 100% {
      transform: scale(1);
      box-shadow: 0 0 6px 2px rgba(255, 255, 255, 0.1);
    }
    50% {
      transform: scale(1.15);
      box-shadow: 0 0 10px 4px rgba(255, 255, 255, 0.15);
    }
  }

  @keyframes progress {
    0% {
      transform: translateX(-100%);
    }
    100% {
      transform: translateX(400%);
    }
  }
</style>
