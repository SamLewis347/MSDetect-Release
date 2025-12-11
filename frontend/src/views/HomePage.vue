<template>
  <div class="text-slate-100">

    <!-- HERO SECTION -->
    <section class="relative min-h-screen flex items-center justify-center text-center px-6 lg:px-0 overflow-hidden">

      <!-- Content -->
      <div class="relative z-10 max-w-3xl space-y-6">
        <h1 class="text-4xl sm:text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-bold leading-tight text-white">
          Welcome to<br>MS<span class="text-sky-400">Detect</span>
        </h1>
        <p class="text-base sm:text-lg md:text-xl text-slate-300">
          AI-assisted MRI Neurological Screening Portal
        </p>
        <router-link
          to="/diagnostic-tool"
          class="bg-sky-600 text-white inline-block px-6 py-3 rounded-md font-semibold shadow-lg hover:opacity-90 hover:scale-[1.02] transition-transform duration-200"
        >
          Open Analysis Tool
        </router-link>
      </div>
      <!-- Bottom Fade -->
      <div class="absolute bottom-0 left-0 w-full h-20 bg-gradient-to-t from-slate-950/90 to-transparent pointer-events-none"></div>
    </section>

    <!-- ABOUT / MISSION / WHAT IS MS -->
    <section class="py-12 sm:py-16 lg:py-20 bg-slate-950/80">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:flex lg:gap-12 lg:items-start">

        <!-- Left Column: Mission + About -->
        <div class="lg:w-1/2 space-y-6 sm:space-y-8">
          <h2 class="text-2xl sm:text-3xl font-bold text-white mb-2">About MS<span class="text-sky-400">Detect</span></h2>

          <p class="text-sm sm:text-base text-slate-300 leading-relaxed">
            MSDetect is a senior research project developed at Florida Southern College.
            Its goal is to explore how artificial intelligence can support earlier,
            more accurate detection of Multiple Sclerosis (MS) using MRI data.
          </p>

          <p class="text-sm sm:text-base text-slate-300 leading-relaxed">
            This project is deeply personal — born from witnessing firsthand the
            challenges of MS within family. Autoimmune diseases like MS remain
            under-researched, underfunded, and often diagnosed only after significant
            neurological damage has already occurred.
          </p>

          <p class="text-sm sm:text-base text-slate-300 leading-relaxed">
            Our mission is to demonstrate the potential of modern AI tools to help
            clinicians catch MS earlier in its progression and to inspire further
            medical research in autoimmune disorders.
          </p>

          <!-- Bullet Points -->
          <ul class="space-y-4 sm:space-y-6 mt-6 sm:mt-10">

            <!-- What is MS? -->
            <li class="flex items-start gap-3 sm:gap-4 transition-transform duration-500 hover:scale-[1.03]">
              <div class="flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center
                          bg-sky-400 text-slate-900">
                <i class="fa-solid fa-dna text-base sm:text-lg"></i>
              </div>
              <div>
                <h3 class="text-lg sm:text-xl font-semibold text-white">What is Multiple Sclerosis?</h3>
                <p class="text-sm sm:text-base text-slate-300 mt-1">
                  MS is a chronic autoimmune disease where the immune system attacks the
                  protective myelin sheath around nerves, disrupting communication between
                  the brain and body. Early detection significantly improves long-term outcomes.
                </p>
              </div>
            </li>

            <!-- Why This Research Matters -->
            <li class="flex items-start gap-3 sm:gap-4 transition-transform duration-500 hover:scale-[1.03]">
              <div class="flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center
                          bg-sky-400 text-slate-900">
                <i class="fa-solid fa-heart-pulse text-base sm:text-lg"></i>
              </div>
              <div>
                <h3 class="text-lg sm:text-xl font-semibold text-white">Why We Built MS<span class="text-sky-400">Detect</span></h3>
                <p class="text-sm sm:text-base text-slate-300 mt-1">
                  Diagnosis of MS is often delayed — sometimes for years — leading to
                  avoidable disability. Our project aims to show how AI-assisted MRI
                  analysis could support clinicians in catching early signs of disease
                  activity sooner.
                </p>
              </div>
            </li>

            <!-- Lack of Autoimmune Research -->
            <li class="flex items-start gap-3 sm:gap-4 transition-transform duration-500 hover:scale-[1.03]">
              <div class="flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center
                          bg-sky-400 text-slate-900">
                <i class="fa-solid fa-flask text-base sm:text-lg"></i>
              </div>
              <div>
                <h3 class="text-lg sm:text-xl font-semibold text-white">Addressing Research Gaps</h3>
                <p class="text-sm sm:text-base text-slate-300 mt-1">
                  Despite affecting millions, autoimmune disorders remain poorly understood.
                  MSDetect advocates for increased research attention and demonstrates how
                  accessible AI tools can accelerate scientific progress.
                </p>
              </div>
            </li>

            <!-- Inspiring Future Work -->
            <li class="flex items-start gap-3 sm:gap-4 transition-transform duration-500 hover:scale-[1.03]">
              <div class="flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center
                          bg-sky-400 text-slate-900">
                <i class="fa-solid fa-lightbulb text-base sm:text-lg"></i>
              </div>
              <div>
                <h3 class="text-lg sm:text-xl font-semibold text-white">Inspiring Future Innovation</h3>
                <p class="text-sm sm:text-base text-slate-300 mt-1">
                  We hope this project encourages students, clinicians, and researchers
                  to explore new ways AI can support the diagnosis and treatment of MS
                  and other autoimmune disorders.
                </p>
              </div>
            </li>

          </ul>
        </div>

        <!-- Right Column: Visual -->
        <div class="lg:w-1/2 mt-8 sm:mt-12 lg:mt-0 flex flex-col items-center">
          <div class="overflow-hidden rounded-xl shadow-xl w-full sm:w-2/3 max-w-full">
            <img :src="sample_mri_and_heatmap_output"
                alt=""
                class="w-full h-auto transition-transform duration-500 hover:scale-[1.03]"
            >
          </div>
          <!-- Subtitle below the image, matching width -->
          <p class="mt-2 text-xs sm:text-sm text-slate-400 w-full sm:w-2/3 text-center px-4 sm:px-0">
            Sample MRI Slice and Corresponding AI-Generated Heatmap Highlighting Regions of Interest for Multiple Sclerosis Detection
          </p>
        </div>
      </div>
    </section>

    <!-- PIPELINE SECTION -->
    <section class="py-12 sm:py-16 lg:py-24 bg-slate-900/80 relative">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 space-y-8 sm:space-y-12 lg:space-y-16">
        <h2 class="text-2xl sm:text-3xl font-bold text-white text-center">AI Analysis <span class="text-sky-400">Pipeline</span></h2>

        <!-- MOBILE/TABLET VIEW (< lg) -->
        <div class="lg:hidden flex flex-col items-center space-y-8 px-4">

          <!-- Background decorative elements -->
          <div class="absolute inset-0 pointer-events-none overflow-hidden">
            <div class="absolute top-0 left-1/2 -translate-x-1/2 w-64 h-64 bg-sky-500/5 rounded-full blur-3xl"></div>
          </div>

          <!-- Step 1: Raw MRI -->
          <div class="relative flex flex-col items-center text-center space-y-3 max-w-sm group">
            <div class="relative">
              <div class="absolute inset-0 bg-sky-400/20 rounded-full blur-lg group-hover:blur-xl transition-all duration-500"></div>
              <div class="relative w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gradient-to-br from-sky-500/30 to-sky-600/30 border-2 border-sky-400/60 flex items-center justify-center text-sky-400 text-3xl sm:text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 transition-all duration-500">
                <i class="fa-solid fa-brain"></i>
              </div>
            </div>
            <div class="space-y-1">
              <h3 class="text-lg sm:text-xl font-bold text-white group-hover:text-sky-300 transition-colors duration-300">Raw MRI (NIfTI)</h3>
              <p class="text-slate-400 text-sm leading-relaxed">Patients upload a 3D MRI scan (.nii).</p>
            </div>
          </div>

          <!-- Animated connector -->
          <div class="relative w-1 h-12">
            <div class="absolute inset-0 bg-gradient-to-b from-sky-400 via-sky-400/70 to-sky-400/50 rounded-full"></div>
            <div class="absolute inset-0 bg-gradient-to-b from-sky-400/50 to-transparent animate-pulse"></div>
          </div>

          <!-- Step 2: Slice Extraction -->
          <div class="relative flex flex-col items-center text-center space-y-3 max-w-sm group">
            <div class="relative">
              <div class="absolute inset-0 bg-sky-400/20 rounded-full blur-lg group-hover:blur-xl transition-all duration-500"></div>
              <div class="relative w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gradient-to-br from-sky-500/30 to-sky-600/30 border-2 border-sky-400/60 flex items-center justify-center text-sky-400 text-3xl sm:text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 transition-all duration-500">
                <i class="fa-solid fa-image"></i>
              </div>
            </div>
            <div class="space-y-1">
              <h3 class="text-lg sm:text-xl font-bold text-white group-hover:text-sky-300 transition-colors duration-300">Slice Extraction</h3>
              <p class="text-slate-400 text-sm leading-relaxed">Convert 3D scan into 2D PNG slices.</p>
            </div>
          </div>

          <!-- Fork indicator with labels -->
          <div class="flex flex-col items-center gap-4 w-full max-w-md">
            <div class="flex items-center justify-center gap-3 w-full">
              <div class="flex-1 h-1 bg-gradient-to-r from-transparent via-amber-400/50 to-amber-400 rounded-full"></div>
              <div class="relative">
                <div class="absolute inset-0 bg-sky-400/30 rounded-full blur-md animate-pulse"></div>
                <div class="relative w-4 h-4 rounded-full bg-gradient-to-br from-sky-400 to-sky-500 shadow-lg"></div>
              </div>
              <div class="flex-1 h-1 bg-gradient-to-l from-transparent via-emerald-400/50 to-emerald-400 rounded-full"></div>
            </div>
          </div>

          <!-- Parallel paths -->
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-8 sm:gap-12 w-full max-w-2xl">

            <!-- LEFT PATH: Training -->
            <div class="flex flex-col items-center space-y-6 relative">
              <!-- Path label -->
              <div class="bg-gradient-to-r from-amber-500/20 to-amber-600/20 border border-amber-400/40 rounded-full px-4 py-1.5 text-amber-300 text-xs font-semibold tracking-wider backdrop-blur-sm shadow-lg mb-2">
                TRAINING PATH
              </div>

              <!-- Step 3: Patch Extraction -->
              <div class="flex flex-col items-center text-center space-y-3 group">
                <div class="relative">
                  <div class="absolute inset-0 bg-amber-400/20 rounded-full blur-lg group-hover:blur-xl transition-all duration-500"></div>
                  <div class="relative w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gradient-to-br from-amber-500/30 to-amber-600/30 border-2 border-amber-400/60 flex items-center justify-center text-amber-400 text-3xl sm:text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 transition-all duration-500">
                    <i class="fa-solid fa-th-large"></i>
                  </div>
                </div>
                <div class="space-y-1">
                  <h3 class="text-base sm:text-lg font-bold text-white group-hover:text-amber-300 transition-colors duration-300">Patch Extraction</h3>
                  <p class="text-slate-400 text-xs sm:text-sm leading-relaxed">Each slice is divided into 32×32 patches; only patches with ≥50% brain tissue are used.</p>
                </div>
              </div>

              <div class="relative w-1 h-10">
                <div class="absolute inset-0 bg-gradient-to-b from-amber-400 via-amber-400/70 to-amber-400/50 rounded-full"></div>
                <div class="absolute inset-0 bg-gradient-to-b from-amber-400/50 to-transparent animate-pulse"></div>
              </div>

              <!-- Step 4: Model Training -->
              <div class="flex flex-col items-center text-center space-y-3 group">
                <div class="relative">
                  <div class="absolute inset-0 bg-amber-400/20 rounded-full blur-lg group-hover:blur-xl transition-all duration-500"></div>
                  <div class="relative w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gradient-to-br from-amber-500/30 to-amber-600/30 border-2 border-amber-400/60 flex items-center justify-center text-amber-400 text-3xl sm:text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 transition-all duration-500">
                    <i class="fa-solid fa-gears"></i>
                  </div>
                </div>
                <div class="space-y-1">
                  <h3 class="text-base sm:text-lg font-bold text-white group-hover:text-amber-300 transition-colors duration-300">Patch-Based CNN Training</h3>
                  <p class="text-slate-400 text-xs sm:text-sm leading-relaxed">The model is trained to classify small patches for lesion likelihood.</p>
                </div>
              </div>
            </div>

            <!-- RIGHT PATH: Inference -->
            <div class="flex flex-col items-center space-y-6 relative">
              <!-- Path label -->
              <div class="bg-gradient-to-r from-emerald-500/20 to-emerald-600/20 border border-emerald-400/40 rounded-full px-4 py-1.5 text-emerald-300 text-xs font-semibold tracking-wider backdrop-blur-sm shadow-lg mb-2">
                INFERENCE PATH
              </div>

              <!-- Step 5: Sliding Window -->
              <div class="flex flex-col items-center text-center space-y-3 group">
                <div class="relative">
                  <div class="absolute inset-0 bg-emerald-400/20 rounded-full blur-lg group-hover:blur-xl transition-all duration-500"></div>
                  <div class="relative w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gradient-to-br from-emerald-500/30 to-emerald-600/30 border-2 border-emerald-400/60 flex items-center justify-center text-emerald-400 text-3xl sm:text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 transition-all duration-500">
                    <i class="fa-solid fa-border-all"></i>
                  </div>
                </div>
                <div class="space-y-1">
                  <h3 class="text-base sm:text-lg font-bold text-white group-hover:text-emerald-300 transition-colors duration-300">Sliding Window Inference</h3>
                  <p class="text-slate-400 text-xs sm:text-sm leading-relaxed">At test time, a sliding window generates patch predictions for every pixel area.</p>
                </div>
              </div>

              <div class="relative w-1 h-10">
                <div class="absolute inset-0 bg-gradient-to-b from-emerald-400 via-emerald-400/70 to-emerald-400/50 rounded-full"></div>
                <div class="absolute inset-0 bg-gradient-to-b from-emerald-400/50 to-transparent animate-pulse"></div>
              </div>

              <!-- Step 6: Heatmap Generation -->
              <div class="flex flex-col items-center text-center space-y-3 group">
                <div class="relative">
                  <div class="absolute inset-0 bg-emerald-400/20 rounded-full blur-lg group-hover:blur-xl transition-all duration-500"></div>
                  <div class="relative w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gradient-to-br from-emerald-500/30 to-emerald-600/30 border-2 border-emerald-400/60 flex items-center justify-center text-emerald-400 text-3xl sm:text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 transition-all duration-500">
                    <i class="fa-solid fa-fire"></i>
                  </div>
                </div>
                <div class="space-y-1">
                  <h3 class="text-base sm:text-lg font-bold text-white group-hover:text-emerald-300 transition-colors duration-300">Heatmap Generation</h3>
                  <p class="text-slate-400 text-xs sm:text-sm leading-relaxed">Patch predictions are stitched into a heatmap to visualize lesion probability.</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- DESKTOP VIEW (>= lg) -->
        <div class="hidden lg:block">
          <div class="flex justify-center">
            <div class="w-full max-w-6xl relative">

              <!-- Background decorative elements -->
              <div class="absolute inset-0 pointer-events-none overflow-hidden">
                <div class="absolute top-0 left-1/2 -translate-x-1/2 w-96 h-96 bg-sky-500/5 rounded-full blur-3xl"></div>
                <div class="absolute bottom-0 left-1/4 w-64 h-64 bg-sky-400/5 rounded-full blur-3xl"></div>
                <div class="absolute bottom-0 right-1/4 w-64 h-64 bg-sky-400/5 rounded-full blur-3xl"></div>
              </div>

              <div class="relative grid grid-cols-5 gap-16 auto-rows-[minmax(120px,auto)]">

                <!-- Raw MRI node (NODE1) - Centered at top -->
                <div id="node1" class="col-start-3 row-start-1 flex flex-col items-center text-center space-y-4 group">
                  <div class="relative">
                    <div class="absolute inset-0 bg-sky-400/20 rounded-full blur-xl group-hover:blur-2xl transition-all duration-500"></div>
                    <div class="relative w-20 h-20 rounded-full bg-gradient-to-br from-sky-500/30 to-sky-600/30 border-2 border-sky-400/60 flex items-center justify-center text-sky-400 text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 group-hover:border-sky-300 transition-all duration-500">
                      <i class="fa-solid fa-brain"></i>
                    </div>
                  </div>
                  <div class="space-y-2">
                    <h3 class="text-xl font-bold text-white group-hover:text-sky-300 transition-colors duration-300">Raw MRI (NIfTI)</h3>
                    <p class="text-slate-400 text-sm max-w-xs leading-relaxed">Patients upload a 3D MRI scan (.nii).</p>
                  </div>
                </div>

                <!-- Slice extraction node (NODE2) - Center column -->
                <div id="node2" class="col-start-3 row-start-2 flex flex-col items-center text-center space-y-4 group">
                  <div class="relative">
                    <div class="absolute inset-0 bg-sky-400/20 rounded-full blur-xl group-hover:blur-2xl transition-all duration-500"></div>
                    <div class="relative w-20 h-20 rounded-full bg-gradient-to-br from-sky-500/30 to-sky-600/30 border-2 border-sky-400/60 flex items-center justify-center text-sky-400 text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 group-hover:border-sky-300 transition-all duration-500">
                      <i class="fa-solid fa-image"></i>
                    </div>
                  </div>
                  <div class="space-y-2">
                    <h3 class="text-xl font-bold text-white group-hover:text-sky-300 transition-colors duration-300">Slice Extraction</h3>
                    <p class="text-slate-400 text-sm max-w-xs leading-relaxed">Convert 3D scan into 2D PNG slices.</p>
                  </div>
                </div>

                <!-- LEFT BRANCH LABEL -->
                <div class="col-start-2 row-start-3 flex items-center justify-center">
                  <div class="bg-gradient-to-r from-amber-500/20 to-amber-600/20 border border-amber-400/40 rounded-full px-4 py-2 text-amber-300 text-xs font-semibold tracking-wider backdrop-blur-sm shadow-lg">
                    TRAINING PATH
                  </div>
                </div>

                <!-- RIGHT BRANCH LABEL -->
                <div class="col-start-4 row-start-3 flex items-center justify-center">
                  <div class="bg-gradient-to-r from-emerald-500/20 to-emerald-600/20 border border-emerald-400/40 rounded-full px-4 py-2 text-emerald-300 text-xs font-semibold tracking-wider backdrop-blur-sm shadow-lg">
                    INFERENCE PATH
                  </div>
                </div>

                <!-- Patch extraction node (NODE3) - Left branch -->
                <div id="node3" class="col-start-2 row-start-4 flex flex-col items-center text-center space-y-4 group">
                  <div class="relative">
                    <div class="absolute inset-0 bg-amber-400/20 rounded-full blur-xl group-hover:blur-2xl transition-all duration-500"></div>
                    <div class="relative w-20 h-20 rounded-full bg-gradient-to-br from-amber-500/30 to-amber-600/30 border-2 border-amber-400/60 flex items-center justify-center text-amber-400 text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 group-hover:border-amber-300 transition-all duration-500">
                      <i class="fa-solid fa-th-large"></i>
                    </div>
                  </div>
                  <div class="space-y-2">
                    <h3 class="text-xl font-bold text-white group-hover:text-amber-300 transition-colors duration-300">Patch Extraction</h3>
                    <p class="text-slate-400 text-sm max-w-xs leading-relaxed">
                      Each slice is divided into 32×32 patches; only patches with ≥50% brain tissue are used.
                    </p>
                  </div>
                </div>

                <!-- Sliding window inference node (NODE5) - Right branch -->
                <div id="node5" class="col-start-4 row-start-4 flex flex-col items-center text-center space-y-4 group">
                  <div class="relative">
                    <div class="absolute inset-0 bg-emerald-400/20 rounded-full blur-xl group-hover:blur-2xl transition-all duration-500"></div>
                    <div class="relative w-20 h-20 rounded-full bg-gradient-to-br from-emerald-500/30 to-emerald-600/30 border-2 border-emerald-400/60 flex items-center justify-center text-emerald-400 text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 group-hover:border-emerald-300 transition-all duration-500">
                      <i class="fa-solid fa-border-all"></i>
                    </div>
                  </div>
                  <div class="space-y-2">
                    <h3 class="text-xl font-bold text-white group-hover:text-emerald-300 transition-colors duration-300">Sliding Window Inference</h3>
                    <p class="text-slate-400 text-sm max-w-xs leading-relaxed">
                      At test time, a sliding window generates patch predictions for every pixel area.
                    </p>
                  </div>
                </div>

                <!-- Model training node (NODE4) - Left branch bottom -->
                <div id="node4" class="col-start-2 row-start-5 flex flex-col items-center text-center space-y-4 group">
                  <div class="relative">
                    <div class="absolute inset-0 bg-amber-400/20 rounded-full blur-xl group-hover:blur-2xl transition-all duration-500"></div>
                    <div class="relative w-20 h-20 rounded-full bg-gradient-to-br from-amber-500/30 to-amber-600/30 border-2 border-amber-400/60 flex items-center justify-center text-amber-400 text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 group-hover:border-amber-300 transition-all duration-500">
                      <i class="fa-solid fa-gears"></i>
                    </div>
                  </div>
                  <div class="space-y-2">
                    <h3 class="text-xl font-bold text-white group-hover:text-amber-300 transition-colors duration-300">Patch-Based CNN Training</h3>
                    <p class="text-slate-400 text-sm max-w-xs leading-relaxed">
                      The model is trained to classify small patches for lesion likelihood.
                    </p>
                  </div>
                </div>

                <!-- Heatmap generation node (NODE6) - Right branch bottom -->
                <div id="node6" class="col-start-4 row-start-5 flex flex-col items-center text-center space-y-4 group">
                  <div class="relative">
                    <div class="absolute inset-0 bg-emerald-400/20 rounded-full blur-xl group-hover:blur-2xl transition-all duration-500"></div>
                    <div class="relative w-20 h-20 rounded-full bg-gradient-to-br from-emerald-500/30 to-emerald-600/30 border-2 border-emerald-400/60 flex items-center justify-center text-emerald-400 text-4xl shadow-2xl backdrop-blur-md group-hover:scale-110 group-hover:border-emerald-300 transition-all duration-500">
                      <i class="fa-solid fa-fire"></i>
                    </div>
                  </div>
                  <div class="space-y-2">
                    <h3 class="text-xl font-bold text-white group-hover:text-emerald-300 transition-colors duration-300">Heatmap Generation</h3>
                    <p class="text-slate-400 text-sm max-w-xs leading-relaxed">
                      Patch predictions are stitched into a heatmap to visualize lesion probability.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- TEAM SECTION -->
    <section class="py-12 sm:py-16 lg:py-20 w-full bg-slate-950/80">

      <!-- Inner constrained container -->
      <div class="max-w-7xl mx-auto px-4 sm:px-6 space-y-8 sm:space-y-12">
        <h2 class="text-2xl sm:text-3xl font-bold text-white text-center">Meet the <span class="text-sky-400">Team</span></h2>

        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-8">
          <TeamCard
            v-for="(member, idx) in teamMembers"
            :key="idx"
            :member="member"
          />
        </div>
      </div>
    </section>

    <!-- ACHIEVEMENTS -->
    <section class="py-12 sm:py-16 bg-slate-900/80">
      <!-- Section Heading -->
      <h2 class="mb-8 sm:mb-12 text-2xl sm:text-3xl font-bold text-white text-center px-4">
        Our <span class="text-sky-400">Progress</span>
      </h2>
      <div class="max-w-5xl mx-auto px-4 sm:px-6 grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 text-center">

        <!-- Patient Scans -->
        <div class="relative group bg-slate-950/70
                    border border-sky-400/20 rounded-2xl p-6 sm:p-8 shadow-lg
                    hover:shadow-[0_0_40px_rgba(14,165,233,0.25)]
                    transition-all duration-300 hover:scale-[1.03]">

          <!-- Glow edge accent -->
          <div class="absolute inset-0 rounded-2xl pointer-events-none
                      opacity-0 group-hover:opacity-100 transition-opacity duration-300
                      bg-gradient-to-br from-sky-400/20 to-transparent"></div>

          <h3 class="text-2xl sm:text-3xl font-bold text-sky-400">Over 250</h3>
          <p class="text-slate-300 mt-2 text-base sm:text-lg">Patient Scans Analyzed</p>
        </div>

        <!-- Detection Accuracy -->
        <div class="relative group bg-slate-950/70
                    border border-sky-400/20 rounded-2xl p-6 sm:p-8 shadow-lg
                    hover:shadow-[0_0_40px_rgba(14,165,233,0.25)]
                    transition-all duration-300 hover:scale-[1.03]">

          <div class="absolute inset-0 rounded-2xl pointer-events-none
                      opacity-0 group-hover:opacity-100 transition-opacity duration-300
                      bg-gradient-to-br from-sky-400/20 to-transparent"></div>

          <h3 class="text-2xl sm:text-3xl font-bold text-sky-400">Over 75%</h3>
          <p class="text-slate-300 mt-2 text-base sm:text-lg">Detection Accuracy on Real World Data</p>
        </div>
      </div>
    </section>
    <!-- Top fade before footer -->
    <div class="w-full h-20 bg-gradient-to-b from-slate-900/80 to-transparent pointer-events-none"></div>

  </div>
</template>

<script setup>
import sample_mri_and_heatmap_output from "@/assets/sample_mri_and_heatmap_output.png";
import TeamCard from "@/components/TeamCard.vue";
import { onMounted, onBeforeUnmount } from "vue";
import mayfieldImg from "@/assets/team/mayfield-headshot.jpg";
import lewisImg from "@/assets/team/lewis-headshot.JPEG";
import bensonImg from "@/assets/team/headshot-benson.PNG";
import eicholtzImg from "@/assets/team/headshot-eicholtz.JPG";

let lines = [];

function connectNodes() {
  // Only run LeaderLine on desktop
  if (window.innerWidth < 1024) return;

  lines.forEach(line => line.remove());
  lines = [];

  if (!window.LeaderLine) return;

  // Main vertical spine (sky blue for initial path)
  lines.push(new window.LeaderLine(document.getElementById("node1"), document.getElementById("node2"), {
    color: "#38bdf8", size: 5, endPlug: "arrow3", path: "straight", startSocket: "bottom", endSocket: "top",
    dash: { animation: true }
  }));

  // Split to left (Training path - amber)
  lines.push(new window.LeaderLine(document.getElementById("node2"), document.getElementById("node3"), {
    color: "#fbbf24", size: 5, endPlug: "arrow3", path: "straight", startSocket: "bottom", endSocket: "top",
    gradient: true, dash: { animation: true }
  }));

  // Continue training path
  lines.push(new window.LeaderLine(document.getElementById("node3"), document.getElementById("node4"), {
    color: "#fbbf24", size: 5, endPlug: "arrow3", path: "straight", startSocket: "bottom", endSocket: "top",
    dash: { animation: true }
  }));

  // Split to right (Inference path - emerald)
  lines.push(new window.LeaderLine(document.getElementById("node2"), document.getElementById("node5"), {
    color: "#34d399", size: 5, endPlug: "arrow3", path: "straight", startSocket: "bottom", endSocket: "top",
    gradient: true, dash: { animation: true }
  }));

  // Continue inference path
  lines.push(new window.LeaderLine(document.getElementById("node5"), document.getElementById("node6"), {
    color: "#34d399", size: 5, endPlug: "arrow3", path: "straight", startSocket: "bottom", endSocket: "top",
    dash: { animation: true }
  }));
}

onMounted(() => {
  setTimeout(connectNodes, 50);
  window.addEventListener("resize", connectNodes);
});

onBeforeUnmount(() => {
  lines.forEach(line => line.remove());
  window.removeEventListener("resize", connectNodes);
});

const teamMembers = [
  { name: "Tucker Mayfield", image: mayfieldImg },
  { name: "Samuel Lewis", image: lewisImg },
  { name: "Daniel Benson", image: bensonImg },
  { name: "Dr. Matthew Eicholtz", image: eicholtzImg },
];
</script>
