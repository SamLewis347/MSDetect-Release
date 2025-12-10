<template>
  <div class="text-slate-100">

    <!-- HERO SECTION -->
    <section class="relative min-h-screen flex items-center justify-center text-center px-6 lg:px-0 overflow-hidden">

      <!-- Content -->
      <div class="relative z-10 max-w-3xl space-y-6">
        <h1 class="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold leading-tight text-white">
          Welcome to<br>MS<span class="text-sky-400">Detect</span>
        </h1>
        <p class="text-lg md:text-xl text-slate-300">
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
    <section class="py-20 bg-slate-950/80">
      <div class="max-w-7xl mx-auto px-6 lg:flex lg:gap-12 lg:items-start">

        <!-- Left Column: Mission + About -->
        <div class="lg:w-1/2 space-y-8">
          <h2 class="text-3xl font-bold text-white mb-2">About MS<span class="text-sky-400">Detect</span></h2>

          <p class="text-slate-300 leading-relaxed">
            MSDetect is a senior research project developed at Florida Southern College.
            Its goal is to explore how artificial intelligence can support earlier,
            more accurate detection of Multiple Sclerosis (MS) using MRI data.
          </p>

          <p class="text-slate-300 leading-relaxed">
            This project is deeply personal — born from witnessing firsthand the
            challenges of MS within family. Autoimmune diseases like MS remain
            under-researched, underfunded, and often diagnosed only after significant
            neurological damage has already occurred.
          </p>

          <p class="text-slate-300 leading-relaxed">
            Our mission is to demonstrate the potential of modern AI tools to help
            clinicians catch MS earlier in its progression and to inspire further
            medical research in autoimmune disorders.
          </p>

          <!-- Bullet Points -->
          <ul class="space-y-6 mt-10">

            <!-- What is MS? -->
            <li class="flex items-start gap-4 transition-transform duration-500 hover:scale-[1.03]">
              <div class="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
                          bg-sky-400 text-slate-900">
                <i class="fa-solid fa-dna text-lg"></i>
              </div>
              <div>
                <h3 class="text-xl font-semibold text-white">What is Multiple Sclerosis?</h3>
                <p class="text-slate-300">
                  MS is a chronic autoimmune disease where the immune system attacks the
                  protective myelin sheath around nerves, disrupting communication between
                  the brain and body. Early detection significantly improves long-term outcomes.
                </p>
              </div>
            </li>

            <!-- Why This Research Matters -->
            <li class="flex items-start gap-4 transition-transform duration-500 hover:scale-[1.03]">
              <div class="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
                          bg-sky-400 text-slate-900">
                <i class="fa-solid fa-heart-pulse text-lg"></i>
              </div>
              <div>
                <h3 class="text-xl font-semibold text-white">Why We Built MS<span class="text-sky-400">Detect</span></h3>
                <p class="text-slate-300">
                  Diagnosis of MS is often delayed — sometimes for years — leading to
                  avoidable disability. Our project aims to show how AI-assisted MRI
                  analysis could support clinicians in catching early signs of disease
                  activity sooner.
                </p>
              </div>
            </li>

            <!-- Lack of Autoimmune Research -->
            <li class="flex items-start gap-4 transition-transform duration-500 hover:scale-[1.03]">
              <div class="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
                          bg-sky-400 text-slate-900">
                <i class="fa-solid fa-flask text-lg"></i>
              </div>
              <div>
                <h3 class="text-xl font-semibold text-white">Addressing Research Gaps</h3>
                <p class="text-slate-300">
                  Despite affecting millions, autoimmune disorders remain poorly understood.
                  MSDetect advocates for increased research attention and demonstrates how
                  accessible AI tools can accelerate scientific progress.
                </p>
              </div>
            </li>

            <!-- Inspiring Future Work -->
            <li class="flex items-start gap-4 transition-transform duration-500 hover:scale-[1.03]">
              <div class="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
                          bg-sky-400 text-slate-900">
                <i class="fa-solid fa-lightbulb text-lg"></i>
              </div>
              <div>
                <h3 class="text-xl font-semibold text-white">Inspiring Future Innovation</h3>
                <p class="text-slate-300">
                  We hope this project encourages students, clinicians, and researchers
                  to explore new ways AI can support the diagnosis and treatment of MS
                  and other autoimmune disorders.
                </p>
              </div>
            </li>

          </ul>
        </div>

        <!-- Right Column: Visual -->
        <div class="lg:w-1/2 mt-12 lg:mt-0 flex flex-col items-center">
          <div class="overflow-hidden rounded-lg shadow-xl w-2/3 max-w-full">
            <img :src="sample_mri_and_heatmap_output"
                alt=""
                class="mt-12 w-full h-auto transition-transform duration-500 hover:scale-[1.03]"
            >
          </div>
          <!-- Subtitle below the image, matching width -->
          <p class="mt-2 text-sm text-slate-400 w-2/3 text-center">
            Sample MRI Slice and Corresponding AI-Generated Heatmap Highlighting Regions of Interest for Multiple Sclerosis Detection
          </p>
        </div>
      </div>
    </section>

    <!-- PIPELINE SECTION -->
    <section class="py-24 bg-slate-900/80 relative">
      <div class="max-w-7xl mx-auto px-6 space-y-16">
        <h2 class="text-3xl font-bold text-white text-center">AI Analysis <span class="text-sky-400">Pipeline</span></h2>

        <!-- FLOWCHART GRID WRAPPER -->
        <div class="flex justify-center">
          <div class="transform scale-100 sm:scale-90 xs:scale-75 origin-top w-full max-w-5xl">
            <div class="grid grid-cols-4 gap-12 auto-rows-[minmax(100px,auto)] relative">

              <!-- Raw MRI node (NODE1) -->
              <div id="node1" class="col-start-2 row-start-1 flex flex-col items-center text-center space-y-4 transition-transform duration-500 hover:scale-[1.03]">
                <div class="w-16 h-16 rounded-full bg-sky-500/20 border border-sky-400/40 flex items-center justify-center text-sky-400 text-3xl shadow-lg backdrop-blur-md">
                  <i class="fa-solid fa-brain"></i>
                </div>
                <h3 class="text-xl font-semibold text-white">Raw MRI (NIfTI)</h3>
                <p class="mb-5 text-slate-300 text-sm">Patients upload a 3D MRI scan (.nii).</p>
              </div>

              <!-- Slice extraction node (NODE2) -->
              <div id="node2" class="col-start-2 row-start-2 flex flex-col items-center text-center space-y-4 transition-transform duration-500 hover:scale-[1.03]">
                <div class="w-16 h-16 mt-5 rounded-full bg-sky-500/20 border border-sky-400/40 flex items-center justify-center text-sky-400 text-3xl shadow-lg backdrop-blur-md">
                  <i class="fa-solid fa-image"></i>
                </div>
                <h3 class="text-xl font-semibold text-white">Slice Extraction</h3>
                <p class="text-slate-300 text-sm">Convert 3D scan into 2D PNG slices.</p>
              </div>

              <!-- Sliding window inference node (NODE5) -->
              <div id="node5" class="col-start-3 row-start-2 flex flex-col items-center text-center space-y-4 transition-transform duration-500 hover:scale-[1.03]">
                <div class="w-16 h-16 mt-5 rounded-full bg-sky-500/20 border border-sky-400/40 flex items-center justify-center text-sky-400 text-3xl shadow-lg backdrop-blur-md">
                  <i class="fa-solid fa-border-all"></i>
                </div>
                <h3 class="text-xl font-semibold text-white">Sliding Window Inference</h3>
                <p class="mb-5 text-slate-300 text-sm">
                  At test time, a sliding window generates patch predictions for every pixel area.
                </p>
              </div>

              <!-- Patch extraction node (NODE3) -->
              <div id="node3" class="col-start-2 row-start-3 flex flex-col items-center text-center space-y-4 transition-transform duration-500 hover:scale-[1.03]">
                <div class="w-16 h-16 mt-5 rounded-full bg-sky-500/20 border border-sky-400/40 flex items-center justify-center text-sky-400 text-3xl shadow-lg backdrop-blur-md">
                  <i class="fa-solid fa-th-large"></i>
                </div>
                <h3 class="text-xl font-semibold text-white">Patch Extraction</h3>
                <p class="mb-5 text-slate-300 text-sm">
                  Each slice is divided into 32×32 patches; only patches with ≥50% brain tissue are used.
                </p>
              </div>

              <!-- Model training node (NODE4) -->
              <div id="node4" class="col-start-2 row-start-4 flex flex-col items-center text-center space-y-4 transition-transform duration-500 hover:scale-[1.03]">
                <div class="w-16 h-16 mt-5 rounded-full bg-sky-500/20 border border-sky-400/40 flex items-center justify-center text-sky-400 text-3xl shadow-lg backdrop-blur-md">
                  <i class="fa-solid fa-gears"></i>
                </div>
                <h3 class="text-xl font-semibold text-white">Patch-Based CNN Training</h3>
                <p class="text-slate-300 text-sm">
                  The model is trained to classify small patches for lesion likelihood.
                </p>
              </div>

              <!-- Heatmap generation node (NODE6) -->
              <div id="node6" class="col-start-3 row-start-4 flex flex-col items-center text-center space-y-4 transition-transform duration-500 hover:scale-[1.03]">
                <div class="w-16 h-16 mt-5 rounded-full bg-sky-500/20 border border-sky-400/40 flex items-center justify-center text-sky-400 text-3xl shadow-lg backdrop-blur-md">
                  <i class="fa-solid fa-fire"></i>
                </div>
                <h3 class="text-xl font-semibold text-white">Heatmap Generation</h3>
                <p class="text-slate-300 text-sm">
                  Patch predictions are stitched into a heatmap to visualize lesion probability.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- TEAM SECTION -->
    <section class="py-20 w-full bg-slate-950/80">

      <!-- Inner constrained container -->
      <div class="max-w-7xl mx-auto px-6 space-y-12">
        <h2 class="text-3xl font-bold text-white text-center">Meet the <span class="text-sky-400">Team</span></h2>

        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
          <TeamCard
            v-for="(member, idx) in teamMembers"
            :key="idx"
            :member="member"
          />
        </div>
      </div>
    </section>

    <!-- ACHIEVEMENTS -->
    <section class="py-16 bg-slate-900/80">
      <!-- Section Heading -->
      <h2 class="mb-12 text-3xl font-bold text-white text-center">
        Our <span class="text-sky-400">Progress</span>
      </h2>
      <div class="max-w-5xl mx-auto px-6 grid grid-cols-1 md:grid-cols-2 gap-6 text-center">

        <!-- Patient Scans -->
        <div class="relative group bg-slate-950/70
                    border border-sky-400/20 rounded-2xl p-8 shadow-lg
                    hover:shadow-[0_0_40px_rgba(14,165,233,0.25)]
                    transition-all duration-300 hover:scale-[1.03]">

          <!-- Glow edge accent -->
          <div class="absolute inset-0 rounded-2xl pointer-events-none
                      opacity-0 group-hover:opacity-100 transition-opacity duration-300
                      bg-gradient-to-br from-sky-400/20 to-transparent"></div>

          <h3 class="text-3xl font-bold text-sky-400">Over 250+</h3>
          <p class="text-slate-300 mt-2 text-lg">Patient Scans Analyzed</p>
        </div>

        <!-- Detection Accuracy -->
        <div class="relative group bg-slate-950/70
                    border border-sky-400/20 rounded-2xl p-8 shadow-lg
                    hover:shadow-[0_0_40px_rgba(14,165,233,0.25)]
                    transition-all duration-300 hover:scale-[1.03]">

          <div class="absolute inset-0 rounded-2xl pointer-events-none
                      opacity-0 group-hover:opacity-100 transition-opacity duration-300
                      bg-gradient-to-br from-sky-400/20 to-transparent"></div>

          <h3 class="text-3xl font-bold text-sky-400">75%</h3>
          <p class="text-slate-300 mt-2 text-lg">Detection Accuracy on Real World Data</p>
        </div>
      </div>
    </section>
    <!-- Top fade before footer -->
    <div class="w-full h-20 bg-gradient-to-b from-slate-900/80 to-transparent pointer-events-none"></div>

    <!-- Spacer for the footer -->
    <div class="mb-24"></div>
  </div>
</template>

<script setup>
import sample_mri_and_heatmap_output from "@/assets/sample_mri_and_heatmap_output.png";
import TeamCard from "@/components/TeamCard.vue";
import { onMounted, onBeforeUnmount } from "vue";
import mayfieldImg from "@/assets/team/mayfield-headshot.jpg";
import lewisImg from "@/assets/team/lewis-headshot.JPEG";
import bensonImg from "@/assets/team/benson-headshot.jpg";
import eicholtzImg from "@/assets/team/headshot-eicholtz.JPG";

let lines = [];

function connectNodes() {
  lines.forEach(line => line.remove());
  lines = [];

  if (!window.LeaderLine) return;

  // Vertical spine
  lines.push(new window.LeaderLine(document.getElementById("node1"), document.getElementById("node2"), {
    color: "#38bdf8", size: 4, endPlug: "arrow1", path: "straight", startSocket: "bottom", endSocket: "top"
  }));
  lines.push(new window.LeaderLine(document.getElementById("node2"), document.getElementById("node3"), {
    color: "#38bdf8", size: 4, endPlug: "arrow1", path: "straight", startSocket: "bottom", endSocket: "top"
  }));
  lines.push(new window.LeaderLine(document.getElementById("node3"), document.getElementById("node4"), {
    color: "#38bdf8", size: 4, endPlug: "arrow1", path: "straight", startSocket: "bottom", endSocket: "top"
  }));

  // Right horizontal branch
  lines.push(new window.LeaderLine(document.getElementById("node2"), document.getElementById("node5"), {
    color: "#38bdf8", size: 4, endPlug: "arrow1", path: "straight", startSocket: "right", endSocket: "left",
    startPlugSize: 1.5,
  }));

  lines.push(new window.LeaderLine(document.getElementById("node5"), document.getElementById("node6"), {
    color: "#38bdf8", size: 4, endPlug: "arrow1", path: "straight", startSocket: "bottom", endSocket: "top",
    startPlugSize: 1.5
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
