<template>
  <div class="relative min-h-screen">

    <!-- Base dark gradient (fixed) -->
    <div class="fixed inset-0 z-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-950"></div>

    <!-- SVG network background (fixed) -->
    <svg class="fixed inset-0 w-full h-full z-0" preserveAspectRatio="none">
      <defs>
        <radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="rgb(14,165,233)" stop-opacity="0.8"/>
          <stop offset="100%" stop-color="rgb(14,165,233)" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <g v-for="node in nodes" :key="node.id">
        <!-- lines to connected nodes -->
        <line
          v-for="targetId in node.connections"
          :key="node.id + '-' + targetId"
          :x1="node.x"
          :y1="node.y"
          :x2="nodesMap[targetId].x"
          :y2="nodesMap[targetId].y"
          stroke="rgba(14,165,233,0.2)"
          stroke-width="1"
        />
        <!-- node circle with glow -->
        <circle
          :cx="node.x"
          :cy="node.y"
          r="4"
          fill="url(#nodeGlow)"
        />
        <circle
          :cx="node.x"
          :cy="node.y"
          r="2"
          fill="rgb(14,165,233)"
        />
      </g>
    </svg>

    <!-- Content Layer (scrollable) -->
    <div class="relative z-10 flex flex-col">
      <AppHeader />
      <main class="flex-1">
        <router-view />
      </main>
      <AppFooter />
    </div>

  </div>
</template>

<script setup>
import AppHeader from './components/AppHeader.vue';
import AppFooter from './components/AppFooter.vue';
import { reactive, computed, onMounted, ref } from "vue";

// Track viewport size
const width = ref(window.innerWidth);
const height = ref(window.innerHeight);
window.addEventListener('resize', () => {
  width.value = window.innerWidth;
  height.value = window.innerHeight;
});

const nodes = reactive(Array.from({ length: 30 }, (_, i) => ({
  id: i,
  x: Math.random() * width.value,
  y: Math.random() * height.value,
  dx: (Math.random()-0.5) * 0.3,
  dy: (Math.random()-0.5) * 0.3,
  connections: []
})));

// Map nodes by id
const nodesMap = computed(() => {
  const map = {};
  nodes.forEach(n => map[n.id] = n);
  return map;
});

// Random connections
nodes.forEach(node => {
  const numConnections = Math.floor(Math.random() * 3) + 1;
  for(let i=0;i<numConnections;i++){
    const target = Math.floor(Math.random() * nodes.length);
    if(target !== node.id && !node.connections.includes(target)){
      node.connections.push(target);
    }
  }
});

// Animate nodes
const animate = () => {
  nodes.forEach(node => {
    node.x += node.dx;
    node.y += node.dy;

    if(node.x < 0 || node.x > width.value) node.dx *= -1;
    if(node.y < 0 || node.y > height.value) node.dy *= -1;
  });
  requestAnimationFrame(animate);
};

onMounted(() => {
  animate();
});
</script>

<style>
svg {
  pointer-events: none;
}
</style>
