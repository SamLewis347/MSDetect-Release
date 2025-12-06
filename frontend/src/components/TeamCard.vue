<template>
  <div
    class="relative group bg-gradient-to-br from-slate-800/40 to-slate-900/40
           border border-sky-400/10 hover:border-sky-400/40
           rounded-2xl p-6 sm:p-8
           shadow-lg hover:shadow-sky-400/20
           transition-all duration-300 hover:scale-[1.03]"
  >

    <!-- Glow edge accent -->
    <div
      class="absolute inset-0 rounded-2xl pointer-events-none
             opacity-0 group-hover:opacity-100 transition-opacity duration-300
             bg-gradient-to-br from-sky-400/10 to-transparent"
    ></div>

    <!-- Profile image -->
    <div class="relative w-28 h-28 sm:w-32 sm:h-32 mb-5 mx-auto">
      <img
        :src="member.image"
        :alt="member.name"
        class="w-full h-full rounded-full object-cover border-2 border-sky-300/30"
      />
      <div
        class="absolute inset-0 rounded-full bg-sky-400/20 blur-xl opacity-40 group-hover:opacity-70 transition"
      ></div>
    </div>

    <!-- Name -->
    <h3 class="text-center text-xl font-bold text-white mb-1 tracking-wide">
      {{ member.name }}
    </h3>
  </div>
</template>

<script setup>
defineProps({
  member: Object
});
</script>
