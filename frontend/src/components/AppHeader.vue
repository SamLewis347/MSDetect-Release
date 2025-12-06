<template>
  <header
    :class="[
      'fixed w-full top-0 left-0 z-50 transition-colors duration-300 backdrop-blur-xl border-b border-white/20',
      scrolled ? 'bg-slate-900/50 ' : 'bg-transparent'
    ]"
  >
    <div class="mx-auto px-6 lg:px-12">
      <div class="flex justify-between items-center h-16">
        <!-- Logo + Name -->
        <div class="flex items-center space-x-3">
          <img
            :src="logo"
            alt="AI for MS Logo"
            class="h-10 w-auto max-w-[150px] object-contain rounded-md"
          />
          <h1 class="text-xl font-extrabold text-white whitespace-nowrap">
            MS<span class="text-sky-400">Detect</span>
          </h1>
        </div>

        <!-- Desktop nav -->
        <nav class="hidden md:flex space-x-6">
          <a
            href="#"
            @click.prevent="handleNavClick"
            class="text-slate-200 hover:text-sky-400 transition-colors duration-200"
          >
            {{ navLinkText }}
          </a>
          <a
            href="https://github.com/SamLewis347/MSDetect-Release"
            target="_blank"
            rel="noopener noreferrer"
            class="text-slate-200 hover:text-sky-400 transition-colors duration-200"
          >
            View on GitHub
          </a>
          <a
            href="#"
            class="text-slate-200 hover:text-sky-400 transition-colors duration-200"
          >
            Contact Us
          </a>
        </nav>

        <!-- Mobile menu button -->
        <button
          @click="isOpen = !isOpen"
          class="md:hidden p-2 rounded-md text-slate-200 hover:text-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-400"
        >
          <i v-if="!isOpen" class="fa-solid fa-bars text-2xl"></i>
          <i v-else class="fa-solid fa-xmark text-2xl"></i>
        </button>
      </div>
    </div>

    <!-- Mobile dropdown menu -->
    <transition
      enter-active-class="transition duration-200 ease-out"
      enter-from-class="transform scale-y-0 opacity-0"
      enter-to-class="transform scale-y-100 opacity-100"
      leave-active-class="transition duration-150 ease-in"
      leave-from-class="transform scale-y-100 opacity-100"
      leave-to-class="transform scale-y-0 opacity-0"
    >
      <div
        v-if="isOpen"
        class="bg-slate-900/90 md:hidden border-t border-white/20 backdrop-blur-xl relative z-50"
      >
        <nav class="px-6 py-4 flex flex-col space-y-3">
          <a
            href="#"
            @click.prevent="handleNavClick"
            class="text-slate-200 hover:text-sky-400 transition-colors duration-200"
          >
            {{ navLinkText }}
          </a>
          <a
            href="#"
            class="text-slate-200 hover:text-sky-400 transition-colors duration-200"
          >
            View on GitHub
          </a>
          <a
            href="#"
            class="text-slate-200 hover:text-sky-400 transition-colors duration-200"
          >
            Contact Us
          </a>
        </nav>
      </div>
    </transition>

    <!-- Dark overlay for mobile menu -->
    <transition
      enter-active-class="transition duration-200"
      enter-from-class="opacity-0"
      enter-to-class="opacity-50"
      leave-active-class="transition duration-150"
      leave-from-class="opacity-50"
      leave-to-class="opacity-0"
    >
      <div
        v-if="isOpen"
        class="fixed top-16 bottom-0 left-0 w-full bg-black/50 z-40 md:hidden"
        @click="isOpen = false"
      ></div>
    </transition>
  </header>
</template>


<script setup>
  import { ref, onMounted, onUnmounted, watch, computed } from "vue";
  import { useRoute, useRouter } from "vue-router";
  import logo from "@/assets/logo.png";

  // State to track if the mobile navbar dropdown is open
  const isOpen = ref(false);
  // State to track is the user has scrolled
  const scrolled = ref(false);

  // Router composables
  const route = useRoute();
  const router = useRouter();

  // Dynamically update nav text and link destinations
  const navLinkText = computed (() =>
    route.path === "/diagnostic-tool" ? "Back to Homepage" : "Try the Analysis Tool"
  );

  const navLinkTo = computed (() =>
    route.path === "/diagnostic-tool" ? "/" : "/diagnostic-tool"
  );

  // Navigation handler
  const handleNavClick = () => {
    router.push(navLinkTo.value);
    isOpen.value = false;
  };

  // Watch for when the mobile navbar dropdown is open to disable scrolling
  watch(isOpen, (open) => {
    document.documentElement.classList.toggle("overflow-hidden", open);
  });

  // Function that will close the mobile navbar dropdown when "Escape" is pressed
  const handleKey = (e) => {
    if (e.key === "Escape") isOpen.value = false;
  };

  // Functon that will set the scrolled state if the window scroll is > 0
  const handleScroll = () => {
    scrolled.value = window.scrollY > 32;
  }

  // Add/remove the event listener for keypress when the component is mounted/unmounted
  onMounted(() => window.addEventListener("keydown", handleKey));
  onUnmounted(() => window.removeEventListener("keydown", handleKey));

  // Add/remove the event listener for scroll when the component is mounted/unmounted
  onMounted(() => window.addEventListener("scroll", handleScroll));
  onUnmounted(() => window.removeEventListener("scroll", handleScroll));
</script>
