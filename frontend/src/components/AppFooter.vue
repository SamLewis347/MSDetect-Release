<template>
  <footer class="w-full flex justify-center pb-10 px-4">
    <div class="relative group max-w-5xl w-full px-6 sm:px-8 py-7
                bg-slate-950/70 border border-sky-400/20 rounded-2xl shadow-lg
                transition-all duration-300
                hover:shadow-[0_0_30px_rgba(14,165,233,0.2)]">

      <!-- Glow edge accent -->
      <div class="absolute inset-0 rounded-2xl pointer-events-none
                  opacity-0 group-hover:opacity-100 transition-opacity duration-300
                  bg-gradient-to-br from-sky-400/20 to-transparent"></div>

      <div class="relative z-10 space-y-5">
        <!-- Logo -->
        <div class="text-center">
          <p class="font-extrabold text-white text-xl mb-1">
            MS<span class="text-sky-400">Detect</span>
          </p>
          <p class="text-slate-400 text-sm">Senior Project · Florida Southern College</p>
        </div>

        <!-- Contact Links -->
        <div class="border-t border-white/5 pt-5">
          <p class="text-slate-400 text-xs text-center mb-3">Contact the Team</p>
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 max-w-2xl mx-auto">
            <a href="mailto:tmayfield@flsouthern.edu"
               class="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-slate-900/40 border border-white/10
                      text-slate-300 hover:text-sky-400 hover:border-sky-400/40 hover:bg-slate-800/60
                      transition-all duration-300 group/email">
              <i class="fa-solid fa-envelope text-sky-400 group-hover/email:scale-110 transition-transform"></i>
              <span class="text-sm truncate">tuckmayf@gmail.com</span>
            </a>

            <a href="mailto:slewis@flsouthern.edu"
               class="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-slate-900/40 border border-white/10
                      text-slate-300 hover:text-sky-400 hover:border-sky-400/40 hover:bg-slate-800/60
                      transition-all duration-300 group/email">
              <i class="fa-solid fa-envelope text-sky-400 group-hover/email:scale-110 transition-transform"></i>
              <span class="text-sm truncate">samuellewis92@icloud.com</span>
            </a>

            <a href="mailto:dbenson@flsouthern.edu"
               class="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-slate-900/40 border border-white/10
                      text-slate-300 hover:text-sky-400 hover:border-sky-400/40 hover:bg-slate-800/60
                      transition-all duration-300 group/email">
              <i class="fa-solid fa-envelope text-sky-400 group-hover/email:scale-110 transition-transform"></i>
              <span class="text-sm truncate">dbensonposts@gmail.com</span>
            </a>
          </div>
        </div>

        <!-- Copyright -->
        <div class="text-center border-t border-white/5 pt-4">
          <p class="text-slate-500 text-xs">
            &copy; 2025 MS Detect Team. Prototype for academic use only.
          </p>
        </div>
      </div>
    </div>
  </footer>
</template>
