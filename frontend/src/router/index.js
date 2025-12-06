import { createRouter, createWebHistory } from 'vue-router';

// Import views/components
import HomePage from '@/views/HomePage.vue';
import DiagnosticTool from '@/views/DiagnosticTool.vue';

const routes = [
  {
    path: '/',
    name: 'Home',
    component: HomePage,
  },
  {
    path: '/diagnostic-tool',
    name: 'DiagnosticTool',
    component: DiagnosticTool,
  },
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
});

export default router;
