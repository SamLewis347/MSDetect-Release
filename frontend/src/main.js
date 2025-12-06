// Import statements
import { createApp } from 'vue';
import App from './App.vue';
import router from './router/index.js';
import './assets/main.css';
import '@fortawesome/fontawesome-free/css/all.min.css';

// Create and mount the Vue application
const app = createApp(App);
app.use(router); // Connect the app to the router
app.mount('#app');
