import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
  server: {
    proxy: {
      '/api/textcraft': {
        target: 'http://localhost:36001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/textcraft/, '')
      },
      '/api/babyai': {
        target: 'http://localhost:36002',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/babyai/, '')
      },
      '/api/sciworld': {
        target: 'http://localhost:36003',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/sciworld/, '')
      },
      '/api/webarena': {
        target: 'http://localhost:36004',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/webarena/, '')
      },
      '/api/searchqa': {
        target: 'http://localhost:36005',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/searchqa/, '')
      }
    }
  }
})
