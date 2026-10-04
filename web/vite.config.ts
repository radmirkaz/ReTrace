import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Sharing through a tunnel: build with VITE_API_URL=/api and serve with API_PROXY=http://<gpu-host>:8000,
// so the browser reaches the GPU backend through the same address as the site.
const target = process.env.API_PROXY
const proxy = target ? { '/api': { target, changeOrigin: true, rewrite: (p: string) => p.replace(/^\/api/, '') } } : undefined
const allowedHosts = target ? true : undefined

export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173, proxy, allowedHosts },
  preview: { host: true, port: 4173, proxy, allowedHosts },
})
