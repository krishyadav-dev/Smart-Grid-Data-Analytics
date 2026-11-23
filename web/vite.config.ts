import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: the FastAPI service (python -m src.dashboard) on 8100; /api is proxied.
// Build: web/dist, served by the same service.
export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5173, proxy: { "/api": "http://127.0.0.1:8100" } },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 1800,
    rollupOptions: { output: { manualChunks: { mapbox: ["mapbox-gl"], echarts: ["echarts"] } } },
  },
});
