import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// The studio talks only to the neurosurfer Phase-2 gateway. In dev we proxy
// /v1/* to it so the browser has no CORS/token concerns during local work.
const GATEWAY = process.env.NEUROSURFER_GATEWAY || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5273,
    proxy: {
      "/v1": {
        target: GATEWAY,
        changeOrigin: true,
        // SSE needs an un-buffered, long-lived connection.
        ws: false,
      },
    },
  },
});
