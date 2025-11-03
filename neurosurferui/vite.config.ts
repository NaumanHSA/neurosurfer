// import { defineConfig } from 'vite'
// import react from '@vitejs/plugin-react'

// export default defineConfig({
//   plugins: [react()],
//   server: {
//     port: 5173,
//     host: true
//   }
// })

// vite.config.ts
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  // Allow override with VITE_BACKEND_URL (e.g., http://127.0.0.1:8081)
  const target = env.VITE_BACKEND_URL || "http://127.0.0.1:8081";

  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: 5173,
      strictPort: false,
      proxy: {
        // Proxy API calls from the Vite host to the backend
        "^/v1": {
          target,
          changeOrigin: true,
        },
      },
    },
  };
});
