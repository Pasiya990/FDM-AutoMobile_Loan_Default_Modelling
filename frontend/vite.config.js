import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built page (npm run build -> dist/) is served by FastAPI at "/".
// During development (npm run dev) the API calls are forwarded to FastAPI,
// so start the API first: uvicorn backend.app:app --reload
const API = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": API,
      "/schema": API,
      "/predict": API,
      "/examples": API,
    },
  },
});
