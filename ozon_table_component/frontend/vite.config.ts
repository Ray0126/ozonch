import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",          // <-- КРИТИЧНО для Streamlit Components
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
