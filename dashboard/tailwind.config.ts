import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        fraud: {
          safe: "#22c55e",
          warning: "#f97316",
          danger: "#ef4444",
          critical: "#dc2626",
        },
        surface: {
          50: "#f8fafc",
          100: "#1e293b",
          200: "#1a2332",
          300: "#151d2b",
          400: "#111827",
          500: "#0d1320",
          600: "#0a0f1a",
          700: "#070b14",
          800: "#050810",
          900: "#030509",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 0.3s ease-in-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/forms")],
};

export default config;
