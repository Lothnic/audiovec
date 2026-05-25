import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#0B0E14",
          card: "#141A24",
          border: "#1E293B",
          hover: "#1A212D",
        },
        accent: {
          purple: "#A78BFA",
          pink: "#EC4899",
          amber: "#F59E0B",
        },
        text: {
          primary: "#F1F5F9",
          secondary: "#CBD5E1",
          muted: "#8B95A8",
          dim: "#64748B",
        },
      },
      fontFamily: {
        mono: [
          "SF Mono", "Fira Code", "Fira Mono", "Roboto Mono",
          "ui-monospace", "monospace",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
