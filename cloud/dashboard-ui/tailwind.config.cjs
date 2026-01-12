module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Space Grotesk", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(15, 23, 42, 0.06), 0 20px 40px -20px rgba(15, 23, 42, 0.25)",
      },
    },
  },
  plugins: [],
};
