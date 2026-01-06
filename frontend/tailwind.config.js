/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Core palette
        void: 'rgb(var(--color-void) / <alpha-value>)',
        surface: {
          DEFAULT: 'rgb(var(--color-surface) / <alpha-value>)',
          elevated: 'rgb(var(--color-surface-elevated) / <alpha-value>)',
          hover: 'rgb(var(--color-surface-hover) / <alpha-value>)',
        },
        border: {
          DEFAULT: 'rgb(var(--color-border) / <alpha-value>)',
          glow: 'rgb(var(--color-border-glow) / <alpha-value>)',
        },
        // Text colors
        text: {
          primary: 'rgb(var(--color-text-primary) / <alpha-value>)',
          secondary: 'rgb(var(--color-text-secondary) / <alpha-value>)',
          muted: 'rgb(var(--color-text-muted) / <alpha-value>)',
        },
        // Accent colors
        cyan: {
          DEFAULT: 'rgb(var(--color-cyan) / <alpha-value>)',
          dim: 'rgb(var(--color-cyan-dim) / <alpha-value>)',
        },
        amber: {
          DEFAULT: 'rgb(var(--color-amber) / <alpha-value>)',
          dim: 'rgb(var(--color-amber-dim) / <alpha-value>)',
        },
        emerald: 'rgb(var(--color-emerald) / <alpha-value>)',
        rose: 'rgb(var(--color-rose) / <alpha-value>)',
        // Alpha signals
        alpha: {
          buy: 'rgb(var(--color-alpha-buy) / <alpha-value>)',
          sell: 'rgb(var(--color-alpha-sell) / <alpha-value>)',
        },
      },
      fontFamily: {
        display: ['var(--font-syne)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-jetbrains)', 'ui-monospace', 'monospace'],
      },
      animation: {
        'fade-in': 'fade-in 0.5s ease-out forwards',
        'slide-in': 'slide-in-left 0.4s ease-out forwards',
        'glow-pulse': 'glow-pulse 2s ease-in-out infinite',
        'live-ping': 'live-ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite',
      },
      boxShadow: {
        'glow-cyan': '0 0 20px rgba(0, 212, 255, 0.4), 0 0 40px rgba(0, 212, 255, 0.2)',
        'glow-amber': '0 0 20px rgba(255, 191, 0, 0.4), 0 0 40px rgba(255, 191, 0, 0.2)',
        'glow-emerald': '0 0 20px rgba(16, 185, 129, 0.4), 0 0 40px rgba(16, 185, 129, 0.2)',
        'glow-alpha-buy': '0 0 30px rgba(34, 197, 94, 0.3)',
        'glow-alpha-sell': '0 0 30px rgba(239, 68, 68, 0.3)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [],
}
