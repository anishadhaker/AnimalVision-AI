/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      colors: {
        // SaaS Canvas & Neutral Palette (Linear / Vercel-inspired)
        dark: {
          bg: '#0B1020',
          secondary: '#111827',
          card: '#111827',
          elevated: '#1F2937',
          hover: '#1F2937',
          text: '#F8FAFC',
          muted: '#9CA3AF',
          subtle: '#6B7280',
          border: '#374151',
          'border-subtle': 'rgba(255, 255, 255, 0.08)',
          resnet: '#818CF8',
          swin: '#A78BFA',
          accent: '#818CF8',
          'accent-secondary': '#A78BFA',
        },
        light: {
          bg: '#F8FAFC',
          secondary: '#F8FAFC',
          card: '#FFFFFF',
          elevated: '#F1F5F9',
          hover: '#F1F5F9',
          text: '#0F172A',
          muted: '#64748B',
          subtle: '#94A3B8',
          border: '#E2E8F0',
          'border-subtle': 'rgba(0, 0, 0, 0.06)',
          resnet: '#6366F1',
          swin: '#8B5CF6',
          accent: '#6366F1',
          'accent-secondary': '#8B5CF6',
        },
        // Restrained Indigo / Violet / Blue accents
        brand: {
          50: '#EEF2FF',
          100: '#E0E7FF',
          200: '#C7D2FE',
          300: '#A5B4FC',
          400: '#818CF8',
          500: '#6366F1',
          600: '#4F46E5',
          700: '#4338CA',
          800: '#3730A3',
          900: '#312E81',
        },
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'card-light': '0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.04)',
        'card-dark': '0 1px 3px 0 rgba(0, 0, 0, 0.35), 0 1px 2px -1px rgba(0, 0, 0, 0.35)',
        'elevated-light': '0 4px 14px -2px rgba(15, 23, 42, 0.06)',
        'elevated-dark': '0 4px 20px -2px rgba(0, 0, 0, 0.5)',
      },
    },
  },
  plugins: [],
}
