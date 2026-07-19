/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Chef's Loop Peach Theme
        peach: {
          DEFAULT: '#DB7256',
          50: '#FCF5F3',
          100: '#F9E8E4',
          200: '#F2CFC6',
          300: '#EAB0A0',
          400: '#E28B75',
          500: '#DB7256',
          600: '#C75435',
          700: '#A44329',
          800: '#833620',
          900: '#6A2C1A',
        },
        // Accent colors
        cream: '#FFF8F5',
        'cream-dark': '#F5E6E0',
      },
      fontFamily: {
        // Distinctive typography
        display: ['Fraunces', 'Georgia', 'serif'],
        body: ['Outfit', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        shimmer: 'shimmer 1.5s infinite',
        'hint-life-right': 'hintLifeRight 8s ease-in-out both',
        'hint-life-left': 'hintLifeLeft 8s ease-in-out both',
        'nudge-right': 'nudgeRight 1.6s ease-in-out infinite',
        'nudge-left': 'nudgeLeft 1.6s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        // Sweeps a highlight across a skeleton; the element starts at
        // -translate-x-full so the run is edge to edge.
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
        // Whole lifecycle of a tap hint in one pass: slide in from the edge
        // it points at, hold, slide back out. Duration owns the 8s budget.
        hintLifeRight: {
          '0%': { opacity: '0', transform: 'translateX(12px)' },
          '5%, 92%': { opacity: '1', transform: 'translateX(0)' },
          '100%': { opacity: '0', transform: 'translateX(12px)' },
        },
        hintLifeLeft: {
          '0%': { opacity: '0', transform: 'translateX(-12px)' },
          '5%, 92%': { opacity: '1', transform: 'translateX(0)' },
          '100%': { opacity: '0', transform: 'translateX(-12px)' },
        },
        nudgeRight: {
          '0%, 100%': { transform: 'translateX(0)' },
          '50%': { transform: 'translateX(3px)' },
        },
        nudgeLeft: {
          '0%, 100%': { transform: 'translateX(0)' },
          '50%': { transform: 'translateX(-3px)' },
        },
      },
    },
  },
  plugins: [],
}

