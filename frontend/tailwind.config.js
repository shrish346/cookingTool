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
      },
    },
  },
  plugins: [],
}

