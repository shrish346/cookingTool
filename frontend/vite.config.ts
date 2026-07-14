import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['apple-touch-icon.png', 'favicon-32x32.png', 'favicon-16x16.png'],
      manifest: {
        name: 'MakerAI',
        short_name: 'MakerAI',
        description: 'Transform cooking videos into step-by-step recipes',
        theme_color: '#DB7256',
        background_color: '#DB7256',
        display: 'standalone',
        orientation: 'any',
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png'
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        runtimeCaching: [
          {
            // Step clips, served cross-origin from the R2 Worker.
            //
            // `rangeRequests` is the whole point. A <video> asks for byte ranges and
            // wants a 206 back; a plain fetch()-warmed HTTP cache entry is a full 200.
            // Chrome will slice a 206 out of that cached 200, but WebKit will not — so
            // on iOS every clip was re-downloaded from the network no matter how
            // eagerly we prefetched. This plugin stores the full response in Cache
            // Storage and builds the 206 in JS, which doesn't depend on the browser's
            // media-cache behaviour at all.
            //
            // Requires the <video> to be crossOrigin="anonymous", or the response is
            // opaque and can't be cached or sliced.
            urlPattern: /\.mp4$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'recipe-clips',
              rangeRequests: true,
              cacheableResponse: { statuses: [200] },
              expiration: {
                maxEntries: 60,
                maxAgeSeconds: 60 * 60 * 24 * 30 // 30 days
              }
            }
          }
        ]
      }
    })
  ],
  server: {
    port: 5173,
    host: true,
    // Permit cloudflared quick-tunnel hosts (dev-tunnel.sh). Leading dot = any
    // subdomain. Localhost and LAN IPs are allowed by default, so this only
    // matters when serving through a tunnel; harmless otherwise.
    allowedHosts: ['.trycloudflare.com']
  }
})

