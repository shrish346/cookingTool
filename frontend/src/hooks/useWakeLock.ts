import { useEffect } from 'react'

/**
 * Keeps the screen awake while the calling component is mounted.
 * Scoped by mount/unmount: mount it in CookingView and the screen only
 * stays on during cooking mode. No-ops silently where unsupported
 * (iOS Safari < 16.4) or when the OS refuses (e.g. battery saver).
 */
export function useWakeLock(): void {
  useEffect(() => {
    if (!('wakeLock' in navigator)) return

    let sentinel: WakeLockSentinel | null = null
    let cancelled = false

    const request = async () => {
      try {
        const lock = await navigator.wakeLock.request('screen')
        if (cancelled) {
          // Unmounted while the request was in flight; don't leak the lock.
          lock.release().catch(() => {})
          return
        }
        sentinel = lock
      } catch {
        // Refused (battery saver, hidden tab). Screen dims as normal.
      }
    }

    // The OS auto-releases the lock whenever the page is hidden (app switch,
    // notification shade, screen off). Re-acquire on return.
    const handleVisibility = () => {
      if (document.visibilityState === 'visible') request()
    }

    request()
    document.addEventListener('visibilitychange', handleVisibility)

    return () => {
      cancelled = true
      document.removeEventListener('visibilitychange', handleVisibility)
      sentinel?.release().catch(() => {})
      sentinel = null
    }
  }, [])
}
