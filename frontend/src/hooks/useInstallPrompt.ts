import { useCallback, useEffect, useRef, useState } from 'react'

export type InstallPlatform = 'ios' | 'android' | 'other'

/**
 * How the user can actually get to a home screen icon from where they are:
 * - `native`      Chromium handed us a real install event; Ok installs it outright.
 * - `instructions` A real mobile browser, but no install API (all of iOS). Ok can only
 *                  point them at the browser's own control.
 * - `in-app`      An embedded webview (opened from Instagram/TikTok). There is no
 *                  Add to Home Screen here at all - they have to leave first.
 */
export type InstallMode = 'native' | 'instructions' | 'in-app'

/**
 * Non-standard and Chromium-only, so it isn't in TypeScript's DOM lib. Safari has never
 * implemented this or any equivalent, which is why iOS can only ever be `instructions`.
 */
interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

/**
 * Where "already answered this session" is remembered. sessionStorage, not localStorage,
 * on purpose: it's scoped to the tab's session, so a reload within the same visit stays
 * quiet but a fresh visit later brings the reminder back.
 */
const SESSION_DISMISSED_KEY = 'chefs-loop-install-dismissed'

function readSessionDismissed(): boolean {
  if (typeof sessionStorage === 'undefined') return false
  try {
    return sessionStorage.getItem(SESSION_DISMISSED_KEY) === '1'
  } catch {
    return false
  }
}

/** UA tokens of the embedded browsers our traffic actually arrives through. */
const IN_APP_TOKENS =
  /Instagram|FBAN|FBAV|FB_IAB|TikTok|BytedanceWebview|Snapchat|Twitter|LinkedInApp|Pinterest/i

/**
 * `?install=ios|android|in-app` forces the prompt on, whatever the device says.
 *
 * The real thing is unreachable from a desktop browser - it keys off the user agent and
 * off browser install APIs that only exist on a phone - so without this the copy and
 * layout can only be checked by deploying and picking up a handset. Mirrors the existing
 * `?recipe=<video_id>` debug hatch in `src/api/storedRecipe.ts`.
 */
function forcedOverride(): { platform: InstallPlatform; inApp: boolean } | null {
  if (typeof window === 'undefined') return null
  const forced = new URLSearchParams(window.location.search).get('install')
  if (!forced) return null

  if (forced === 'ios') return { platform: 'ios', inApp: false }
  if (forced === 'android') return { platform: 'android', inApp: false }
  if (forced === 'in-app') return { platform: 'ios', inApp: true }
  return null
}

function detectPlatform(): InstallPlatform {
  const forced = forcedOverride()
  if (forced) return forced.platform

  if (typeof navigator === 'undefined') return 'other'
  const ua = navigator.userAgent

  // An iPad on iPadOS 13+ reports itself as a Mac, so the UA alone misses it. A Mac with
  // a touchscreen doesn't exist, so maxTouchPoints is what separates the two.
  const isIPadOS = navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1
  if (/iPhone|iPad|iPod/i.test(ua) || isIPadOS) return 'ios'
  if (/Android/i.test(ua)) return 'android'
  return 'other'
}

/** Already added to the home screen and launched from there - nothing left to ask for. */
function isInstalled(): boolean {
  if (typeof window === 'undefined') return false
  if (forcedOverride()) return false
  const standalone = window.matchMedia?.('(display-mode: standalone)').matches
  // iOS Safari never reports display-mode: standalone; it has its own flag instead.
  const iosStandalone = (navigator as { standalone?: boolean }).standalone === true
  return Boolean(standalone) || iosStandalone
}

/**
 * Decides whether to offer "add this to your home screen", and how.
 *
 * The app is built to be full-screen - cooking mode is a fixed landscape surface - but in
 * a browser tab the URL bar eats that. Nothing else in the app tells anyone installing is
 * even possible.
 *
 * Shown once per browser session: on desktop it warns the visitor to switch to a phone,
 * on a phone it nudges them to install. Suppressed only when already installed. Dismissal
 * lives in sessionStorage, so it reappears on the next visit but not on a same-session
 * reload.
 */
export function useInstallPrompt() {
  const [platform] = useState<InstallPlatform>(detectPlatform)
  const [installed, setInstalled] = useState(isInstalled)
  // Suppresses the dialog for the rest of this browser session. Seeded from sessionStorage
  // so a reload after "Ignore"/"Ok" stays quiet; a new session starts fresh.
  const [hidden, setHidden] = useState(readSessionDismissed)

  // Held in a ref, not state: it's a one-shot handle to fire on Ok, not something the UI
  // renders. `nativeReady` is the render-visible half.
  const deferredRef = useRef<BeforeInstallPromptEvent | null>(null)
  const [nativeReady, setNativeReady] = useState(false)

  useEffect(() => {
    // Chromium fires this only once it considers the app installable (manifest, icons,
    // service worker, https). Default-prevented so the browser holds its own mini-infobar
    // and we can open the real dialog from our Ok button instead.
    const onBeforeInstall = (e: Event) => {
      e.preventDefault()
      deferredRef.current = e as BeforeInstallPromptEvent
      setNativeReady(true)
    }

    // Covers installs that happen outside our dialog (the browser's own menu), so the
    // prompt can't linger for someone who has already done it.
    const onInstalled = () => setInstalled(true)

    window.addEventListener('beforeinstallprompt', onBeforeInstall)
    window.addEventListener('appinstalled', onInstalled)
    return () => {
      window.removeEventListener('beforeinstallprompt', onBeforeInstall)
      window.removeEventListener('appinstalled', onInstalled)
    }
  }, [])

  const forced = forcedOverride()
  const inApp = forced
    ? forced.inApp
    : typeof navigator !== 'undefined' && IN_APP_TOKENS.test(navigator.userAgent)

  // Desktop never gets 'native': there's no home screen to install to, so a computer only
  // ever sees the "switch to your phone" warning, not an install button.
  const mode: InstallMode =
    inApp ? 'in-app' : platform !== 'other' && nativeReady ? 'native' : 'instructions'

  const shouldShow = !hidden && (Boolean(forced) || !installed)

  const dismiss = useCallback(() => {
    setHidden(true)
    try {
      sessionStorage.setItem(SESSION_DISMISSED_KEY, '1')
    } catch {
      // sessionStorage can throw in private modes; the in-memory `hidden` still holds.
    }
  }, [])

  /**
   * Opens the browser's real install dialog. Resolves false when there was no event to
   * fire, which is the caller's cue to fall back to instructions.
   *
   * Dismissal is the caller's job, not ours - declining the native dialog is still an
   * answer, and we don't ask twice.
   */
  const promptInstall = useCallback(async (): Promise<boolean> => {
    const deferred = deferredRef.current
    if (!deferred) return false

    // Single-use: the browser won't accept the same event twice.
    deferredRef.current = null
    setNativeReady(false)

    try {
      await deferred.prompt()
      await deferred.userChoice
      return true
    } catch {
      return false
    }
  }, [])

  return { shouldShow, platform, mode, promptInstall, dismiss }
}
