import { useEffect, useState } from 'react'
import { useInstallPrompt } from '../hooks'
import type { InstallMode, InstallPlatform } from '../hooks'
import { ConfirmDialog } from './ConfirmDialog'

/** How long the arrow stays up before fading out on its own. */
const HINT_MS = 6000

/** Where the browser control we're pointing at actually is. */
type HintAnchor = 'bottom' | 'top-right'

function instructionFor(platform: InstallPlatform, mode: InstallMode): string {
  if (mode === 'in-app') {
    // An embedded webview has no Add to Home Screen at all, so the only useful
    // instruction is how to get out of it.
    return platform === 'ios'
      ? 'Tap the ⋯ button and choose "Open in Safari" first.'
      : 'Tap the ⋮ button and choose "Open in Chrome" first.'
  }

  if (mode === 'native') return "Tap Ok and we'll add it to your home screen."

  return platform === 'ios'
    ? 'Tap the Share button below, then choose "Add to Home Screen".'
    : 'Tap the ⋮ menu, then "Install app".'
}

/**
 * The arrow left behind after Ok on the platforms we can't install for. Purely a pointer
 * at the browser's own chrome, so it never takes clicks.
 */
function ShareHint({ anchor, label }: { anchor: HintAnchor; label: string }) {
  const bottom = anchor === 'bottom'

  return (
    <div
      className={`fixed z-[70] pointer-events-none flex flex-col items-center animate-fade-in ${
        bottom ? 'bottom-2 left-0 right-0' : 'top-2 right-4'
      }`}
    >
      <div className={`flex flex-col items-center ${bottom ? '' : 'order-last'}`}>
        <span className="px-3 py-1.5 rounded-lg bg-white text-peach text-sm font-medium shadow-lg max-w-[16rem] text-center">
          {label}
        </span>
        <span
          className={`text-white text-2xl leading-none animate-bounce ${bottom ? 'mt-1' : 'mb-1'}`}
        >
          {bottom ? '↓' : '↑'}
        </span>
      </div>
    </div>
  )
}

/**
 * First-visit nudge to install the app to the home screen.
 *
 * Cooking mode is a fixed, full-screen landscape surface, and a browser URL bar eats the
 * room it needs - but nothing otherwise tells anyone that installing is an option. The
 * hook decides whether to ask at all (phone only, not already installed, not dismissed);
 * this decides what to say and what Ok can actually do.
 *
 * Android gets a real one-tap install via the browser's install event. iOS has no such
 * API and never has, so Ok can only point at Safari's Share button - which is why the
 * copy and the Ok behavior are per-platform rather than shared.
 */
export function InstallPrompt() {
  const { shouldShow, platform, mode, promptInstall, dismiss } = useInstallPrompt()
  const [hint, setHint] = useState<HintAnchor | null>(null)

  useEffect(() => {
    if (!hint) return
    const timer = setTimeout(() => setHint(null), HINT_MS)
    return () => clearTimeout(timer)
  }, [hint])

  const handleOk = async () => {
    // Dismiss either way: declining the browser's own install dialog is still an answer,
    // and re-asking after that is nagging.
    if (mode === 'native' && (await promptInstall())) {
      dismiss()
      return
    }

    dismiss()

    // Nothing to point at from inside a webview - the instruction was to leave it.
    if (mode === 'in-app') return

    setHint(platform === 'ios' ? 'bottom' : 'top-right')
  }

  if (!shouldShow) {
    return hint ? (
      <ShareHint
        anchor={hint}
        label={
          platform === 'ios'
            ? 'Tap Share, then "Add to Home Screen"'
            : 'Tap the menu, then "Install app"'
        }
      />
    ) : null
  }

  return (
    <ConfirmDialog
      confirmLabel="Ok"
      cancelLabel="Ignore"
      onConfirm={handleOk}
      onCancel={dismiss}
      message={
        <>
          <p>
            Hey! This might be your first time using this, but we built this to be viewed
            in full-screen, just like an app! You're on{' '}
            <span className="font-medium">{platform === 'ios' ? 'iOS' : 'Android'}</span>.
          </p>
          <p className="mt-3 text-white/80 text-sm">{instructionFor(platform, mode)}</p>
        </>
      }
    />
  )
}
