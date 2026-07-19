import { useInstallPrompt } from '../hooks'
import type { InstallMode, InstallPlatform } from '../hooks'
import { ConfirmDialog } from './ConfirmDialog'

function instructionFor(platform: InstallPlatform, mode: InstallMode): string {
  // A computer can't add anything to a home screen, so the only useful instruction is to
  // move to a phone.
  if (platform === 'other') {
    return 'Open this on your phone and add it to your home screen for the full experience.'
  }

  if (mode === 'in-app') {
    // An embedded webview has no Add to Home Screen at all, so the only useful
    // instruction is how to get out of it.
    return 'Tap the ⋯ button and open in your browser first, then add to home screen.'
  }

  if (mode === 'native') return "Tap Ok and we'll add it to your home screen."

  return platform === 'ios'
    ? 'Tap the Share button, then choose "Add to Home Screen".'
    : 'Tap the ⋮ menu, then "Install app".'
}

/**
 * Once-per-session nudge about how the app is meant to be used.
 *
 * Cooking mode is a fixed, full-screen landscape surface, and a browser URL bar eats the
 * room it needs - but nothing otherwise tells anyone that installing is an option. The
 * hook decides whether to ask at all (not already installed, not dismissed this session);
 * this decides what to say and what Ok can actually do.
 *
 * On a computer it's a warning to switch to a phone. On a phone it's an install nudge:
 * Android gets a real one-tap install via the browser's install event, while iOS has no
 * such API and never has, so its copy just tells the user where the browser's own control
 * is - which is why the instruction is per-platform.
 */
export function InstallPrompt() {
  const { shouldShow, platform, mode, promptInstall, dismiss } = useInstallPrompt()

  const handleOk = async () => {
    // Dismiss either way: declining the browser's own install dialog is still an answer.
    if (mode === 'native') await promptInstall()
    dismiss()
  }

  if (!shouldShow) return null

  return (
    <ConfirmDialog
      confirmLabel="Ok"
      cancelLabel="Ignore"
      onConfirm={handleOk}
      onCancel={dismiss}
      message={
        platform === 'other' ? (
          <>
            <p>
              Hey! We noticed you're on a <span className="font-medium">computer</span>. Maker AI is best when viewed on
              your phone.
            </p>
            <p className="mt-3 text-white/80 text-sm">{instructionFor(platform, mode)}</p>
          </>
        ) : (
          <>
            <p>
              Hey! We noticed you're on{' '}
              <span className="font-medium">{platform === 'ios' ? 'iOS' : 'Android'}</span>,
              but on the web. Add this to your homescreen for a better experience, just like
              an app.
            </p>
            <p className="mt-3 text-white/80 text-sm">{instructionFor(platform, mode)}</p>
          </>
        )
      }
    />
  )
}
