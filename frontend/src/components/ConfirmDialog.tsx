interface ConfirmDialogProps {
  message: React.ReactNode
  confirmLabel?: string
  cancelLabel?: string
  onConfirm: () => void
  onCancel: () => void
}

/**
 * Small yes/no box over whatever is on screen. Every way out of a recipe throws away
 * the recipe and the saved cooking position, so each one asks first.
 *
 * z-[60] puts it above RotateOverlay (z-50) - the exit buttons are reachable from the
 * portrait rotate screen too, so the dialog has to clear it.
 */
export function ConfirmDialog({
  message,
  confirmLabel = 'Yes',
  cancelLabel = 'No',
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-6"
      // The cooking view navigates on taps to the left/right of the screen. Without this
      // a tap anywhere on the dialog would step the recipe underneath it.
      onClick={(e) => {
        e.stopPropagation()
        onCancel()
      }}
    >
      <div
        className="glass-card w-full max-w-xs p-5 animate-slide-up"
        onClick={(e) => e.stopPropagation()}
      >
        {/* A div, not a p: callers pass block content (see InstallPrompt), which is
            invalid nested inside a paragraph. */}
        <div className="text-white text-center mb-5">{message}</div>
        <div className="flex gap-3">
          <button
            onClick={onCancel}
            className="flex-1 py-2 rounded-xl bg-white/10 text-white hover:bg-white/20 transition-colors"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 py-2 rounded-xl bg-white text-peach font-medium hover:bg-white/90 transition-colors"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
