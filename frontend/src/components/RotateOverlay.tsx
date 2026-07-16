import { useOrientation } from '../hooks'

interface RotateOverlayProps {
  children: React.ReactNode
  /**
   * Rendered under the rotate message. Turning the phone shouldn't trap the cook on a
   * screen with no way off it, so cooking mode passes its Exit / View Recipe buttons
   * through to here.
   */
  actions?: React.ReactNode
}

/**
 * Shows "Please rotate your phone" overlay when in portrait mode
 * during cooking mode
 */
export function RotateOverlay({ children, actions }: RotateOverlayProps) {
  const orientation = useOrientation()

  if (orientation === 'landscape') {
    return <>{children}</>
  }

  return (
    <div className="fixed inset-0 bg-peach flex flex-col items-center justify-center p-8 z-50">
      {/* Rotate phone icon */}
      <svg
        className="w-24 h-24 mb-6 animate-pulse"
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Phone outline - tilted */}
        <rect
          x="25"
          y="20"
          width="30"
          height="50"
          rx="4"
          stroke="white"
          strokeWidth="3"
          transform="rotate(-30 40 45)"
        />
        {/* Arrow indicating rotation */}
        <path
          d="M65 35C70 40 72 48 70 55"
          stroke="white"
          strokeWidth="3"
          strokeLinecap="round"
        />
        <path
          d="M70 55L75 50M70 55L65 52"
          stroke="white"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      <h2 className="text-2xl font-display font-semibold text-white text-center mb-2">
        Rotate Your Phone
      </h2>
      <p className="text-white/80 text-center max-w-xs">
        For the best cooking experience, please rotate your device to landscape mode
      </p>

      {actions && <div className="mt-8">{actions}</div>}
    </div>
  )
}


