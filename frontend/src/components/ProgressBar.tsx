import { LoadingDots } from './LoadingDots'

interface ProgressBarProps {
  progress: number
  className?: string
}

/**
 * Animated progress bar for loading state
 */
export function ProgressBar({ progress, className = '' }: ProgressBarProps) {
  return (
    <div className={`w-full ${className}`}>
      <div className="w-full h-2 bg-white/20 rounded-full overflow-hidden">
        <div
          className="h-full bg-white rounded-full transition-all duration-500 ease-out"
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
        />
      </div>
      <div className="flex justify-center mt-3">
        <LoadingDots />
      </div>
    </div>
  )
}


