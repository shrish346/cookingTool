interface LoadingDotsProps {
  className?: string
}

/**
 * Three staggered pulsing dots. Used on the loading screen (under the progress bar) and
 * in cooking mode while the user is scrubbing between steps.
 */
export function LoadingDots({ className = '' }: LoadingDotsProps) {
  return (
    <div className={`flex gap-2 ${className}`}>
      <span className="w-2 h-2 bg-white/60 rounded-full animate-pulse" style={{ animationDelay: '0ms' }} />
      <span className="w-2 h-2 bg-white/60 rounded-full animate-pulse" style={{ animationDelay: '200ms' }} />
      <span className="w-2 h-2 bg-white/60 rounded-full animate-pulse" style={{ animationDelay: '400ms' }} />
    </div>
  )
}
