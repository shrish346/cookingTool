import { ProgressBar, AdPlaceholder } from '../components'

interface LoadingViewProps {
  progress: number
  message: string
}

/**
 * Loading view with progress bar and ad placeholder
 */
export function LoadingView({ progress, message }: LoadingViewProps) {
  return (
    <div className="h-full overflow-y-auto flex flex-col items-center justify-center p-6">
      {/* Status message */}
      <div className="text-center mb-8 animate-fade-in">
        <h2 className="text-2xl font-display font-semibold text-white mb-2">
          Link found!
        </h2>
        <p className="text-white/80">
          {message || 'Generating recipe. Please allow up to 1 minute'}
        </p>
      </div>

      {/* Ad placeholder */}
      <div className="mb-8 animate-slide-up" style={{ animationDelay: '100ms' }}>
        <AdPlaceholder />
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-sm animate-slide-up" style={{ animationDelay: '200ms' }}>
        <ProgressBar progress={progress} />
      </div>
    </div>
  )
}


