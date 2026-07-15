import { useEffect, useState } from 'react'
import { ProgressBar } from '../components'

interface LoadingViewProps {
  progress: number
  message: string
}

// Expected total wall-clock time for a fresh recipe, and how long each backend
// progress checkpoint is expected to have taken by the time it's reported. Backend
// progress is stage-based, not linear (see backend/api/routes.py), so we anchor the
// ETA to these checkpoints rather than extrapolating from the raw percentage. Retune
// after watching real runs — nothing else depends on these numbers.
const TOTAL_SECONDS = 120
// [progressCheckpoint, expectedElapsedSecondsAtThatCheckpoint]
const STAGE_SCHEDULE: [number, number][] = [
  [5, 0],
  [10, 8],
  [25, 22],
  [50, 45],
  [62, 80],
  [72, 105],
  [100, 120],
]

/** Expected elapsed seconds at the checkpoint at or just below `progress`. */
function expectedElapsedFor(progress: number): number {
  let elapsed = 0
  for (const [checkpoint, seconds] of STAGE_SCHEDULE) {
    if (progress >= checkpoint) elapsed = seconds
    else break
  }
  return elapsed
}

/** Expected elapsed seconds at the next checkpoint after `progress` (the stage cap). */
function nextExpectedElapsedFor(progress: number): number {
  for (const [checkpoint, seconds] of STAGE_SCHEDULE) {
    if (progress < checkpoint) return seconds
  }
  return TOTAL_SECONDS
}

function formatDuration(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds))
  const minutes = Math.floor(s / 60)
  const seconds = s % 60
  if (minutes === 0) return `${seconds}s`
  return `${minutes}m ${seconds}s`
}

/**
 * Loading view with progress bar and a stage-aware estimated time remaining.
 */
export function LoadingView({ progress, message }: LoadingViewProps) {
  const [startTime] = useState(() => Date.now())
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsed((Date.now() - startTime) / 1000)
    }, 1000)
    return () => clearInterval(timer)
  }, [startTime])

  // Clamp wall-clock elapsed into the window the backend has actually reached, so the
  // estimate ticks down smoothly but never runs past the reported stage. When the
  // backend advances a stage, the window jumps forward and the ETA drops with it.
  const lo = expectedElapsedFor(progress)
  const hi = nextExpectedElapsedFor(progress) - 1
  const clampedElapsed = Math.min(Math.max(elapsed, lo), hi)
  const remaining = TOTAL_SECONDS - clampedElapsed

  const overrun = elapsed >= TOTAL_SECONDS
  const etaLabel = overrun
    ? `Working for ${formatDuration(elapsed)}…`
    : `About ${formatDuration(remaining)} remaining`

  return (
    <div className="h-full overflow-y-auto flex flex-col items-center justify-center p-6">
      {/* Status message */}
      <div className="text-center mb-8 animate-fade-in">
        <h2 className="text-2xl font-display font-semibold text-white mb-2">
          Link found!
        </h2>
        <p className="text-white/80">
          {message || 'Generating your recipe…'}
        </p>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-sm animate-slide-up" style={{ animationDelay: '200ms' }}>
        <ProgressBar progress={progress} />
      </div>

      {/* Estimated time remaining */}
      <p className="mt-4 text-white/60 text-sm animate-fade-in">{etaLabel}</p>

      {/* Reassurance */}
      <p className="mt-10 text-white/50 text-sm text-center animate-fade-in">
        Feel free to swipe out of the app and check back later.
      </p>
    </div>
  )
}
