import { useState } from 'react'
import { ChefIcon } from '../components'

interface LandingViewProps {
  onSubmit: (url: string) => void
  isValidating: boolean
  error?: string
}

/**
 * Landing view with URL input
 */
export function LandingView({ onSubmit, isValidating, error }: LandingViewProps) {
  const [url, setUrl] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (url.trim() && !isValidating) {
      onSubmit(url.trim())
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      {/* Logo and Title */}
      <div className="flex flex-col items-center mb-12 animate-fade-in">
        <ChefIcon className="w-20 h-20 text-white mb-6" />
        <h1 className="text-4xl md:text-5xl font-display font-semibold text-white text-center mb-3">
          Transform Your Cooking Videos
        </h1>
        <p className="text-white/80 text-center text-lg">
          Paste a YouTube or TikTok link to get started
        </p>
      </div>

      {/* Input Form */}
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-md animate-slide-up"
        style={{ animationDelay: '100ms' }}
      >
        <div className="glass-card p-6 space-y-4">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            className="input-field"
            disabled={isValidating}
            autoFocus
          />

          <button
            type="submit"
            disabled={!url.trim() || isValidating}
            className="btn-primary w-full"
          >
            {isValidating ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Validating...
              </span>
            ) : (
              'Analyze Recipe'
            )}
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mt-4 p-4 bg-red-500/20 border border-red-300/30 rounded-xl text-white text-center animate-fade-in">
            {error}
          </div>
        )}
      </form>
    </div>
  )
}


