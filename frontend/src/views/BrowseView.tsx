import { ChefIcon, InstallPrompt } from '../components'
import type { BrowseRecipe } from '../config/site'

interface BrowseViewProps {
  /** The site-wide note. Its presence is what put us in this view — see config/site.ts. */
  notice: string
  recipes: ReadonlyArray<BrowseRecipe>
  onSelect: (videoId: string) => void
  /** The recipe currently being fetched from storage, if any. */
  loadingId?: string
  error?: string
}

/**
 * Notice mode's front door: the note, then the recipes already in storage.
 *
 * Deliberately has no URL input. While the pipeline is offline a link box would take
 * a link and fail, which reads worse than not offering it at all.
 */
export function BrowseView({ notice, recipes, onSelect, loadingId, error }: BrowseViewProps) {
  const busy = loadingId !== undefined

  return (
    // Same scroll shape as LandingView: the scroll container is the full-height element
    // and the content sits in a min-h-full wrapper, so nothing gets clipped once the
    // list is taller than the viewport.
    <div className="h-full overflow-y-auto">
      <div className="min-h-full flex flex-col items-center p-6 pb-12">
        {/* Header */}
        <div className="flex flex-col items-center mt-6 mb-8 animate-fade-in">
          <ChefIcon className="w-20 h-20 mb-5" />
          <h1 className="text-3xl md:text-4xl font-display font-semibold text-white text-center">
            Transform videos into interactive guides
          </h1>
        </div>

        {/* The note itself — part of the page, not a dismissible banner. */}
        <div
          className="w-full max-w-md glass-card p-5 mb-6 animate-slide-up"
          style={{ animationDelay: '50ms' }}
        >
          <p className="text-white/90 text-center leading-relaxed">{notice}</p>
        </div>

        {/* Recipe list */}
        <div
          className="w-full max-w-md glass-card p-4 animate-slide-up"
          style={{ animationDelay: '100ms' }}
        >
          <h2 className="text-white/70 text-xs uppercase tracking-wider px-2 pb-3">
            Recipes to try
          </h2>

          <ul className="space-y-2">
            {recipes.map((recipe) => {
              const isLoading = loadingId === recipe.id

              return (
                <li key={recipe.id}>
                  <button
                    type="button"
                    onClick={() => onSelect(recipe.id)}
                    disabled={busy}
                    className="w-full flex items-center gap-3 text-left px-3 py-3 rounded-xl
                               bg-white/10 hover:bg-white/20 active:scale-[0.99]
                               transition-all disabled:opacity-50
                               disabled:hover:bg-white/10 disabled:active:scale-100"
                  >
                    <span className="flex-1 text-white font-medium">{recipe.title}</span>

                    {isLoading ? (
                      <svg className="animate-spin w-5 h-5 shrink-0 text-white" viewBox="0 0 24 24">
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
                    ) : (
                      <svg
                        className="w-5 h-5 shrink-0 text-white/50"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <path d="M9 18l6-6-6-6" />
                      </svg>
                    )}
                  </button>
                </li>
              )
            })}
          </ul>
        </div>

        {error && (
          <div className="w-full max-w-md mt-4 p-4 bg-red-500/20 border border-red-300/30 rounded-xl text-white text-center animate-fade-in">
            {error}
          </div>
        )}
      </div>

      {/* Self-suppressing: renders nothing on desktop, once installed, or once dismissed.
          Mounted here because in notice mode this view replaces Landing as the entry
          point, and it must not be mounted anywhere that can interrupt a cook. */}
      <InstallPrompt />
    </div>
  )
}
