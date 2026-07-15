import type { Provenance, Recipe, Source } from '../types'

interface WarmProgress {
  done: number
  total: number
}

interface RecipeViewProps {
  recipe: Recipe
  /** The server has rendered and uploaded the clips. */
  clipsReady: boolean
  /** Those clips are now downloaded on this device. */
  clipsWarm: boolean
  warmed: WarmProgress
  hasStartedCooking: boolean
  onStartCooking: () => void
  onRestart: () => void
}

/**
 * Cooking stays shut until the clips are on the device. Someone who clicks straight
 * through a still-downloading recipe gets a blank box on every step, which reads as
 * broken - better to say what we're waiting on and let them read the ingredients.
 */
function cookingButtonLabel({
  clipsReady,
  clipsWarm,
  hasStartedCooking,
}: Pick<RecipeViewProps, 'clipsReady' | 'clipsWarm' | 'hasStartedCooking'>): string {
  if (!clipsReady) return 'Preparing video clips...'

  if (!clipsWarm) {
    return 'Preparing Guide...'
  }

  return hasStartedCooking ? 'Resume Cooking' : 'Start Cooking'
}

/**
 * Says where a quantity came from. The video rarely states every amount, so the
 * expansion pass fills the gaps - this is what keeps that honest rather than
 * silently presenting a guess as something the video showed.
 */
function ProvenanceBadge({
  provenance,
  sources,
  sourceId,
}: {
  provenance: Provenance
  sources?: Source[]
  sourceId?: string
}) {
  const source: Source | undefined = sourceId
    ? sources?.find((s) => s.id === sourceId)
    : undefined

  const label =
    provenance === 'video'
      ? 'seen in video'
      : provenance === 'reference'
        ? `from ${source?.site ?? source?.title ?? 'a published recipe'}`
        : 'our estimate'

  const tone =
    provenance === 'video'
      ? 'bg-white/15 text-white/80'
      : provenance === 'reference'
        ? 'bg-white/10 text-white/60'
        : 'bg-white/5 text-white/50'

  return (
    <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wide ${tone}`}>
      {label}
    </span>
  )
}

/**
 * Recipe overview view with ingredients and steps
 */
export function RecipeView({
  recipe,
  clipsReady,
  clipsWarm,
  hasStartedCooking,
  onStartCooking,
  onRestart,
}: RecipeViewProps) {
  return (
    <div className="h-full overflow-y-auto pb-32 scroll-smooth">
      {/* Header */}
      <div className="p-6 pt-8">
        <h1 className="text-3xl md:text-4xl font-display font-bold text-white mb-2 animate-fade-in">
          {recipe.title}
        </h1>
        {recipe.description && (
          <p className="text-white/80 animate-fade-in" style={{ animationDelay: '50ms' }}>
            {recipe.description}
          </p>
        )}
      </div>

      {/* Time and Servings */}
      <div
        className="flex justify-start gap-4 px-6 mb-6 animate-slide-up"
        style={{ animationDelay: '100ms' }}
      >
        {recipe.prep_time_minutes && (
          <div className="glass-card px-4 py-3 text-center">
            <div className="text-white/60 text-xs mb-1 flex items-center justify-center gap-1">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <circle cx="12" cy="12" r="10" strokeWidth="2" />
                <path strokeWidth="2" d="M12 6v6l4 2" />
              </svg>
              Prep Time
            </div>
            <div className="text-white font-medium">{recipe.prep_time_minutes} mins</div>
          </div>
        )}
        {recipe.cook_time_minutes && (
          <div className="glass-card px-4 py-3 text-center">
            <div className="text-white/60 text-xs mb-1 flex items-center justify-center gap-1">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeWidth="2" d="M12 8v4l2 2m-2-6a8 8 0 100 16 8 8 0 000-16z" />
              </svg>
              Cook Time
            </div>
            <div className="text-white font-medium">{recipe.cook_time_minutes} mins</div>
          </div>
        )}
        <div className="glass-card px-4 py-3 text-center">
          <div className="text-white/60 text-xs mb-1 flex items-center justify-center gap-1">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Servings
          </div>
          <div className="text-white font-medium">{recipe.servings}</div>
        </div>
      </div>

      {/* Tools */}
      {recipe.tools?.length > 0 && (
        <div className="px-6 mb-6 animate-slide-up" style={{ animationDelay: '130ms' }}>
          <h2 className="text-xl font-display font-semibold text-white mb-3">Tools</h2>
          <div className="glass-card p-4">
            <ul className="space-y-2">
              {recipe.tools.map((tool) => (
                <li key={tool.id} className="flex justify-between items-start gap-3 text-white">
                  <span>
                    {tool.name}
                    {tool.substitute && (
                      <span className="block text-white/50 text-sm">No {tool.name.toLowerCase()}? {tool.substitute}</span>
                    )}
                  </span>
                  <ProvenanceBadge provenance={tool.provenance} sources={recipe.sources} sourceId={tool.source_id} />
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Ingredients */}
      <div className="px-6 mb-6 animate-slide-up" style={{ animationDelay: '150ms' }}>
        <h2 className="text-xl font-display font-semibold text-white mb-3">Ingredients</h2>
        <div className="glass-card p-4">
          <ul className="space-y-3">
            {recipe.ingredients.map((ing) => (
              <li key={ing.id} className="text-white">
                <div className="flex justify-between gap-3">
                  <span>
                    {ing.name}
                    {ing.preparation && (
                      <span className="text-white/60 text-sm ml-1">({ing.preparation})</span>
                    )}
                    {ing.optional && (
                      <span className="text-white/40 text-sm ml-1">optional</span>
                    )}
                  </span>
                  <span className="text-white/80 whitespace-nowrap">
                    {ing.quantity} {ing.unit}
                  </span>
                </div>
                <div className="flex items-center gap-2 mt-1">
                  <ProvenanceBadge provenance={ing.provenance} sources={recipe.sources} sourceId={ing.source_id} />
                  {ing.note && <span className="text-white/50 text-xs">{ing.note}</span>}
                </div>
              </li>
            ))}
          </ul>
          <p className="text-white/40 text-xs mt-4 pt-3 border-t border-white/10">
            The video doesn't state every amount. Anything not shown on camera is filled in
            from a published recipe or estimated — each one is labeled above.
          </p>
        </div>
      </div>

      {/* Instructions */}
      <div className="px-6 animate-slide-up" style={{ animationDelay: '200ms' }}>
        <h2 className="text-xl font-display font-semibold text-white mb-3">Instructions</h2>
        <div className="glass-card p-4 space-y-4">
          {recipe.steps.map((step) => (
            <div key={step.id ?? step.order} className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-white/20 flex items-center justify-center text-white font-medium">
                {step.order}
              </div>
              <div className="flex-1">
                {(step.display_title || step.title) && (
                  <h3 className="text-white font-medium mb-1">{step.display_title || step.title}</h3>
                )}
                <p className="text-white">{step.instruction}</p>
                {step.detail && (
                  <p className="text-white/60 text-sm mt-1.5 leading-relaxed">{step.detail}</p>
                )}
                {step.doneness_cue && (
                  <p className="text-white/70 text-sm mt-2 pl-3 border-l-2 border-white/20">
                    <span className="text-white/40">You'll know it's right when </span>
                    {step.doneness_cue}
                  </p>
                )}
                {step.tips && step.tips.length > 0 && (
                  <ul className="mt-2 space-y-1">
                    {step.tips.map((tip, i) => (
                      <li key={i} className="text-white/70 text-sm flex items-start gap-1">
                        <span>💡</span> {tip}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Sources */}
      {recipe.sources?.length > 0 && (
        <div className="px-6 mt-6 animate-slide-up" style={{ animationDelay: '250ms' }}>
          <h2 className="text-xl font-display font-semibold text-white mb-3">Sources</h2>
          <div className="glass-card p-4">
            <ul className="space-y-1.5">
              {recipe.sources.map((source) => (
                <li key={source.id} className="text-white/70 text-sm">
                  {source.url ? (
                    <a href={source.url} target="_blank" rel="noreferrer" className="hover:text-white underline">
                      {source.title}
                    </a>
                  ) : (
                    source.title
                  )}
                  {source.site && <span className="text-white/40"> · {source.site}</span>}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Fixed bottom buttons */}
      <div className="fixed bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-peach via-peach to-transparent pt-8">
        <button
          onClick={onStartCooking}
          disabled={!clipsReady || !clipsWarm}
          className="btn-primary w-full mb-2"
        >
          {cookingButtonLabel({ clipsReady, clipsWarm, hasStartedCooking })}
        </button>
        <button
          onClick={onRestart}
          className="w-full py-2 text-white/80 hover:text-white transition-colors text-sm"
        >
          Restart Recipe
        </button>
      </div>
    </div>
  )
}


