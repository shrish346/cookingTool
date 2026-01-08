import type { Recipe } from '../types'

interface RecipeViewProps {
  recipe: Recipe
  clipsReady: boolean
  onStartCooking: () => void
  onRestart: () => void
}

/**
 * Recipe overview view with ingredients and steps
 */
export function RecipeView({ recipe, clipsReady, onStartCooking, onRestart }: RecipeViewProps) {
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

      {/* Ingredients */}
      <div className="px-6 mb-6 animate-slide-up" style={{ animationDelay: '150ms' }}>
        <h2 className="text-xl font-display font-semibold text-white mb-3">Ingredients</h2>
        <div className="glass-card p-4">
          <ul className="space-y-2">
            {recipe.ingredients.map((ing, index) => (
              <li key={index} className="flex justify-between text-white">
                <span>
                  {ing.name}
                  {ing.preparation && (
                    <span className="text-white/60 text-sm ml-1">({ing.preparation})</span>
                  )}
                </span>
                <span className="text-white/80">
                  {ing.quantity} {ing.unit}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Instructions */}
      <div className="px-6 animate-slide-up" style={{ animationDelay: '200ms' }}>
        <h2 className="text-xl font-display font-semibold text-white mb-3">Instructions</h2>
        <div className="glass-card p-4 space-y-4">
          {recipe.steps.map((step) => (
            <div key={step.order} className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-white/20 flex items-center justify-center text-white font-medium">
                {step.order}
              </div>
              <div className="flex-1">
                <p className="text-white">{step.instruction}</p>
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

      {/* Fixed bottom buttons */}
      <div className="fixed bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-peach via-peach to-transparent pt-8">
        <button
          onClick={onStartCooking}
          disabled={!clipsReady}
          className="btn-primary w-full mb-2"
        >
          {clipsReady ? 'Start Cooking' : 'Preparing video clips...'}
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


