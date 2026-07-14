import { useEffect, useRef } from 'react'
import { RotateOverlay } from '../components'
import type { Recipe, Step } from '../types'

interface CookingViewProps {
  recipe: Recipe
  /** Owned by App, so stepping out to the recipe and back resumes on this step. */
  currentStep: number
  onStepChange: (step: number) => void
  onExit: () => void
  onViewRecipe: () => void
}

/**
 * The two setup steps have no clip - the thing to look at is the list itself, so
 * it gets the panel a clip would have had: a plain scrolling list, no video frame
 * around it.
 */
function GatherList({ recipe, step }: { recipe: Recipe; step: Step }) {
  const isTools = step.kind === 'gather_tools'

  const items = isTools
    ? recipe.tools
        .filter((tool) => step.tool_ids.includes(tool.id))
        .map((tool) => ({
          key: tool.id,
          label: tool.name,
          hint: tool.substitute,
        }))
    : recipe.ingredients
        .filter((ing) => step.ingredient_ids.includes(ing.id))
        .map((ing) => ({
          key: ing.id,
          label: `${formatQuantity(ing.quantity)} ${ing.unit} ${ing.name}`,
          hint: ing.preparation,
        }))

  // h-full bounds the <ul> so it scrolls rather than growing past the panel.
  return (
    <div className="flex flex-col h-full min-h-0 w-full max-w-sm">
      <h2 className="shrink-0 text-xs uppercase tracking-wide text-white/40 mb-3">
        {isTools ? 'Tools' : 'Ingredients'} · {items.length}
      </h2>
      <ul className="flex-1 min-h-0 overflow-y-auto space-y-3 pr-2">
        {items.map((item) => (
          <li key={item.key} className="flex items-start gap-3 text-white/90">
            <span className="mt-2 w-1.5 h-1.5 rounded-full bg-white/50 shrink-0" />
            <span className="text-base leading-snug">
              {item.label}
              {item.hint && <span className="text-white/50"> — {item.hint}</span>}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** 0.25 -> "1/4", 3 -> "3", 1.5 -> "1.5" */
function formatQuantity(quantity: number): string {
  const fractions: Record<number, string> = {
    0.25: '1/4',
    0.33: '1/3',
    0.5: '1/2',
    0.67: '2/3',
    0.75: '3/4',
  }
  const whole = Math.floor(quantity)
  const remainder = Number((quantity - whole).toFixed(2))
  const fraction = fractions[remainder]

  if (!fraction) return String(Number(quantity.toFixed(2)))
  return whole > 0 ? `${whole} ${fraction}` : fraction
}

/**
 * Landscape cooking mode with video loops and step navigation
 */

export function CookingView({
  recipe,
  currentStep,
  onStepChange,
  onExit,
  onViewRecipe,
}: CookingViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null)

  const step: Step = recipe.steps[currentStep]
  const totalSteps = recipe.steps.length

  const hasVideo = Boolean(step?.video_clip_url)
  // A step the video never showed will never get a clip, so don't sit on a spinner
  // for it. Only a step that *is* grounded and whose clip hasn't landed yet is pending.
  const clipPending = !hasVideo && step?.has_video_clip !== false && !recipe.clips_ready
  const isGather = step?.kind === 'gather_tools' || step?.kind === 'gather_ingredients'

  // A step with no clip and none coming gets no right panel at all - the text takes
  // the full width rather than sitting next to an empty frame. (Future artifacts that
  // stand in for a clip will bring their own display type, not this box.)
  const showPanel = hasVideo || clipPending || isGather

  // Long steps used to overflow their column and ride up over the step counter. The
  // column now scrolls instead, and the type scales down first so that scrolling is
  // the exception rather than the norm.
  const charCount =
    (step?.instruction?.length ?? 0) +
    (step?.detail?.length ?? 0) +
    (step?.doneness_cue?.length ?? 0)
  const dense = charCount > 400

  // No prefetching here: App warms every clip in step order as soon as the recipe
  // lands, which is strictly earlier and covers more steps than this view could.

  // Handle tap navigation
  const handleTap = (e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const tapX = e.clientX - rect.left
    const zoneWidth = rect.width * 0.3 // 30% width zones on each side

    if (tapX < zoneWidth) {
      // Tap left - previous step
      onStepChange(Math.max(0, currentStep - 1))
    } else if (tapX > rect.width - zoneWidth) {
      // Tap right - next step
      onStepChange(Math.min(totalSteps - 1, currentStep + 1))
    }
  }

  // Auto-play video when step changes
  useEffect(() => {
    if (videoRef.current && hasVideo) {
      videoRef.current.play().catch(() => {
        // Autoplay might be blocked by browser
      })
    }
  }, [currentStep, hasVideo])

  return (
    <RotateOverlay>
      <div
        className="cooking-mode flex"
        onClick={handleTap}
      >
        {/* Left panel - Step info */}
        <div
          className={`${showPanel ? 'flex-[1.2]' : 'flex-1'} flex flex-col min-h-0 p-6 lg:p-8`}
        >
          {/* Step counter. shrink-0 so a long step can never push into it. */}
          <div className="shrink-0 text-white/60 text-sm ml-9">
            Step {currentStep + 1} of {totalSteps}
          </div>

          {/* Step title and instruction. Centered while it fits, scrolls once it doesn't. */}
          <div className="flex-1 min-h-0 flex flex-col justify-center ml-9">
            {/* Keyed by step so it remounts on navigation. Without that, React reuses
                the node and a step scrolled halfway down hands its scroll offset to the
                next step, which opens partway into its own text. */}
            <div
              key={step.id}
              className={`overflow-y-auto py-4 pr-4 ${showPanel ? 'max-w-lg' : 'max-w-3xl'}`}
            >
              <h1
                className={`${dense ? 'text-2xl lg:text-3xl' : 'text-3xl lg:text-4xl'} font-display font-bold text-white mb-4`}
              >
                {step.display_title || step.title || `Step ${step.order}`}
              </h1>
              <p
                className={`text-white/90 ${dense ? 'text-base lg:text-lg' : 'text-lg lg:text-xl'} leading-relaxed`}
              >
                {step.instruction}
              </p>
              {step.detail && (
                <p className="text-white/70 text-base leading-relaxed mt-3">
                  {step.detail}
                </p>
              )}
              {step.doneness_cue && (
                <p className="text-white/80 text-base leading-relaxed mt-4 pl-3 border-l-2 border-white/30">
                  <span className="text-white/40">You'll know it's right when </span>
                  {step.doneness_cue}
                </p>
              )}
              {step.duration_minutes && (
                <p className="text-white/60 mt-4">
                  ⏱️ {step.duration_minutes} minutes
                </p>
              )}
            </div>
          </div>

          {/* Navigation buttons */}
          <div className="shrink-0 flex items-center justify-center gap-4 text-white/80">
            <button
              onClick={(e) => {
                e.stopPropagation()
                onExit()
              }}
              className="hover:text-white transition-colors"
            >
              Exit
            </button>
            <span className="text-white/40">|</span>
            <button
              onClick={(e) => {
                e.stopPropagation()
                onViewRecipe()
              }}
              className="hover:text-white transition-colors"
            >
              View Recipe
            </button>
          </div>
        </div>

        {/* Right panel - only when there is something to put in it. A step with no
            clip coming renders no panel at all; its text has the full width instead. */}
        {showPanel && (
          <div className="flex-[0.8] flex flex-col justify-center items-end min-h-0 p-8 pr-12 lg:pr-16">
            {isGather ? (
              // Keyed for the same reason as the text column: its <ul> scrolls too.
              <GatherList key={step.id} recipe={recipe} step={step} />
            ) : (
              <div className="relative w-full max-w-[300px] lg:max-w-md aspect-square bg-peach-600/30 rounded-2xl overflow-hidden shadow-2xl">
                {hasVideo ? (
                  <video
                    key={step.video_clip_url}
                    ref={videoRef}
                    src={step.video_clip_url}
                    loop
                    muted
                    autoPlay
                    playsInline
                    preload="auto"
                    // Without this the clip request is no-cors and the response is
                    // opaque, which the service worker can neither cache nor slice a
                    // Range out of - so the prewarmed copy would go unused. The Worker
                    // sends access-control-allow-origin: *, so CORS mode is fine.
                    crossOrigin="anonymous"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  /* clipPending: the clip is still rendering server-side. */
                  <div className="w-full h-full flex flex-col items-center justify-center text-white/60">
                    <div className="w-10 h-10 mb-3 rounded-full border-2 border-white/20 border-t-white/70 animate-spin" />
                    <span className="text-sm">Preparing this clip…</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Tap zones indicator (subtle) */}
        <div className="absolute inset-0 pointer-events-none flex opacity-0 hover:opacity-100 transition-opacity">
          <div className="w-[30%] flex items-center justify-start pl-4">
            {currentStep > 0 && (
              <span className="text-white/30 text-4xl">‹</span>
            )}
          </div>
          <div className="flex-1" />
          <div className="w-[30%] flex items-center justify-end pr-4">
            {currentStep < totalSteps - 1 && (
              <span className="text-white/30 text-4xl">›</span>
            )}
          </div>
        </div>
      </div>
    </RotateOverlay>
  )
}


