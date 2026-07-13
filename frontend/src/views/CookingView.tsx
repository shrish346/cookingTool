import { useState, useEffect, useRef } from 'react'
import { RotateOverlay } from '../components'
import type { Recipe, Step } from '../types'

interface CookingViewProps {
  recipe: Recipe
  onExit: () => void
  onViewRecipe: () => void
}

/**
 * The two setup steps have no clip - they show what to lay out instead.
 */
function GatherChecklist({ recipe, step }: { recipe: Recipe; step: Step }) {
  const items =
    step.kind === 'gather_tools'
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

  return (
    <div className="w-full h-full overflow-y-auto p-6">
      <ul className="space-y-2.5">
        {items.map((item) => (
          <li key={item.key} className="flex items-start gap-2.5 text-white/90">
            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-white/50 shrink-0" />
            <span className="text-sm leading-snug">
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

export function CookingView({ recipe, onExit, onViewRecipe }: CookingViewProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const videoRef = useRef<HTMLVideoElement>(null)

  const step: Step = recipe.steps[currentStep]
  const totalSteps = recipe.steps.length

  const hasVideo = Boolean(step?.video_clip_url)
  // A step the video never showed will never get a clip, so don't sit on a spinner
  // for it. Only a step that *is* grounded and whose clip hasn't landed yet is pending.
  const clipPending = !hasVideo && step?.has_video_clip !== false && !recipe.clips_ready
  const isGather = step?.kind === 'gather_tools' || step?.kind === 'gather_ingredients'

  // Clips adjacent to the current step, warmed ahead of time. Two forward (the
  // common direction) and one back.
  const adjacentUrls = [
    recipe.steps[currentStep + 1]?.video_clip_url,
    recipe.steps[currentStep + 2]?.video_clip_url,
    recipe.steps[currentStep - 1]?.video_clip_url,
  ].filter((url): url is string => Boolean(url))

  // Warm the browser's HTTP cache for those clips.
  //
  // This used to be hidden <video preload="auto"> elements, which did nothing on
  // a phone: iOS Safari ignores `preload` on video and refuses to fetch media
  // bytes until the user actually plays it, and `display: none` suppresses
  // preloading in other browsers too. So every clip was really loading on demand
  // — the blank-video stall.
  //
  // Fetching the bytes ourselves is a plain HTTP GET, which iOS does honor. The
  // response lands in the HTTP cache (clips are uploaded `immutable`, 1y max-age),
  // so the later <video src> resolves from cache instead of the network.
  const prefetched = useRef<Set<string>>(new Set())

  useEffect(() => {
    for (const url of adjacentUrls) {
      if (prefetched.current.has(url)) continue
      prefetched.current.add(url)

      // Deliberately not aborted on cleanup: stepping forward re-runs this
      // effect, and the in-flight fetch is usually the clip about to be needed.
      fetch(url, { cache: 'force-cache' })
        .then((res) => res.arrayBuffer()) // drain so the body is fully cached
        .catch(() => {
          // Prefetch is best-effort; the <video> will still load on demand.
          prefetched.current.delete(url)
        })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adjacentUrls.join(',')])

  // Handle tap navigation
  const handleTap = (e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const tapX = e.clientX - rect.left
    const zoneWidth = rect.width * 0.3 // 30% width zones on each side

    if (tapX < zoneWidth) {
      // Tap left - previous step
      setCurrentStep((prev) => Math.max(0, prev - 1))
    } else if (tapX > rect.width - zoneWidth) {
      // Tap right - next step
      setCurrentStep((prev) => Math.min(totalSteps - 1, prev + 1))
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
        <div className="flex-[1.2] flex flex-col justify-between p-6 lg:p-8">
          {/* Step counter */}
          <div className="text-white/60 text-sm ml-9 -mt-2">
            Step {currentStep + 1} of {totalSteps}
          </div>

          {/* Step title and instruction */}
          <div className="flex-1 flex flex-col justify-center ml-9 -mt-12 overflow-y-auto">
            <h1 className="text-3xl lg:text-4xl font-display font-bold text-white mb-4">
              {step.display_title || step.title || `Step ${step.order}`}
            </h1>
            <p className="text-white/90 text-lg lg:text-xl leading-relaxed max-w-lg">
              {step.instruction}
            </p>
            {step.detail && (
              <p className="text-white/70 text-base leading-relaxed max-w-lg mt-3">
                {step.detail}
              </p>
            )}
            {/* Only shown here when the clip panel isn't already showing it. */}
            {step.doneness_cue && hasVideo && (
              <p className="text-white/80 text-base leading-relaxed max-w-lg mt-4 pl-3 border-l-2 border-white/30">
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

          {/* Navigation buttons */}
          <div className="flex items-center justify-center gap-4 text-white/80">
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

        {/* Right panel - the clip, or whatever stands in for it */}
        <div className="flex-[0.8] flex items-center justify-end p-8 pr-12 lg:pr-16">
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
                className="w-full h-full object-cover"
              />
            ) : isGather ? (
              <GatherChecklist recipe={recipe} step={step} />
            ) : clipPending ? (
              <div className="w-full h-full flex flex-col items-center justify-center text-white/60">
                <div className="w-10 h-10 mb-3 rounded-full border-2 border-white/20 border-t-white/70 animate-spin" />
                <span className="text-sm">Preparing this clip…</span>
              </div>
            ) : (
              /* A step the video never showed. No clip is coming - show the coaching
                 instead of a video box that stays empty forever. */
              <div className="w-full h-full flex flex-col justify-center p-6 text-white/80">
                {step.doneness_cue ? (
                  <>
                    <span className="text-xs uppercase tracking-wide text-white/40 mb-2">
                      You'll know it's right when
                    </span>
                    <p className="text-base leading-relaxed">{step.doneness_cue}</p>
                  </>
                ) : (
                  <p className="text-base leading-relaxed text-white/70">{step.detail}</p>
                )}
              </div>
            )}
          </div>
        </div>

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


