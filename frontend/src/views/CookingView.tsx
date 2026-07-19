import { useEffect, useRef, useState } from 'react'
import { ConfirmDialog, RotateOverlay } from '../components'
import { useWakeLock } from '../hooks'
import { hasSeenHint, markHintSeen, type HintId } from '../lib/hintStore'
import type { Recipe, Step } from '../types'

// A cooldown between step changes: a tap within this window of the previous one is
// ignored, so the recipe can't be scrubbed through faster than a step can settle and
// animate in. A reading cook waits seconds between taps, so it only bites rapid tapping.
// Matches the step-entrance animation duration (animate-slide-up, 0.4s).
const STEP_COOLDOWN_MS = 400

interface CookingViewProps {
  recipe: Recipe
  /** Owned by App, so stepping out to the recipe and back resumes on this step. */
  currentStep: number
  onStepChange: (step: number) => void
  /** Leaves the recipe for good - confirmed first, see ExitControls. */
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
          label:
            ing.quantity != null
              ? `${formatQuantity(ing.quantity)} ${ing.unit ?? ''} ${ing.name}`.replace(/\s+/g, ' ')
              : `${ing.name}${ing.amount_text ? `, ${ing.amount_text}` : ''}`,
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
 * One clip in the stack. Persistent for as long as its step stays inside the window,
 * so navigating between neighbouring steps reuses an already-decoded element instead of
 * remounting a fresh <video> that has to re-open and re-decode before it can paint.
 *
 * Only the current layer plays; neighbours are mounted, paused and hidden. They preload
 * so the *next* step's first frame is already decoded when the user advances - which is
 * what makes the transition instant on browsers that pre-decode a paused, offscreen
 * video. iOS/WebKit may defer that decode until play(); the skeleton below covers that
 * window (and any far jump that lands outside the preloaded set) so the box is never
 * blank, only briefly shimmering.
 */
function ClipLayer({ url, isCurrent }: { url: string; isCurrent: boolean }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [ready, setReady] = useState(false)

  // Play only while current; pause the moment it isn't, so at most one clip ever decodes
  // continuously (no extra battery/heat from the preloaded neighbours).
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    if (isCurrent) {
      video.play().catch(() => {
        // Autoplay might be blocked by the browser.
      })
    } else {
      video.pause()
    }
  }, [isCurrent])

  return (
    <>
      <video
        ref={videoRef}
        src={url}
        loop
        muted
        playsInline
        preload="auto"
        // Without this the clip request is no-cors and the response is opaque, which the
        // service worker can neither cache nor slice a Range out of - so the prewarmed
        // copy would go unused. The Worker sends access-control-allow-origin: *, so CORS
        // mode is fine.
        crossOrigin="anonymous"
        onLoadedData={() => setReady(true)}
        className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-200 ${
          isCurrent ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
      />
      {isCurrent && !ready && <ClipSkeleton />}
    </>
  )
}

/**
 * Shimmer placeholder that fills the clip frame while a video is loading -
 * either in the browser (ClipLayer) or still rendering server-side (clipPending).
 */
function ClipSkeleton() {
  return (
    <div className="absolute inset-0 bg-white/5 overflow-hidden">
      <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/10 to-transparent animate-shimmer" />
    </div>
  )
}

/**
 * A bounded window of clips - the current step plus its immediate neighbours - stacked in
 * the frame box. Keyed by step id so a clip that stays in the window across a step change
 * is reused, not remounted. The window is small (<=3) and only one plays at a time.
 */
function ClipStack({
  recipe,
  currentStep,
}: {
  recipe: Recipe
  currentStep: number
}) {
  const indices = [currentStep - 1, currentStep, currentStep + 1].filter(
    (i) =>
      i >= 0 &&
      i < recipe.steps.length &&
      Boolean(recipe.steps[i]?.video_clip_url)
  )

  return (
    <>
      {indices.map((i) => (
        <ClipLayer
          key={recipe.steps[i].id}
          url={recipe.steps[i].video_clip_url as string}
          isCurrent={i === currentStep}
        />
      ))}
    </>
  )
}

/**
 * Tracks whether a scroll container overflows and where it sits, driving the
 * scroll arrows. Re-run via deps whenever the scroller remounts (key={step.id})
 * or changes size class, so the ref points at the fresh node.
 */
function useScrollState(
  ref: React.RefObject<HTMLDivElement>,
  deps: React.DependencyList
) {
  const [state, setState] = useState({ canScroll: false, atTop: true, atBottom: true })

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => {
      // ±1px tolerances absorb sub-pixel rounding on mobile, where scrollTop
      // can settle a fraction short of the end.
      const next = {
        canScroll: el.scrollHeight > el.clientHeight + 1,
        atTop: el.scrollTop <= 1,
        atBottom: el.scrollTop + el.clientHeight >= el.scrollHeight - 1,
      }
      setState((prev) =>
        prev.canScroll === next.canScroll &&
        prev.atTop === next.atTop &&
        prev.atBottom === next.atBottom
          ? prev
          : next
      )
    }
    update()
    el.addEventListener('scroll', update, { passive: true })
    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => {
      el.removeEventListener('scroll', update)
      observer.disconnect()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return state
}

/**
 * The messy-hands scroll controls: chevron glyphs with a full 44px tap target
 * each, so a knuckle-tap lands without needing a clean fingertip.
 *
 * Dimmed ends use the disabled attribute, never pointer-events-none: a disabled
 * button swallows the tap entirely, while pointer-events-none would let it fall
 * through to the whole-screen tap navigator and change the step - the exact
 * accident these arrows exist to prevent. Live taps stopPropagation for the
 * same reason.
 */
function ScrollArrows({
  atTop,
  atBottom,
  onScroll,
  horizontal = false,
}: {
  atTop: boolean
  atBottom: boolean
  onScroll: (dir: 1 | -1) => void
  horizontal?: boolean
}) {
  const arrow = (dir: 1 | -1) => (
    <button
      type="button"
      aria-label={dir === -1 ? 'Scroll instructions up' : 'Scroll instructions down'}
      disabled={dir === -1 ? atTop : atBottom}
      onClick={(e) => {
        e.stopPropagation()
        onScroll(dir)
      }}
      className="w-11 h-11 flex items-center justify-center rounded-full text-white/50 transition-colors hover:text-white/90 hover:bg-white/10 active:bg-white/15 disabled:opacity-25 disabled:hover:bg-transparent disabled:hover:text-white/50"
    >
      <svg
        viewBox="0 0 24 24"
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <polyline points={dir === -1 ? '6 15 12 9 18 15' : '6 9 12 15 18 9'} />
      </svg>
    </button>
  )

  return (
    <div className={`flex gap-1 ${horizontal ? '' : 'flex-col'}`}>
      {arrow(-1)}
      {arrow(1)}
    </div>
  )
}

/** Keep in sync with the hint-life animation duration in tailwind.config.js. */
const HINT_DURATION_MS = 8000

/**
 * One-time teaching label for the tap-to-navigate zones, shown on the gather
 * steps. pointer-events-none is the point: the hint sits inside the very zone
 * it describes, so tapping "on" it falls through and actually navigates.
 */
function TapHint({ side, text }: { side: 'left' | 'right'; text: string }) {
  const [visible, setVisible] = useState(true)

  useEffect(() => {
    const timer = setTimeout(() => setVisible(false), HINT_DURATION_MS)
    return () => clearTimeout(timer)
  }, [])

  if (!visible) return null
  return (
    <div
      className={`absolute top-3 z-10 pointer-events-none flex items-center gap-1.5 bg-black/20 backdrop-blur-sm rounded-full px-3 py-1.5 text-[11px] text-white/60 ${
        side === 'right'
          ? 'right-4 animate-hint-life-right'
          : 'left-4 animate-hint-life-left'
      }`}
    >
      {side === 'left' && <span className="text-white/80 animate-nudge-left">‹</span>}
      <span>{text}</span>
      {side === 'right' && <span className="text-white/80 animate-nudge-right">›</span>}
    </div>
  )
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
  const [confirmingExit, setConfirmingExit] = useState(false)
  const step: Step = recipe.steps[currentStep]
  const totalSteps = recipe.steps.length

  // Cooking means the phone is propped up with wet hands nearby - keep the
  // screen on for exactly as long as this view is mounted.
  useWakeLock()

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

  // Scroll arrows for the description, so messy hands can page through a long
  // step without touch-scrolling. Deps cover the scroller's remount (key=step.id)
  // and everything that changes its width or type size.
  const scrollerRef = useRef<HTMLDivElement>(null)
  const { canScroll, atTop, atBottom } = useScrollState(scrollerRef, [
    step?.id,
    showPanel,
    dense,
  ])
  const scrollByChunk = (dir: 1 | -1) => {
    const el = scrollerRef.current
    if (!el) return
    // 0.65 x the viewport keeps a third of the previous chunk visible for continuity.
    el.scrollBy({ top: dir * el.clientHeight * 0.65, behavior: 'smooth' })
  }

  // One-time tap-navigation hints on the two gather steps, once per recipe.
  // Marked seen on first show, so revisiting the step never replays them.
  const recipeKey = recipe.video_id ?? recipe.source_url ?? recipe.title
  const [activeHint, setActiveHint] = useState<HintId | null>(null)
  useEffect(() => {
    const hint: HintId | null =
      step?.kind === 'gather_tools'
        ? 'tap-next'
        : step?.kind === 'gather_ingredients'
          ? 'tap-prev'
          : null
    if (!hint || hasSeenHint(recipeKey, hint)) {
      setActiveHint(null)
      return
    }
    markHintSeen(recipeKey, hint)
    setActiveHint(hint)
  }, [step?.kind, recipeKey])

  // No prefetching here: App warms every clip in step order as soon as the recipe
  // lands, which is strictly earlier and covers more steps than this view could.

  // Rate-limit navigation so the recipe can't be scrubbed through. A tap that lands
  // inside the cooldown of the last accepted one is dropped, which - together with the
  // step-entrance animation - keeps each step on screen long enough to read and its clip
  // long enough to load.
  const lastNavRef = useRef(0)
  const navigate = (target: number) => {
    const clamped = Math.max(0, Math.min(totalSteps - 1, target))
    if (clamped === currentStep) return
    const now = Date.now()
    if (now - lastNavRef.current < STEP_COOLDOWN_MS) return
    lastNavRef.current = now
    onStepChange(clamped)
  }

  // Handle tap navigation
  const handleTap = (e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const tapX = e.clientX - rect.left
    const zoneWidth = rect.width * 0.3 // 30% width zones on each side

    if (tapX < zoneWidth) {
      // Tap left - previous step
      navigate(currentStep - 1)
    } else if (tapX > rect.width - zoneWidth) {
      // Tap right - next step
      navigate(currentStep + 1)
    }
  }

  // The same pair serves the landscape footer and the portrait rotate screen: turning the
  // phone mid-recipe shouldn't strand the cook with no way out or back to the recipe.
  // View Recipe keeps `currentStep` (App owns it), so coming back resumes here.
  const exitControls = (
    <div className="flex items-center justify-center gap-4 text-white/80">
      <button
        onClick={(e) => {
          e.stopPropagation()
          setConfirmingExit(true)
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
  )

  return (
    <>
    <RotateOverlay actions={exitControls}>
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
          <div className="relative flex-1 min-h-0 flex flex-col justify-center ml-9">
            {/* Keyed by step so it remounts on navigation. Without that, React reuses
                the node and a step scrolled halfway down hands its scroll offset to the
                next step, which opens partway into its own text. */}
            <div
              key={step.id}
              ref={scrollerRef}
              className={`overflow-y-auto py-4 animate-slide-up ${showPanel && canScroll ? 'pr-16' : 'pr-4'} ${showPanel ? 'max-w-lg' : 'max-w-3xl'}`}
            >
              {step.kind === 'prep_component' && (
                <span className="inline-block mb-3 px-2 py-0.5 rounded-full bg-white/15 text-white/70 text-[11px] uppercase tracking-wider">
                  Make ahead
                </span>
              )}
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

            {/* Scroll arrows, panel mode: a vertical rail on the boundary between
                the text and the video. Hidden entirely when the step fits. */}
            {showPanel && canScroll && (
              <div className="absolute right-0 top-1/2 -translate-y-1/2">
                <ScrollArrows atTop={atTop} atBottom={atBottom} onScroll={scrollByChunk} />
              </div>
            )}
          </div>

          {/* Scroll arrows, full-width mode: a horizontal pair at the bottom left,
              just above the exit controls. */}
          {!showPanel && canScroll && (
            <div className="shrink-0 ml-9 mb-2">
              <ScrollArrows atTop={atTop} atBottom={atBottom} onScroll={scrollByChunk} horizontal />
            </div>
          )}

          {/* Navigation buttons */}
          <div className="shrink-0">{exitControls}</div>
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
                  <ClipStack recipe={recipe} currentStep={currentStep} />
                ) : (
                  /* clipPending: the clip is still rendering server-side. Same
                     skeleton as a loading clip; the label distinguishes the wait. */
                  <div className="relative w-full h-full flex items-center justify-center text-white/60">
                    <ClipSkeleton />
                    <span className="relative text-sm">Preparing this clip…</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* One-time tap-navigation hints, keyed so each restarts its 8s life. */}
        {activeHint === 'tap-next' && (
          <TapHint
            key="tap-next"
            side="right"
            text="Tap right side of screen to move to next step"
          />
        )}
        {activeHint === 'tap-prev' && (
          <TapHint
            key="tap-prev"
            side="left"
            text="Tap left side of screen to move to previous step"
          />
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

    {/* Outside RotateOverlay so it shows in either orientation, and outside the tap
        container so dismissing it can't also step the recipe. */}
    {confirmingExit && (
      <ConfirmDialog
        message="Are you sure you want to exit?"
        onConfirm={onExit}
        onCancel={() => setConfirmingExit(false)}
      />
    )}
    </>
  )
}


