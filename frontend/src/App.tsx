import { useState, useEffect, useCallback } from 'react'
import { Analytics } from '@vercel/analytics/react'
import { LandingView, BrowseView, LoadingView, RecipeView, CookingView } from './views'
import { useSavedRecipe } from './hooks'
import { api, ApiError } from './api/client'
import { prefetchClips } from './api/prefetchClips'
import { fetchStoredRecipe, requestedRecipeId } from './api/storedRecipe'
import { trackEvent, platformFromVideoId } from './lib/analytics'
import { SITE_NOTICE, BROWSE_RECIPES, noticeMode } from './config/site'
import type { Recipe, StatusResponse } from './types'

type AppState = 'landing' | 'browse' | 'loading' | 'recipe' | 'cooking'

// How long the loading screen will wait on clip downloads before showing the recipe
// anyway. Long enough for a handful of clips on a slow phone connection; short enough
// that a dead network doesn't hold the recipe hostage.
const CLIP_WARM_TIMEOUT_MS = 20_000

export default function App() {
  // App state. Notice mode (config/site.ts) swaps the link-entry landing for the notice
  // and the stored-recipe list; every other state behaves identically in both modes.
  const [state, setState] = useState<AppState>(noticeMode ? 'browse' : 'landing')
  const [isValidating, setIsValidating] = useState(false)
  const [error, setError] = useState<string>()

  // The recipe being pulled out of storage from the browse list, if any.
  const [browseLoadingId, setBrowseLoadingId] = useState<string>()
  
  // Loading state
  const [jobId, setJobId] = useState<string>()
  const [progress, setProgress] = useState(0)
  const [statusMessage, setStatusMessage] = useState('')
  
  // Recipe state
  const [recipe, setRecipe] = useState<Recipe | null>(null)
  const [clipsReady, setClipsReady] = useState(false)

  // Clip download state. `clipsWarm` gates the cooking button; `warmed` is just the
  // count shown on it while they come down.
  const [clipsWarm, setClipsWarm] = useState(false)
  const [warmed, setWarmed] = useState({ done: 0, total: 0 })

  // Cooking position lives here, not in CookingView, so that stepping out to look at
  // the recipe and coming back resumes on the same step instead of restarting at one.
  const [cookingStep, setCookingStep] = useState(0)
  const [hasStartedCooking, setHasStartedCooking] = useState(false)
  
  // Persistent storage
  const [savedRecipe, setSavedRecipe] = useSavedRecipe()

  // On mount: `?recipe=<video_id>` reads that recipe straight from object storage and
  // skips the API entirely, so the recipe and cooking views can be worked on without
  // running the backend. It takes precedence over whatever is in localStorage.
  // Otherwise, rehydrate the saved recipe as usual - except in notice mode, where a
  // returning visitor should land on the notice rather than be dropped straight back
  // into whatever they last opened, having never seen why the pipeline is offline.
  useEffect(() => {
    const storedId = requestedRecipeId()

    if (storedId) {
      fetchStoredRecipe(storedId)
        .then((stored) => {
          setRecipe(stored)
          setClipsReady(stored.clips_ready)
          setState('recipe')
          trackEvent('shared_recipe_opened', { platform: platformFromVideoId(storedId) })
        })
        .catch((err: Error) => setError(err.message))
      return
    }

    if (!noticeMode && savedRecipe.recipe && state === 'landing') {
      setRecipe(savedRecipe.recipe)
      setClipsReady(savedRecipe.recipe.clips_ready)
      setState('recipe')
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Pull the clips down as soon as the recipe is on screen, while the user is reading
  // the ingredients. Cooking is held shut until they land (see RecipeView) rather than
  // letting someone click straight through into a step whose video isn't there yet.
  //
  // Capped: a slow connection must not disable the button forever, so once the timeout
  // is up cooking opens regardless and any clip that hasn't landed loads on demand.
  useEffect(() => {
    if (!recipe) return

    const urls = recipe.steps.map((step) => step.video_clip_url)
    if (urls.filter(Boolean).length === 0) {
      setClipsWarm(true)
      return
    }

    let cancelled = false
    let opened = false
    let lastDone = 0
    let lastTotal = urls.filter(Boolean).length
    setClipsWarm(false)

    const openCooking = (timedOut: boolean) => {
      if (cancelled || opened) return
      opened = true
      setClipsWarm(true)
      trackEvent(timedOut ? 'clips_timed_out' : 'clips_warmed', {
        warmed_done: lastDone,
        warmed_total: lastTotal,
      })
    }

    const timer = setTimeout(() => openCooking(true), CLIP_WARM_TIMEOUT_MS)

    void prefetchClips(urls, (done, count) => {
      if (cancelled) return
      lastDone = done
      lastTotal = count
      setWarmed({ done, total: count })
    }).then(() => {
      clearTimeout(timer)
      openCooking(false)
    })

    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [recipe])

  // Poll for job status
  useEffect(() => {
    if (state !== 'loading' || !jobId) return

    const pollInterval = setInterval(async () => {
      try {
        const status: StatusResponse = await api.getStatus(jobId)

        if (status.status === 'completed' && status.recipe) {
          clearInterval(pollInterval)
          setRecipe(status.recipe)
          setClipsReady(status.clips_ready)

          // Save to localStorage
          setSavedRecipe({
            recipe: status.recipe,
            savedAt: new Date().toISOString(),
          })

          setProgress(status.progress)
          setStatusMessage(status.message)
          setState('recipe')
          trackEvent('recipe_generated', {
            platform: platformFromVideoId(status.recipe.video_id ?? jobId),
            step_count: status.recipe.steps.length,
            expansion_failed: status.recipe.expansion_failed ?? false,
          })
        } else if (status.status === 'failed') {
          clearInterval(pollInterval)
          setError(status.error || 'Failed to generate recipe')
          setState('landing')
          trackEvent('recipe_failed', {
            error: status.error ?? 'Failed to generate recipe',
            platform: platformFromVideoId(jobId),
          })
        } else {
          // Pipeline progress. Not applied on completion, so the bar doesn't jump to
          // 100 and then back down for the clip download.
          setProgress(status.progress)
          setStatusMessage(status.message)
        }
      } catch (err) {
        console.error('Error polling status:', err)
      }
    }, 2000) // Poll every 2 seconds

    return () => clearInterval(pollInterval)
  }, [state, jobId, setSavedRecipe])

  // Continue polling for clips after recipe is ready
  useEffect(() => {
    if (state !== 'recipe' || !jobId || clipsReady) return

    const clipsPollInterval = setInterval(async () => {
      try {
        const status = await api.getStatus(jobId)
        if (status.clips_ready) {
          clearInterval(clipsPollInterval)
          setClipsReady(true)
          
          // Update saved recipe with clips
          if (status.recipe) {
            setRecipe(status.recipe)
            setSavedRecipe({
              recipe: status.recipe,
              savedAt: new Date().toISOString(),
            })
          }
        }
      } catch (err) {
        console.error('Error polling clips status:', err)
      }
    }, 3000) // Poll every 3 seconds

    return () => clearInterval(clipsPollInterval)
  }, [state, jobId, clipsReady, setSavedRecipe])

  // Handle URL submission
  const handleSubmit = useCallback(async (url: string) => {
    setError(undefined)
    setIsValidating(true)

    try {
      // Step 1: Validate URL
      const validation = await api.validateUrl(url)
      
      if (!validation.valid) {
        setError(validation.error || 'Invalid URL')
        setIsValidating(false)
        trackEvent('validation_failed', { error: validation.error ?? 'Invalid URL' })
        return
      }

      // Step 2: Start processing
      const processResponse = await api.processVideo(url)
      setJobId(processResponse.video_id)
      setProgress(5)
      setStatusMessage(processResponse.message)

      const platform = platformFromVideoId(processResponse.video_id)
      trackEvent('recipe_submitted', { platform })

      // If already completed (cached), go directly to recipe
      if (processResponse.status === 'completed') {
        const status = await api.getStatus(processResponse.job_id)
        if (status.recipe) {
          setRecipe(status.recipe)
          setClipsReady(status.clips_ready)
          setSavedRecipe({
            recipe: status.recipe,
            savedAt: new Date().toISOString(),
          })
          setState('recipe')
          setIsValidating(false)
          trackEvent('recipe_cache_hit', { platform })
          return
        }
      }

      // Go to loading state
      setState('loading')
      setIsValidating(false)
      
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message)
      } else {
        setError('Something went wrong. Please try again.')
      }
      setIsValidating(false)
      trackEvent('submit_error', {
        error: err instanceof Error ? err.message : String(err),
        is_api_error: err instanceof ApiError,
      })
    }
  }, [setSavedRecipe])

  // Open a recipe from the browse list. Reads it straight out of object storage, the
  // same path `?recipe=` takes - no API call, so this works with the backend down.
  // Deliberately does not write to localStorage: that single slot belongs to a recipe
  // the user actually generated, and notice mode doesn't rehydrate from it anyway.
  const handleSelectStored = useCallback((videoId: string) => {
    setError(undefined)
    setBrowseLoadingId(videoId)

    fetchStoredRecipe(videoId)
      .then((stored) => {
        setRecipe(stored)
        setClipsReady(stored.clips_ready)
        setState('recipe')
        trackEvent('browse_recipe_opened', { platform: platformFromVideoId(videoId) })
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setBrowseLoadingId(undefined))
  }, [])

  // Navigation handlers
  const handleStartCooking = useCallback(() => {
    setHasStartedCooking(true)
    setState('cooking')
    trackEvent('cooking_started', {
      platform: platformFromVideoId(recipe?.video_id ?? jobId),
    })
  }, [recipe, jobId])

  // View Recipe keeps `cookingStep`, so coming back resumes where they left off. Exit
  // is a real exit - it lands on `handleRestart` below and throws the recipe away - so
  // both views confirm before calling it.
  const handleViewRecipe = useCallback(() => {
    setState('recipe')
  }, [])

  const handleRestart = useCallback(() => {
    trackEvent('cooking_exited', { from_state: state })
    setRecipe(null)
    setClipsReady(false)
    setClipsWarm(false)
    setWarmed({ done: 0, total: 0 })
    setCookingStep(0)
    setHasStartedCooking(false)
    setJobId(undefined)
    setProgress(0)
    setError(undefined)
    setSavedRecipe({ recipe: null, savedAt: null })
    setState(noticeMode ? 'browse' : 'landing')
  }, [state, setSavedRecipe])

  // Render current view
  return (
    <div className="h-full bg-peach overflow-hidden">
      <Analytics />
      {state === 'landing' && (
        <LandingView
          onSubmit={handleSubmit}
          isValidating={isValidating}
          error={error}
        />
      )}
      
      {state === 'browse' && (
        <BrowseView
          notice={SITE_NOTICE}
          recipes={BROWSE_RECIPES}
          onSelect={handleSelectStored}
          loadingId={browseLoadingId}
          error={error}
        />
      )}

      {state === 'loading' && (
        <LoadingView
          progress={progress}
          message={statusMessage}
        />
      )}
      
      {state === 'recipe' && recipe && (
        <RecipeView
          recipe={recipe}
          clipsReady={clipsReady}
          clipsWarm={clipsWarm}
          warmed={warmed}
          hasStartedCooking={hasStartedCooking}
          onStartCooking={handleStartCooking}
          onRestart={handleRestart}
          exitLabel={noticeMode ? 'Back to recipes' : undefined}
          confirmOnExit={!noticeMode}
        />
      )}

      {state === 'cooking' && recipe && (
        <CookingView
          recipe={recipe}
          currentStep={cookingStep}
          onStepChange={setCookingStep}
          onExit={handleRestart}
          onViewRecipe={handleViewRecipe}
        />
      )}
    </div>
  )
}


