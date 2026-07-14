import { useState, useEffect, useCallback } from 'react'
import { LandingView, LoadingView, RecipeView, CookingView } from './views'
import { useSavedRecipe } from './hooks'
import { api, ApiError } from './api/client'
import { prefetchClips } from './api/prefetchClips'
import { fetchStoredRecipe, requestedRecipeId } from './api/storedRecipe'
import type { Recipe, StatusResponse } from './types'

type AppState = 'landing' | 'loading' | 'recipe' | 'cooking'

export default function App() {
  // App state
  const [state, setState] = useState<AppState>('landing')
  const [isValidating, setIsValidating] = useState(false)
  const [error, setError] = useState<string>()
  
  // Loading state
  const [jobId, setJobId] = useState<string>()
  const [progress, setProgress] = useState(0)
  const [statusMessage, setStatusMessage] = useState('')
  
  // Recipe state
  const [recipe, setRecipe] = useState<Recipe | null>(null)
  const [clipsReady, setClipsReady] = useState(false)
  
  // Persistent storage
  const [savedRecipe, setSavedRecipe] = useSavedRecipe()

  // On mount: `?recipe=<video_id>` reads that recipe straight from object storage and
  // skips the API entirely, so the recipe and cooking views can be worked on without
  // running the backend. It takes precedence over whatever is in localStorage.
  // Otherwise, rehydrate the saved recipe as usual.
  useEffect(() => {
    const storedId = requestedRecipeId()

    if (storedId) {
      fetchStoredRecipe(storedId)
        .then((stored) => {
          setRecipe(stored)
          setClipsReady(stored.clips_ready)
          setState('recipe')
        })
        .catch((err: Error) => setError(err.message))
      return
    }

    if (savedRecipe.recipe && state === 'landing') {
      setRecipe(savedRecipe.recipe)
      setClipsReady(savedRecipe.recipe.clips_ready)
      setState('recipe')
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Warm every clip as soon as the recipe exists, in step order, rather than waiting
  // for cooking mode to ask for one. The user spends time on the recipe view reading
  // ingredients; that's when the clips get pulled, so stepping through is instant.
  // Clips arrive on a second poll, so this re-runs when the recipe updates - already
  // warmed clips are skipped.
  useEffect(() => {
    if (!recipe) return
    prefetchClips(recipe.steps.map((step) => step.video_clip_url))
  }, [recipe])

  // Poll for job status
  useEffect(() => {
    if (state !== 'loading' || !jobId) return

    const pollInterval = setInterval(async () => {
      try {
        const status: StatusResponse = await api.getStatus(jobId)
        
        setProgress(status.progress)
        setStatusMessage(status.message)

        if (status.status === 'completed' && status.recipe) {
          clearInterval(pollInterval)
          setRecipe(status.recipe)
          setClipsReady(status.clips_ready)
          
          // Save to localStorage
          setSavedRecipe({
            recipe: status.recipe,
            savedAt: new Date().toISOString(),
          })
          
          setState('recipe')
        } else if (status.status === 'failed') {
          clearInterval(pollInterval)
          setError(status.error || 'Failed to generate recipe')
          setState('landing')
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
        return
      }

      // Step 2: Start processing
      const processResponse = await api.processVideo(url)
      setJobId(processResponse.video_id)
      setProgress(5)
      setStatusMessage(processResponse.message)
      
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
    }
  }, [setSavedRecipe])

  // Navigation handlers
  const handleStartCooking = useCallback(() => {
    setState('cooking')
  }, [])

  const handleExitCooking = useCallback(() => {
    setState('recipe')
  }, [])

  const handleViewRecipe = useCallback(() => {
    setState('recipe')
  }, [])

  const handleRestart = useCallback(() => {
    setRecipe(null)
    setClipsReady(false)
    setJobId(undefined)
    setProgress(0)
    setError(undefined)
    setSavedRecipe({ recipe: null, savedAt: null })
    setState('landing')
  }, [setSavedRecipe])

  // Render current view
  return (
    <div className="h-full bg-peach overflow-hidden">
      {state === 'landing' && (
        <LandingView
          onSubmit={handleSubmit}
          isValidating={isValidating}
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
          onStartCooking={handleStartCooking}
          onRestart={handleRestart}
        />
      )}
      
      {state === 'cooking' && recipe && (
        <CookingView
          recipe={recipe}
          onExit={handleExitCooking}
          onViewRecipe={handleViewRecipe}
        />
      )}
    </div>
  )
}


