/**
 * Reads a finished recipe straight out of object storage, bypassing the API.
 *
 * The pipeline writes `{video_id}/recipe.json` next to `{video_id}/clips/*`, and the
 * clips Worker serves any key in the bucket by path with `access-control-allow-origin: *`
 * — so a recipe that has already been processed can be loaded by the browser alone.
 * No API, no Redis, no S3 credentials.
 *
 * This only reaches recipes that already exist. Making a *new* one needs the backend:
 * yt-dlp, the VLM/LLM passes and FFmpeg all run server-side. So this is a way to work
 * on the recipe and cooking views without booting the stack, not a way to run headless.
 */

import type { Recipe } from '../types'

const STORAGE_BASE = import.meta.env.VITE_RECIPE_BASE_URL as string | undefined

/** The video id to load, from `?recipe=<video_id>`. */
export function requestedRecipeId(): string | null {
  return new URLSearchParams(window.location.search).get('recipe')
}

export async function fetchStoredRecipe(videoId: string): Promise<Recipe> {
  if (!STORAGE_BASE) {
    throw new Error('Set VITE_RECIPE_BASE_URL to load a recipe straight from storage.')
  }

  const url = `${STORAGE_BASE.replace(/\/+$/, '')}/${videoId}/recipe.json`
  const res = await fetch(url)

  if (!res.ok) {
    throw new Error(
      res.status === 404
        ? `No recipe stored for "${videoId}" — it has to have been processed once.`
        : `Could not read recipe for "${videoId}" (HTTP ${res.status}).`
    )
  }

  return res.json() as Promise<Recipe>
}
