import { track } from '@vercel/analytics'
import type { VideoSource } from '../types'

// Product funnel events. Keep this union in sync with the call sites in App.tsx — it is the
// single source of truth for what we measure, so the Vercel dashboard's event list matches
// exactly what the code fires.
type EventName =
  | 'recipe_submitted'
  | 'recipe_cache_hit'
  | 'validation_failed'
  | 'submit_error'
  | 'recipe_generated'
  | 'recipe_failed'
  | 'clips_warmed'
  | 'clips_timed_out'
  | 'cooking_started'
  | 'cooking_exited'
  | 'shared_recipe_opened'
  | 'browse_recipe_opened'

// Vercel's track() only accepts flat props whose values are string | number | boolean | null.
type EventProps = Record<string, string | number | boolean | null>

export function trackEvent(name: EventName, props?: EventProps): void {
  track(name, props)
}

// The backend namespaces non-YouTube video IDs (`tt-`, `ig-`) so a bare 11-char ID can't
// collide across platforms — see extract_video_id() in backend/api/routes.py. We reuse that
// prefix here instead of re-parsing the URL, so "which platforms convert / fail" comes for
// free off the ID we already hold.
export function platformFromVideoId(videoId?: string): VideoSource {
  if (!videoId) return 'unknown'
  if (videoId.startsWith('tt-')) return 'tiktok'
  if (videoId.startsWith('ig-')) return 'instagram'
  return 'youtube'
}
