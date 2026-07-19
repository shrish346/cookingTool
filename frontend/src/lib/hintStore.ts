/**
 * Tracks which one-time UI hints the user has already seen, per recipe.
 *
 * This module is the only place that knows where seen-flags live. Today that
 * is localStorage; when user auth lands, swap these two function bodies for
 * reads/writes against the user's profile and no call site changes. The
 * signatures are deliberately synchronous — an async backend can hydrate a
 * local cache at login and write through on mark, so the UI never awaits.
 */

export type HintId = 'tap-next' | 'tap-prev'

const storageKey = (recipeKey: string, hint: HintId) =>
  `makerai:hints:${recipeKey}:${hint}`

export function hasSeenHint(recipeKey: string, hint: HintId): boolean {
  try {
    return localStorage.getItem(storageKey(recipeKey, hint)) !== null
  } catch {
    return false
  }
}

export function markHintSeen(recipeKey: string, hint: HintId): void {
  try {
    localStorage.setItem(storageKey(recipeKey, hint), '1')
  } catch {
    // Storage unavailable (private mode, quota) - the hint just shows again.
  }
}
