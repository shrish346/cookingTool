/**
 * Warms the browser's HTTP cache for step clips.
 *
 * A clip that reaches <video> cold takes ~1.5s to become playable; one whose bytes
 * are already in the HTTP cache takes ~20ms. Clips are uploaded with a 1y max-age,
 * so a plain GET is all it takes to turn the later <video src> into a cache hit.
 *
 * It has to be fetch() and not <video preload="auto">: iOS Safari ignores preload
 * and refuses to fetch media bytes until the user actually plays, and a hidden
 * <video> is suppressed in other browsers too. A plain GET, iOS does honor.
 *
 * State is module-level, not per-component, so a clip already warmed for the recipe
 * view isn't fetched a second time when cooking mode asks for it.
 */

const claimed = new Set<string>()
const queue: string[] = []
let active = 0

// Enough to hide latency, few enough not to saturate a phone's connection and
// starve the clip that's actually on screen.
const MAX_PARALLEL = 3

function pump(): void {
  while (active < MAX_PARALLEL && queue.length > 0) {
    const url = queue.shift() as string
    active += 1

    fetch(url, { cache: 'force-cache' })
      // Drain the body, or it never lands in the cache.
      .then((res) => res.arrayBuffer())
      .catch(() => {
        // Best-effort: unclaim so a later pass can retry, and let the <video>
        // fall back to loading on demand.
        claimed.delete(url)
      })
      .finally(() => {
        active -= 1
        pump()
      })
  }
}

/**
 * Queue clips for warming, in the order given. Already-warmed or already-queued
 * URLs are skipped, so this is cheap to call repeatedly. Non-clip steps pass a
 * null url; those are ignored.
 */
export function prefetchClips(urls: (string | null | undefined)[]): void {
  for (const url of urls) {
    if (!url || claimed.has(url)) continue
    claimed.add(url)
    queue.push(url)
  }
  pump()
}
