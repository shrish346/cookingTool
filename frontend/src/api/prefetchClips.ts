/**
 * Pulls step clips down before cooking starts, so no step waits on the network.
 *
 * There are two different mechanisms here, because the browsers disagree about what a
 * warm cache means for a <video>:
 *
 *  - With a service worker (production / any https origin): write the clip into Cache
 *    Storage under the name the SW's runtimeCaching route reads (see vite.config.ts).
 *    The SW answers the video's Range requests out of that cached full response,
 *    building the 206 itself. This is the only thing that works on iOS — WebKit will
 *    not satisfy a media Range request from an HTTP-cache entry holding a full 200,
 *    so a plain fetch() warms nothing there and the clip is fetched twice.
 *
 *  - Without a service worker (vite dev, plain-http LAN, unsupported browser): fall
 *    back to a plain GET. Nothing intercepts the video's request, but Chrome will
 *    slice a 206 out of the cached 200, which is enough.
 *
 * Doing both would download every clip twice, so pick one.
 */

const CLIP_CACHE = 'recipe-clips'

// Deduped across views: a clip warmed for the recipe view is not fetched again when
// cooking mode mounts.
const claimed = new Set<string>()

/** Is a service worker actually controlling this page (not merely registered)? */
function serviceWorkerActive(): boolean {
  return 'serviceWorker' in navigator && navigator.serviceWorker.controller !== null
}

async function warmOne(url: string): Promise<void> {
  if (serviceWorkerActive() && 'caches' in window) {
    const cache = await caches.open(CLIP_CACHE)
    if (await cache.match(url)) return
    // Stores the full 200. The SW's range plugin slices it per request.
    await cache.add(url)
    return
  }

  // Drain the body, or it never lands in the HTTP cache.
  await fetch(url, { cache: 'force-cache' }).then((res) => res.arrayBuffer())
}

// Enough to hide latency, few enough not to saturate a phone's connection.
const MAX_PARALLEL = 3

/**
 * Warm every clip, in the order given. Resolves once they're all done — a failed clip
 * counts as done, since the <video> can still load it on demand. Safe to call
 * repeatedly; already-warmed clips are skipped.
 */
export async function prefetchClips(
  urls: (string | null | undefined)[],
  onProgress?: (done: number, total: number) => void
): Promise<void> {
  const pending = urls.filter(
    (url): url is string => Boolean(url) && !claimed.has(url as string)
  )
  pending.forEach((url) => claimed.add(url))

  const total = pending.length
  if (total === 0) {
    onProgress?.(0, 0)
    return
  }

  let done = 0
  let next = 0

  const worker = async (): Promise<void> => {
    while (next < pending.length) {
      const url = pending[next++]
      try {
        await warmOne(url)
      } catch {
        // Best effort. Unclaim so a later pass can retry it.
        claimed.delete(url)
      }
      done += 1
      onProgress?.(done, total)
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(MAX_PARALLEL, total) }, () => worker())
  )
}
