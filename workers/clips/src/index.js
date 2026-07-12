/**
 * makerai-clips — serves recipe clips from the R2 bucket over a workers.dev URL.
 *
 * Why this exists: the bucket's public r2.dev URL is rate-limited by Cloudflare
 * and explicitly not for production, which showed up as clips hanging blank for
 * seconds at a time during cooking mode. Reading the bucket through an R2
 * binding avoids that throttle.
 *
 * Range support is the critical part: iOS Safari will not play a video whose
 * server doesn't honor Range requests (it probes with `Range: bytes=0-1` before
 * it will stream anything), so ranges are passed straight through to R2 and a
 * proper 206 + Content-Range is returned.
 */

const CORS = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'GET, HEAD, OPTIONS',
  'access-control-allow-headers': 'range',
  'access-control-expose-headers': 'content-length, content-range, accept-ranges',
};

export default {
  async fetch(request, env) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return new Response('Method Not Allowed', {
        status: 405,
        headers: { allow: 'GET, HEAD, OPTIONS', ...CORS },
      });
    }

    // Key is the path: /<video_id>/<clip>.mp4 — matches the S3 keys the
    // pipeline writes in backend/processors/video.py.
    const url = new URL(request.url);
    const key = decodeURIComponent(url.pathname.replace(/^\/+/, ''));
    if (!key) {
      return new Response('Not Found', { status: 404, headers: CORS });
    }

    // Passing the request headers lets R2 parse Range and If-* itself.
    const object = await env.CLIPS.get(key, {
      range: request.headers,
      onlyIf: request.headers,
    });

    if (object === null) {
      return new Response('Not Found', { status: 404, headers: CORS });
    }

    const headers = new Headers(CORS);
    // Restores content-type and cache-control as stored at upload time
    // (video/mp4, max-age=31536000).
    object.writeHttpMetadata(headers);
    headers.set('etag', object.httpEtag);
    headers.set('accept-ranges', 'bytes');
    if (!headers.has('cache-control')) {
      headers.set('cache-control', 'public, max-age=31536000, immutable');
    }

    // A conditional request that matched: R2 returns metadata with no body.
    if (!object.body) {
      return new Response(null, { status: 304, headers });
    }

    const size = object.size;

    // Only a request that actually asked for a range gets a 206. R2 populates
    // `object.range` even for a full-object read, so keying off it alone would
    // answer every plain GET with a bogus partial response.
    let status = 200;
    if (request.headers.has('range') && object.range) {
      const r = object.range;
      let offset;
      let length;

      // Check the type, not just key presence: R2 hands back a range object
      // whose fields may be present-but-undefined, and `size - undefined` is NaN.
      if (typeof r.suffix === 'number') {
        // e.g. `Range: bytes=-500` — the last N bytes.
        length = Math.min(r.suffix, size);
        offset = size - length;
      } else {
        offset = typeof r.offset === 'number' ? r.offset : 0;
        length = typeof r.length === 'number' ? r.length : size - offset;
      }

      // Clamp to the object so a silly range can't produce a negative or
      // past-the-end Content-Range.
      offset = Math.max(0, Math.min(offset, size));
      length = Math.max(0, Math.min(length, size - offset));

      headers.set('content-range', `bytes ${offset}-${offset + length - 1}/${size}`);
      headers.set('content-length', String(length));
      status = 206;
    } else {
      headers.set('content-length', String(size));
    }

    return new Response(request.method === 'HEAD' ? null : object.body, {
      status,
      headers,
    });
  },
};
