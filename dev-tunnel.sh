#!/usr/bin/env bash
#
# dev-tunnel.sh — run the full MakerAI stack behind https tunnels.
#
# Like dev-lan.sh, but exposes both frontend and backend over public https URLs
# via cloudflared quick tunnels. Use this when you need the PWA to actually
# install / test service-worker + offline behavior on a phone (LAN http can't,
# because a LAN IP isn't a secure context). Works from anywhere, not just WiFi.
#
# Two tunnels are required: a page served over https cannot call an http
# backend, so the backend gets its own https tunnel and VITE_API_URL is pointed
# at it before the frontend starts.
#
# Requires: cloudflared  (brew install cloudflared)
#
# Usage:  ./dev-tunnel.sh

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "!! cloudflared not found. Install it:" >&2
  echo "     brew install cloudflared" >&2
  exit 1
fi

# --- guard: 8000/5173 must be free (Redis on 6379 is reused, see below) -------
# Without this, docker-compose (or a stale run) holding these ports makes uvicorn
# die and Vite silently fall back to :5174, leaving the tunnels pointed at the
# wrong backend.
for p in 8000 5173; do
  if nc -z localhost "$p" 2>/dev/null; then
    echo "!! Port $p is already in use — likely docker-compose or a stale dev server." >&2
    echo "   Free it first (e.g. 'docker compose down'), then rerun." >&2
    exit 1
  fi
done

TMPDIR_RUN="$(mktemp -d)"
BACK_LOG="$TMPDIR_RUN/backend-tunnel.log"
FRONT_LOG="$TMPDIR_RUN/frontend-tunnel.log"

REDIS_CID=""
PIDS=()

cleanup() {
  echo
  echo "==> shutting down..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  [ -n "$REDIS_CID" ] && docker stop "$REDIS_CID" >/dev/null 2>&1 || true
  rm -rf "$TMPDIR_RUN"
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Start a cloudflared quick tunnel for $1, logging to $2, and echo its public URL.
start_tunnel() {
  local port="$1" log="$2"
  cloudflared tunnel --no-autoupdate --url "http://localhost:$port" >"$log" 2>&1 &
  PIDS+=($!)
  local url="" i=0
  while [ -z "$url" ] && [ "$i" -lt 30 ]; do
    url="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$log" 2>/dev/null | head -1 || true)"
    [ -n "$url" ] && break
    sleep 1
    i=$((i + 1))
  done
  if [ -z "$url" ]; then
    echo "!! Timed out waiting for cloudflared URL (port $port). Log:" >&2
    cat "$log" >&2
    exit 1
  fi
  echo "$url"
}

# --- 1. Redis -----------------------------------------------------------------
if nc -z localhost 6379 2>/dev/null; then
  echo "==> Redis already running on :6379, reusing it."
else
  echo "==> starting Redis (docker)..."
  REDIS_CID="$(docker run -d --rm -p 6379:6379 redis:7-alpine)"
fi

# --- 2. Backend + its tunnel --------------------------------------------------
# CORS_ORIGINS defaults to http://localhost:5173, which would reject the phone's
# tunnel origin. Widen it for this dev launch (credentials are off, so * is safe).
echo "==> starting backend on 0.0.0.0:8000..."
CORS_ORIGINS="*" uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000 &
PIDS+=($!)

echo "==> opening backend tunnel..."
BACKEND_URL="$(start_tunnel 8000 "$BACK_LOG")"
echo "    backend:  $BACKEND_URL"

# --- 3. Frontend + its tunnel -------------------------------------------------
echo "==> starting frontend on :5173 (VITE_API_URL=$BACKEND_URL)..."
(
  cd frontend
  VITE_API_URL="$BACKEND_URL" npm run dev
) &
PIDS+=($!)

echo "==> opening frontend tunnel..."
FRONTEND_URL="$(start_tunnel 5173 "$FRONT_LOG")"

echo
echo "======================================================================"
echo "  Open on your phone:  $FRONTEND_URL"
echo "  (backend API:        $BACKEND_URL )"
echo "======================================================================"
echo "  https + public URL, so the PWA will install. Ctrl-C to stop."
wait
