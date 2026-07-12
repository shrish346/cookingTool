#!/usr/bin/env bash
#
# dev-lan.sh — run the full MakerAI stack locally, reachable from your phone.
#
# Starts Redis (Docker), the FastAPI backend (bound to 0.0.0.0), and the Vite
# frontend with VITE_API_URL pointed at this machine's current LAN IP so a phone
# on the same WiFi can reach both. Ctrl-C tears everything down.
#
# Usage:  ./dev-lan.sh
#
# NOTE: served over plain http on a LAN IP, so the PWA service worker won't
# register on the phone (LAN IPs aren't a secure context). Fine for testing the
# UI/recipe flow. For real PWA install/offline testing, use an https tunnel
# (e.g. `cloudflared tunnel --url http://localhost:5173`).

set -euo pipefail

cd "$(dirname "$0")"

# --- detect LAN IP (first non-loopback IPv4 with a route) ---------------------
LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
[ -z "$LAN_IP" ] && LAN_IP="$(ipconfig getifaddr en1 2>/dev/null || true)"
if [ -z "$LAN_IP" ]; then
  echo "!! Could not auto-detect a LAN IP (are you on WiFi/ethernet?)." >&2
  echo "   Set it manually:  LAN_IP=192.168.x.x ./dev-lan.sh" >&2
  exit 1
fi
# allow override:  LAN_IP=... ./dev-lan.sh
LAN_IP="${LAN_IP_OVERRIDE:-$LAN_IP}"

echo "==> LAN IP: $LAN_IP"
echo "==> Phone:  http://$LAN_IP:5173"
echo

# --- guard: 8000/5173 must be free (Redis on 6379 is reused, see below) -------
# Without this, docker-compose (or a stale run) holding these ports makes uvicorn
# die and Vite silently fall back to :5174, leaving the tunnels/pages pointed at
# the wrong backend.
for p in 8000 5173; do
  if nc -z localhost "$p" 2>/dev/null; then
    echo "!! Port $p is already in use — likely docker-compose or a stale dev server." >&2
    echo "   Free it first (e.g. 'docker compose down'), then rerun." >&2
    exit 1
  fi
done

# --- track child pids + docker container for cleanup --------------------------
REDIS_CID=""
PIDS=()

cleanup() {
  echo
  echo "==> shutting down..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  [ -n "$REDIS_CID" ] && docker stop "$REDIS_CID" >/dev/null 2>&1 || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- 1. Redis -----------------------------------------------------------------
if nc -z localhost 6379 2>/dev/null; then
  echo "==> Redis already running on :6379, reusing it."
else
  echo "==> starting Redis (docker)..."
  REDIS_CID="$(docker run -d --rm -p 6379:6379 redis:7-alpine)"
fi

# --- 2. Backend ---------------------------------------------------------------
# CORS_ORIGINS defaults to http://localhost:5173, which would reject the phone's
# LAN origin. Widen it for this dev launch (credentials are off, so * is safe).
echo "==> starting backend on 0.0.0.0:8000..."
CORS_ORIGINS="*" uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000 &
PIDS+=($!)

# --- 3. Frontend --------------------------------------------------------------
echo "==> starting frontend on 0.0.0.0:5173 (VITE_API_URL=http://$LAN_IP:8000)..."
(
  cd frontend
  VITE_API_URL="http://$LAN_IP:8000" npm run dev
) &
PIDS+=($!)

echo
echo "==> all up. Open http://$LAN_IP:5173 on your phone. Ctrl-C to stop."
wait
