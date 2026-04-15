#!/bin/bash
# FauxPix — One-command startup. Run this from the Deepfake-video folder.
# Usage: bash start.sh
# With Groq key: GROQ_API_KEY=gsk_xxx bash start.sh

set -e
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=8001
FRONTEND_PORT=5173

echo ""
echo "══════════════════════════════════════════"
echo "  FauxPix Video Deepfake Detector"
echo "══════════════════════════════════════════"
echo ""

# ── Kill anything on our ports ───────────────
for PORT in $BACKEND_PORT $FRONTEND_PORT; do
  PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "→ Killing process on port $PORT..."
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    sleep 0.5
  fi
done

# ── Pull latest code ─────────────────────────
echo "→ Pulling latest code..."
cd "$REPO_DIR"
git pull --rebase 2>/dev/null || true

# ── Fix App.jsx port (always ensure 8001) ────
sed -i '' "s|http://localhost:8000|http://localhost:$BACKEND_PORT|g" frontend/src/App.jsx 2>/dev/null || true
sed -i '' "s|http://localhost:8002|http://localhost:$BACKEND_PORT|g" frontend/src/App.jsx 2>/dev/null || true

# ── Install Python deps ───────────────────────
echo "→ Installing Python dependencies..."
cd "$REPO_DIR/backend"
pip3 install -r requirements.txt -q 2>/dev/null || pip install -r requirements.txt -q 2>/dev/null

# ── Install Node deps ─────────────────────────
echo "→ Installing Node dependencies..."
cd "$REPO_DIR/frontend"
npm install --silent 2>/dev/null

# ── Start backend ─────────────────────────────
echo ""
echo "→ Starting backend on http://localhost:$BACKEND_PORT"
cd "$REPO_DIR/backend"
if [ -n "$GROQ_API_KEY" ]; then
  echo "  ✓ Groq key detected — all 7 video signals + forensic report active"
  GROQ_API_KEY="$GROQ_API_KEY" python3 -m uvicorn main:app --port $BACKEND_PORT --log-level warning &
else
  echo "  ℹ No Groq key — 6 signals active (paste key in UI for Signal 6)"
  python3 -m uvicorn main:app --port $BACKEND_PORT --log-level warning &
fi
BACKEND_PID=$!

# ── Wait for backend to be ready ─────────────
echo "  Waiting for backend..."
for i in $(seq 1 20); do
  if curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1; then
    echo "  ✓ Backend ready"
    break
  fi
  sleep 0.5
done

# ── Start frontend ────────────────────────────
echo "→ Starting frontend on http://localhost:$FRONTEND_PORT"
cd "$REPO_DIR/frontend"
npm run dev --silent &
FRONTEND_PID=$!

echo ""
echo "══════════════════════════════════════════"
echo "  ✅ FauxPix is running!"
echo ""
echo "  Open: http://localhost:$FRONTEND_PORT"
echo ""
echo "  Drop any .mp4 / .mov / .avi into the UI"
echo "  Paste Groq key in UI for phoneme-viseme"
echo "  Press Ctrl+C to stop everything"
echo "══════════════════════════════════════════"
echo ""

# ── Wait and cleanup on Ctrl+C ───────────────
trap "echo ''; echo 'Stopping FauxPix...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
