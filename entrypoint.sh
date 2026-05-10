#!/usr/bin/env bash
set -euo pipefail

mkdir -p /models /output /config

echo "[entrypoint] Starting audio2mqtt"
echo "[entrypoint] Web admin will be available on ADMIN_BIND=${ADMIN_BIND:-0.0.0.0:8080}"
echo "[entrypoint] Model downloads are handled from the web admin UI or /api/models endpoints."

exec /usr/local/bin/audio2mqtt
